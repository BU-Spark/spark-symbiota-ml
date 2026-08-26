"""Can each pipeline supply the other's confidence signal?

The Azure pipeline (Azure OCR -> gpt-4o-mini) and the Anthropic pipeline (image ->
Claude) share no inputs: one reads OCR text and never sees the sheet, the other reads
the sheet and never sees the OCR. That independence is what makes one useful as a
reference check on the other.

Three measurements, all off cached results:

  A  Claude as the Azure pipeline's independent read, replacing the gpt-4o-mini
     vision call that currently fills that slot. Same model family as the extractor
     means shared failure modes; Claude is cross-vendor and cross-modal.
  B  Azure as the Anthropic pipeline's reference, for the three fields that have no
     free signal (eventDate, recordedBy, barcode).
  C  Where the two pipelines disagree on a value, which one is right.

    python nbs/cross_pipeline.py

Free: reads caches only, no API calls.
"""
import argparse
import json
import os
import re
import sys
from difflib import SequenceMatcher

sys.path.insert(0, "transcription")
sys.path.insert(0, os.path.dirname(__file__))
from confidence import (GbifTaxonMatcher, barcode_confidence,  # noqa: E402
                        eventdate_confidence, location_confidence,
                        recordedby_confidence, scientificname_confidence,
                        _tokenize)
from sonnet5_hallucination import correct, get_field, is_unknown, load_gt  # noqa: E402

OCR = "transcription/results/ocr_cache/azure"
SC = "transcription/results/self_consistency_cache/azure"
VIS = "transcription/results/vision_cache"
CLAUDE = "transcription/results/model_cache/claude-sonnet-4-6"
GT_DIR = "transcription/data/gbif-ne-500"
GT_FILE = {"scientificName": "taxons.txt", "eventDate": "dates.txt",
           "recordedBy": "collectors.txt", "barcode": "catalognumbers.txt"}
FIELDS = ["scientificName", "eventDate", "recordedBy", "barcode", "location"]


def ratio(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def flat_loc(v):
    if isinstance(v, dict):
        def c(k):
            x = v.get(k)
            return "" if (x is None or str(x).strip().upper() in ("", "UNKNOWN")) else str(x).strip()
        return ", ".join(p for p in (c("locality"), c("county"), c("stateProvince")) if p)
    return "" if v is None else str(v)


def admin_accuracy(pred, a):
    parts = [a.get("stateProvince", ""),
             re.sub(r"\bcounty\b", "", a.get("county", ""), flags=re.I),
             a.get("locality", "")]
    g = _tokenize(" ".join(parts))
    if not g:
        return None
    p = _tokenize(pred)
    return sum(1 for t in g if any(ratio(t, q) >= 0.85 for q in p)) / len(g)


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def load():
    """One row per specimen present in every cache."""
    admin = json.load(open(os.path.join(GT_DIR, "gbif_admin_location.json"), encoding="utf-8"))
    gts = {f: load_gt(v) for f, v in GT_FILE.items()}
    rows = []
    for fn in sorted(os.listdir(OCR)):
        if not fn.endswith(".json"):
            continue
        occid = fn[:-5]
        cp = os.path.join(CLAUDE, fn)
        if not os.path.exists(cp):
            continue
        az = json.load(open(os.path.join(OCR, fn), encoding="utf-8"))
        cl = json.load(open(cp, encoding="utf-8"))
        if not isinstance(cl, dict):
            continue
        vp, sp = os.path.join(VIS, fn), os.path.join(SC, fn)
        vis = json.load(open(vp, encoding="utf-8")) if os.path.exists(vp) else {}
        sc = json.load(open(sp, encoding="utf-8")) if os.path.exists(sp) else None
        azf = dict(az["fields"])
        azf["location"] = flat_loc(azf.get("location"))
        if isinstance(vis, dict) and "location" in vis:
            vis = dict(vis)
            vis["location"] = flat_loc(vis["location"])
        clf = {f: get_field(cl, f) for f in FIELDS}
        clf["scientificName"] = cl.get("verbatimScientificName") or clf["scientificName"]
        rows.append({"occid": occid, "az": azf, "cl": clf, "words": az["words"],
                     "vis": vis or {}, "sc": sc, "admin": admin.get(occid), "gt": gts})
    return rows


def score(field, pred, row):
    if field == "location":
        return admin_accuracy(pred, row["admin"]) if row["admin"] else None
    return correct(field, pred, row["gt"][field].get(row["occid"]))


def conf(field, value, words, sc, vision, matcher):
    if field == "scientificName":
        return scientificname_confidence(value, words, matcher, vision)
    if field == "location":
        return location_confidence(value, words, vision)
    if field == "recordedBy":
        return recordedby_confidence(value, words, vision)
    if field == "barcode":
        return barcode_confidence(value, words, [s.get("barcode", "") for s in sc] if sc else None)
    if field == "eventDate":
        return eventdate_confidence(value, words, [s.get("eventDate", "") for s in sc] if sc else None,
                                    vision)
    return None


def rho_for(rows, field, primary, words_from, sc_from, vision_from, matcher):
    xs, ys = [], []
    for r in rows:
        v = r[primary].get(field, "")
        if is_unknown(v):
            continue
        words = r["words"] if words_from else []
        sc = r["sc"] if sc_from else None
        vision = None
        if vision_from == "gpt4o":
            vision = r["vis"].get(field)
        elif vision_from == "claude":
            vision = r["cl"].get(field)
        elif vision_from == "azure":
            vision = r["az"].get(field)
        c = conf(field, v, words, sc, vision, matcher)
        t = score(field, v, r)
        if c is None or t is None:
            continue
        xs.append(c)
        ys.append(t if field == "location" else (1 if t else 0))
    return (spearman(xs, ys), len(xs)) if len(xs) >= 10 else (float("nan"), len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    rows = load()
    matcher = GbifTaxonMatcher()
    print(f"{len(rows)} specimens present in both pipelines' caches\n")

    print("=" * 74)
    print("  A. Claude as the Azure pipeline's independent read")
    print("=" * 74)
    print("  Azure values scored; the reference read is swapped.\n")
    print(f"  {'field':16} {'no reference':>13} {'gpt-4o-mini':>13} {'Claude 4.6':>13}")
    for f in FIELDS:
        a, _ = rho_for(rows, f, "az", True, True, None, matcher)
        b, _ = rho_for(rows, f, "az", True, True, "gpt4o", matcher)
        c, n = rho_for(rows, f, "az", True, True, "claude", matcher)
        print(f"  {f:16} {a:>+13.3f} {b:>+13.3f} {c:>+13.3f}   (n={n})")

    print("\n" + "=" * 74)
    print("  B. Azure as the Anthropic pipeline's reference")
    print("=" * 74)
    print("  Claude values scored. 'free' uses no Azure input at all.\n")
    print(f"  {'field':16} {'free':>13} {'+ Azure OCR':>13} {'+ Azure read':>13}")
    for f in FIELDS:
        a, _ = rho_for(rows, f, "cl", False, False, None, matcher)
        b, _ = rho_for(rows, f, "cl", True, True, None, matcher)
        c, n = rho_for(rows, f, "cl", True, True, "azure", matcher)
        print(f"  {f:16} {a:>+13.3f} {b:>+13.3f} {c:>+13.3f}   (n={n})")

    print("\n" + "=" * 74)
    print("  C. Where the pipelines disagree, which is right")
    print("=" * 74)
    print(f"  {'field':16} {'agree%':>7} {'both right':>11} {'Azure right':>12} "
          f"{'Claude right':>13} {'both wrong':>11}")
    for f in FIELDS:
        agree = az_only = cl_only = both_ok = both_bad = n = 0
        for r in rows:
            a, c = r["az"].get(f, ""), r["cl"].get(f, "")
            if is_unknown(a) or is_unknown(c):
                continue
            ta, tc = score(f, a, r), score(f, c, r)
            if ta is None or tc is None:
                continue
            if f == "location":
                ta, tc = ta >= 0.5, tc >= 0.5
            n += 1
            same = (ratio(a, c) >= 0.9) if f != "location" else (ratio(a, c) >= 0.7)
            if same:
                agree += 1
            if ta and tc:
                both_ok += 1
            elif ta and not tc:
                az_only += 1
            elif tc and not ta:
                cl_only += 1
            else:
                both_bad += 1
        if not n:
            continue
        print(f"  {f:16} {agree/n:>6.0%} {both_ok/n:>11.0%} {az_only/n:>12.0%} "
              f"{cl_only/n:>13.0%} {both_bad/n:>11.0%}   (n={n})")


if __name__ == "__main__":
    main()
