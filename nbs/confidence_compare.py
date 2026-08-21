"""Confidence discrimination for all three pipelines on the same specimens.

Each pipeline is given the signals it actually ships. Reports Spearman rho, AUC,
calibration error, and auto-accept coverage/precision at 0.90.

    python nbs/confidence_compare.py

Free: reads cached outputs.
"""
import json
import os
import re
import sys
from difflib import SequenceMatcher

import pandas as pd

sys.path.insert(0, "transcription")
sys.path.insert(0, os.path.dirname(__file__))
import claude_sonnet as cs  # noqa: E402
from confidence import build_confidence, GbifTaxonMatcher, _tokenize  # noqa: E402
from sonnet5_hallucination import correct, get_field, is_unknown, load_gt  # noqa: E402

GT_DIR = os.environ.get("HERBARIA_GT_DIR", "transcription/data/gbif-ne-500")
AZURE_OCR = "transcription/results/ocr_cache/azure"
GOOGLE_OCR = "transcription/results/ocr_cache/google"
SC = "transcription/results/self_consistency_cache/azure"
GOOGLE_SC = "transcription/results/self_consistency_cache/google"
VIS = "transcription/results/vision_cache"
SONNET5 = "transcription/results/model_cache/claude-sonnet-5"
CHECKER = "transcription/results/model_cache/claude-sonnet-4-6"

FIELDS = ["scientificName", "eventDate", "recordedBy", "barcode", "location"]
GT_FILE = {"scientificName": "taxons.txt", "eventDate": "dates.txt",
           "recordedBy": "collectors.txt", "barcode": "catalognumbers.txt",
           "location": "localities.txt"}


def load(root, occid):
    p = os.path.join(root, f"{occid}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def auc(pairs):
    """P(a correct field outscores a wrong one), ties at 0.5. 0.5 = coin flip."""
    pos = [s for s, ok in pairs if ok]
    neg = [s for s, ok in pairs if not ok]
    if not pos or not neg:
        return None
    ranked = sorted(pairs, key=lambda p: p[0])
    i, rank_sum = 0, 0.0
    while i < len(ranked):                      # average ranks within a tie group
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += sum(avg for k in range(i, j) if ranked[k][1])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ece(pairs, bins=5):
    """Mean gap between stated confidence and actual accuracy. 0 is perfect."""
    if not pairs:
        return None
    buckets = {}
    for s, ok in pairs:
        buckets.setdefault(min(int(s * bins), bins - 1), []).append((s, ok))
    n = sum(len(b) for b in buckets.values())
    return sum(len(b) / n * abs(sum(s for s, _ in b) / len(b)
                                - sum(o for _, o in b) / len(b))
               for b in buckets.values())


def at_cutoff(pairs, cut=0.90):
    """(coverage, precision) if fields at or above `cut` skipped review."""
    kept = [ok for s, ok in pairs if s >= cut]
    if not kept:
        return 0.0, None
    return len(kept) / len(pairs), sum(kept) / len(kept)


def admin_accuracy(pred, a):
    # location correctness against GBIF's structured admin fields.
    parts = [a.get("stateProvince", ""),
             re.sub(r"\bcounty\b", "", a.get("county", ""), flags=re.I),
             a.get("locality", "")]
    g = _tokenize(" ".join(parts))
    if not g:
        return None
    p = _tokenize(pred)
    hits = sum(1 for t in g
               if any(SequenceMatcher(None, t.lower(), q.lower()).ratio() >= 0.85 for q in p))
    return hits / len(g)


def azure_conf(occid, matcher):
    rec = load(AZURE_OCR, occid)
    if not rec:
        return None, None
    conf = build_confidence(rec["fields"], rec["words"],
                            sc_samples=load(SC, occid),
                            vision_fields=load(VIS, occid),
                            taxon_matcher=matcher)
    return rec["fields"], conf


def google_conf(occid, matcher):
    # Baseline: OCR grounding + GBIF only.
    rec = load(GOOGLE_OCR, occid)
    if not rec:
        return None, None
    return rec["fields"], build_confidence(rec["fields"], rec["words"], taxon_matcher=matcher)


def anthropic_conf(occid, matcher):
    rec = load(SONNET5, occid)
    if not isinstance(rec, dict):
        return None, None
    raw = {f: get_field(rec, f) for f in FIELDS}
    raw["scientificName"] = rec.get("verbatimScientificName") or raw["scientificName"]
    azure = load(AZURE_OCR, occid)
    return raw, cs.build_confidence(raw, load(CHECKER, occid),
                                    azure["fields"] if azure else None)


def google_vis_conf(occid, matcher):
    # What google_vision.py ships: grounding + GBIF + vision.
    rec = load(GOOGLE_OCR, occid)
    if not rec:
        return None, None
    return rec["fields"], build_confidence(rec["fields"], rec["words"],
                                           vision_fields=load(VIS, occid),
                                           taxon_matcher=matcher,
                                           calibration="calibration_google")


def google_full_conf(occid, matcher):
    # Vision plus self-consistency, for comparison; not shipped.
    rec = load(GOOGLE_OCR, occid)
    if not rec:
        return None, None
    return rec["fields"], build_confidence(rec["fields"], rec["words"],
                                           sc_samples=load(GOOGLE_SC, occid),
                                           vision_fields=load(VIS, occid),
                                           taxon_matcher=matcher,
                                           calibration="calibration_google")


PIPELINES = {"Azure": azure_conf, "Google": google_conf,
             "Google+vis": google_vis_conf, "Google+all": google_full_conf,
             "Anthropic": anthropic_conf}


def main():
    occids = sorted(f[:-5] for f in os.listdir(GOOGLE_OCR) if f.endswith(".json"))
    gts = {f: load_gt(v) for f, v in GT_FILE.items()}
    admin = json.load(open(os.path.join(GT_DIR, "gbif_admin_location.json"), encoding="utf-8"))
    matcher = GbifTaxonMatcher()
    print(f"scored on {len(occids)} specimens\n")

    pairs = {p: {f: [] for f in FIELDS} for p in PIPELINES}
    for occid in occids:
        for pname, fn in PIPELINES.items():
            fields, conf = fn(occid, matcher)
            if not fields:
                continue
            for f in FIELDS:
                c, pred = conf.get(f), fields.get(f, "")
                if c is None or is_unknown(pred):
                    continue
                if f == "location":
                    acc = admin_accuracy(pred, admin[occid]) if occid in admin else None
                    ok = None if acc is None else acc >= 0.5
                else:
                    gt = gts[f].get(occid)
                    if not gt:
                        continue
                    shipped = pred
                    if f == "scientificName":
                        fixed, _ = matcher.correct(pred)
                        shipped = fixed or pred
                    ok = correct(f, shipped, gt)
                if ok is not None:
                    pairs[pname][f].append((c, 1 if ok else 0))

    names = list(PIPELINES)
    print(f"  {'field':16}" + "".join(f"{n:>22}" for n in names))
    print(f"  {'':16}" + "".join(f"{'rho      n':>22}" for _ in names))
    for f in FIELDS:
        row = f"  {f:16}"
        for n in names:
            p = pairs[n][f]
            if len(p) < 20 or len({c for c, _ in p}) < 2:
                row += f"{'--':>22}"
                continue
            rho = pd.Series([c for c, _ in p]).corr(
                pd.Series([o for _, o in p]), method="spearman")
            row += f"{rho:>+16.3f}{len(p):>6}"
        print(row)

    print("\n  rho is the score's ability to rank right above wrong.")
    print("  Below about +0.3 the score is too weak to route review on.")

    print(f"\n\n===== AUC: P(a correct field outscores a wrong one) "
          f"-- 0.50 = coin flip =====")
    print(f"  {'field':16}" + "".join(f"{n:>12}" for n in names))
    for f in FIELDS:
        row = f"  {f:16}"
        for n in names:
            a = auc(pairs[n][f])
            row += f"{a:>12.3f}" if a is not None else f"{'--':>12}"
        print(row)
    print(f"  {'MEAN':16}" + "".join(
        f"{sum(v for v in (auc(pairs[n][f]) for f in FIELDS) if v is not None) / len(FIELDS):>12.3f}"
        for n in names))

    print(f"\n\n===== calibration error: |stated confidence - actual accuracy| "
          f"-- 0 is perfect =====")
    print(f"  {'field':16}" + "".join(f"{n:>12}" for n in names))
    for f in FIELDS:
        row = f"  {f:16}"
        for n in names:
            e = ece(pairs[n][f])
            row += f"{e:>12.3f}" if e is not None else f"{'--':>12}"
        print(row)

    print(f"\n\n===== auto-accept at confidence >= 0.90 =====")
    print(f"  {'field':16}" + "".join(f"{n:>22}" for n in names))
    print(f"  {'':16}" + "".join(f"{'skipped   correct':>22}" for _ in names))
    for f in FIELDS:
        row = f"  {f:16}"
        for n in names:
            cov, prec = at_cutoff(pairs[n][f])
            row += f"{cov:>15.0%}{prec:>7.0%}" if prec is not None else f"{'--':>22}"
        print(row)


if __name__ == "__main__":
    main()
