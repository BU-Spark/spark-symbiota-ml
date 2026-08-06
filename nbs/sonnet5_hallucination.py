"""Hallucination + accuracy comparison: Claude Sonnet 5 vs the shipped gpt-4o-mini,
on the same gbif-ne-500 specimens.

Hallucination here means the model emitted a confident, well-formed value that is not
real, as distinct from abstaining (UNKNOWN) or being beaten by a GBIF re-determination.
Three measures:

  1. scientificName invented-name rate: non-UNKNOWN species that GBIF's backbone cannot
     match at all (matchType NONE). A non-existent binomial like "Euphrasia kurtkoviana"
     is unambiguous fabrication.
  2. eventDate impossible-year rate: a parseable year outside 1600..now.
  3. ungrounded-value rate: non-UNKNOWN field value whose tokens do not appear in the
     Azure OCR of the same label (proxy for invented text; a vision model can legitimately
     read past OCR, so treat this as an upper bound, not proof).

Also reports plain accuracy vs GBIF GT (lenient) so wrong is separated from abstain.

    python nbs/sonnet5_hallucination.py
"""
import json
import os
import re
import sys
from difflib import SequenceMatcher

sys.path.insert(0, "transcription")
from confidence import GbifTaxonMatcher, _tokenize, _digits, _event_year  # noqa: E402

GT = "transcription/data/gbif-ne-500"
OCR = "transcription/results/ocr_cache/azure"
S5 = "transcription/results/sonnet5_cache"
FIELDS = ["scientificName", "eventDate", "recordedBy", "barcode", "location"]


def load_gt(fn):
    d = {}
    p = os.path.join(GT, fn)
    if os.path.exists(p):
        for line in open(p, encoding="utf-8", errors="ignore"):
            if ":" in line:
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
    return d


def ratio(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def binomial(s):
    return " ".join([t for t in _tokenize(s) if len(t) >= 3][:2])


def is_unknown(v):
    return v is None or str(v).strip() == "" or str(v).strip().upper() == "UNKNOWN"


def flatten_loc(v):
    if isinstance(v, dict):
        parts = [str(v.get(k, "")) for k in ("locality", "county", "stateProvince")]
        return ", ".join(p for p in parts if p and p.upper() != "UNKNOWN")
    return "" if v is None else str(v)


def correct(field, pred, gt):
    if is_unknown(pred) or is_unknown(gt):
        return None
    if field == "scientificName":
        return ratio(binomial(pred), binomial(gt)) >= 0.9
    if field == "barcode":
        return _digits(pred).lstrip("0") == _digits(gt).lstrip("0")
    if field == "eventDate":
        return _event_year(pred) == _event_year(gt)
    if field == "recordedBy":
        p = set(t for t in _tokenize(pred) if len(t) >= 3)
        g = [t for t in _tokenize(gt) if len(t) >= 3]
        return (sum(any(ratio(t, q) >= 0.85 for q in p) for t in g) / len(g) >= 0.5) if g else None
    if field == "location":  # any GT admin token present
        g = [t for t in _tokenize(gt) if len(t) >= 3]
        p = set(t for t in _tokenize(pred) if len(t) >= 3)
        return (sum(any(ratio(t, q) >= 0.85 for q in p) for t in g) / len(g) >= 0.4) if g else None
    return None


def grounded(value, ocr_words):
    # fraction of value tokens present in OCR; grounded if >= 0.5
    toks = [t for t in _tokenize(value) if len(t) >= 3]
    if not toks:
        return None
    ocr = set()
    for w in ocr_words:
        for t in _tokenize(w.get("content", "")):
            ocr.add(t)
    hit = sum(1 for t in toks if any(ratio(t, o) >= 0.85 for o in ocr))
    return hit / len(toks) >= 0.5


# The prompt does not pin the JSON key names, so models use equivalent Darwin Core
# synonyms. Accept them, so a model is scored on its transcription rather than on
# its key naming.
ALIASES = {
    "location":       ("location", "locality"),
    "barcode":        ("barcode", "catalogNumber"),
    "recordedBy":     ("recordedBy", "collector"),
    "scientificName": ("scientificName", "taxon", "species"),
    "eventDate":      ("eventDate", "date"),
}


def get_field(data, field):
    v = ""
    for key in ALIASES.get(field, (field,)):
        if isinstance(data, dict) and data.get(key) not in (None, ""):
            v = data[key]
            break
    if field == "location":
        return flatten_loc(v)
    return v


def score(name, get_data, occids, gts, matcher):
    n = len(occids)
    stats = {f: {"correct": 0, "wrong": 0, "unknown": 0, "scored": 0} for f in FIELDS}
    invented_sci = 0
    sci_nonunknown = 0
    bad_year = 0
    date_nonunknown = 0
    ungrounded = 0
    grounded_denom = 0
    parse_err = 0
    for occid in occids:
        data = get_data(occid)
        if not isinstance(data, dict) or "_error" in data or "_parse_error" in data:
            parse_err += 1
            continue
        ocr = json.load(open(os.path.join(OCR, occid + ".json"), encoding="utf-8"))
        words = ocr["words"]
        for f in FIELDS:
            v = get_field(data, f)
            if is_unknown(v):
                stats[f]["unknown"] += 1
                continue
            gt = gts[f].get(occid)
            ok = correct(f, v, gt)
            if ok is not None:
                stats[f]["scored"] += 1
                stats[f]["correct" if ok else "wrong"] += 1
            # grounding (verbatim fields only)
            if f in ("recordedBy", "barcode", "scientificName"):
                g = grounded(v, words)
                if g is not None:
                    grounded_denom += 1
                    if not g:
                        ungrounded += 1
        # invented species
        sci = get_field(data, "scientificName")
        if not is_unknown(sci):
            sci_nonunknown += 1
            if matcher(sci) is None:  # GBIF matchType NONE
                invented_sci += 1
        # impossible year
        ev = get_field(data, "eventDate")
        if not is_unknown(ev):
            date_nonunknown += 1
            y = _event_year(ev)
            if y is not None and not (1600 <= y <= 2026):
                bad_year += 1
    print(f"\n===== {name}  (n={n}, parse/api errors={parse_err}) =====")
    print(f"  {'field':15} {'acc':>6} {'wrong':>6} {'unknown':>8}  (scored)")
    for f in FIELDS:
        s = stats[f]
        acc = s["correct"] / s["scored"] if s["scored"] else float("nan")
        print(f"  {f:15} {acc:6.1%} {s['wrong']:6} {s['unknown']:8}  ({s['scored']})")
    print("  ---- hallucination measures ----")
    print(f"  scientificName invented (GBIF NONE): {invented_sci}/{sci_nonunknown} = "
          f"{(invented_sci/sci_nonunknown if sci_nonunknown else float('nan')):.1%}")
    print(f"  eventDate impossible year:           {bad_year}/{date_nonunknown} = "
          f"{(bad_year/date_nonunknown if date_nonunknown else float('nan')):.1%}")
    print(f"  ungrounded verbatim value (vs OCR):  {ungrounded}/{grounded_denom} = "
          f"{(ungrounded/grounded_denom if grounded_denom else float('nan')):.1%}")
    return stats


def main():
    gts = {"scientificName": load_gt("taxons.txt"), "eventDate": load_gt("dates.txt"),
           "recordedBy": load_gt("collectors.txt"), "barcode": load_gt("catalognumbers.txt"),
           "location": load_gt("localities.txt")}
    matcher = GbifTaxonMatcher()
    occids = sorted(f[:-5] for f in os.listdir(S5) if f.endswith(".json"))
    print(f"Sonnet 5 outputs cached: {len(occids)}")

    def s5(occid):
        return json.load(open(os.path.join(S5, occid + ".json"), encoding="utf-8"))

    def mini(occid):
        return json.load(open(os.path.join(OCR, occid + ".json"), encoding="utf-8"))["fields"]

    score("Claude Sonnet 5 (image-native)", s5, occids, gts, matcher)
    score("gpt-4o-mini (shipped, Azure OCR + LLM)", mini, occids, gts, matcher)


if __name__ == "__main__":
    main()
