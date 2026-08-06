"""Quantify the missed-error risk: at each confidence cutoff, how much of the data
would be auto-skipped (coverage) and what fraction of those skipped items are
actually WRONG (the errors a reviewer would miss). Uses lenient correctness
metrics so scoring artifacts (leading zeros, "Last, First" name order) don't
inflate the error rate.

    python nbs/confidence_threshold_table.py --gt-dir transcription/data/gbif-ne-500
"""

import argparse
import json
import os
import sys
from difflib import SequenceMatcher

sys.path.insert(0, "transcription")
sys.path.insert(0, os.path.dirname(__file__))
from confidence import build_confidence, GbifTaxonMatcher, _tokenize, _digits, _event_year  # noqa: E402

OCR_CACHE = "transcription/results/ocr_cache/azure"
SC_CACHE = "transcription/results/self_consistency_cache/azure"
VIS_CACHE = "transcription/results/vision_cache"


def ratio(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def load_gt(gt_dir, fn):
    d = {}
    for line in open(os.path.join(gt_dir, fn), encoding="utf-8", errors="ignore"):
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d


def binomial(s):
    return " ".join([t for t in _tokenize(s) if len(t) >= 3][:2])


def correct(field, pred, gt):
    # Lenient: count only real errors, not formatting/order artifacts.
    if field == "scientificName":
        return ratio(binomial(pred), binomial(gt)) >= 0.9
    if field == "barcode":  # ignore leading-zero differences
        return _digits(pred).lstrip("0") == _digits(gt).lstrip("0")
    if field == "eventDate":
        return _event_year(pred) == _event_year(gt)
    if field == "recordedBy":  # order-agnostic: most GT name tokens present in pred
        p = set(t for t in _tokenize(pred) if len(t) >= 3)
        g = [t for t in _tokenize(gt) if len(t) >= 3]
        if not g:
            return None
        return sum(any(ratio(t, q) >= 0.85 for q in p) for t in g) / len(g) >= 0.5
    return None


def run(gt_dir):
    matcher = GbifTaxonMatcher()
    gts = {"scientificName": load_gt(gt_dir, "taxons.txt"),
           "eventDate": load_gt(gt_dir, "dates.txt"),
           "recordedBy": load_gt(gt_dir, "collectors.txt"),
           "barcode": load_gt(gt_dir, "catalognumbers.txt")}

    def load(root, occid):
        p = os.path.join(root, f"{occid}.json")
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

    data = {f: [] for f in gts}  # (conf, correct) per field
    for fn in sorted(os.listdir(OCR_CACHE)):
        if not fn.endswith(".json"):
            continue
        occid = fn[:-5]
        rec = load(OCR_CACHE, occid)
        conf = build_confidence(rec["fields"], rec["words"],
                                sc_samples=load(SC_CACHE, occid),
                                vision_fields=load(VIS_CACHE, occid),
                                taxon_matcher=matcher)
        for field in gts:
            c, pred, gt = conf.get(field), rec["fields"].get(field, ""), gts[field].get(occid)
            if c is None or not gt or not pred or str(pred).upper() == "UNKNOWN":
                continue
            ok = correct(field, pred, gt)
            if ok is not None:
                data[field].append((c, ok))

    cutoffs = [0.8, 0.9, 0.95, 1.0]
    print(f"\n  Missed-error risk if you auto-skip review above a confidence cutoff\n")
    for field, rows in data.items():
        n = len(rows)
        print(f"=== {field} (n={n}) ===")
        for cut in cutoffs:
            kept = [ok for c, ok in rows if c >= cut]
            if not kept:
                print(f"   >= {cut:.2f}:  no items")
                continue
            cov = len(kept) / n
            err = 1 - sum(kept) / len(kept)
            print(f"   >= {cut:.2f}:  skips {cov:5.0%} of items,  {err:5.1%} of those are WRONG (missed)")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="transcription/data/gbif-ne-500")
    args = ap.parse_args()
    run(args.gt_dir)
