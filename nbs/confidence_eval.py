"""Validate that the OCR-derived confidence scores predict field accuracy.

Two modes:

  # Offline (free): score confidence from a cached OCR+LLM dump, then analyze.
  python nbs/confidence_eval.py --cache-dir transcription/results/ocr_cache/azure

  # Live (incurs API cost): run the pipeline end-to-end, then analyze.
  python nbs/confidence_eval.py --pipeline azure

Build the cache once with nbs/ocr_cache.py so the logic in
transcription/confidence.py can be tuned offline without re-calling the APIs.

For the three fields with ground truth (scientificName, location, eventDate) it
scores each prediction against ground truth with a 0-1 string ratio, then per
field reports:
  - Spearman correlation (confidence vs. accuracy),
  - a reliability table (mean accuracy within each confidence bin), and
  - a threshold/coverage table (what auto-accepting at each cutoff would yield).
"""

import argparse
import json
import os
import sys
from difflib import SequenceMatcher

import pandas as pd

SAMPLE_DIR = "transcription/data/new-england-samples/output"

# field -> ground-truth file (occid: value per line)
FIELD_TO_GT = {
    "scientificName": "taxons.txt",
    "location": "localities.txt",
    "eventDate": "dates.txt",
    # GBIF-sourced ground truth (via occid) for the verbatim fields.
    "recordedBy": "collectors.txt",
    "barcode": "catalognumbers.txt",
    "institutionCode": "institutions.txt",
}

# Confidence variants that may be present in a row, in report order.
# coverage is the headline (grounding) score; ocr_read is the OCR-stage
# confidence, reported for comparison only.
CONF_KINDS = ("coverage", "ocr_read")
# Left-closed bins for the reliability table; 1.01 keeps 1.0 in the last bin.
RELIABILITY_BINS = [0.0, 0.5, 0.7, 0.85, 0.95, 1.01]


def load_ground_truth(gt_dir: str) -> dict:
    # {field: {occid: value}}
    gt = {}
    for field, fname in FIELD_TO_GT.items():
        path = os.path.join(gt_dir, fname)
        mapping = {}
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if ":" in line:
                    occid, val = line.split(":", 1)
                    mapping[occid.strip()] = val.strip()
        gt[field] = mapping
    return gt


def ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def _score_row(occid: str, fields: dict, conf: dict, gt: dict):
    # Build one result row: per-field accuracy vs GT and whatever confidence
    # variants are available. `conf[field]` is either a plain float (OCR-only)
    # or a {"ocr":..., "llm":...} dict. Returns None when the image has no GT.
    row = {"occid": occid}
    has_gt = False
    for field in FIELD_TO_GT:
        truth = gt[field].get(occid)
        if truth is None:
            continue
        has_gt = True
        row[f"acc_{field}"] = ratio(fields.get(field, ""), truth)
        c = conf.get(field)
        if isinstance(c, dict):
            row[f"conf_coverage_{field}"] = c.get("coverage")
            row[f"conf_ocr_read_{field}"] = c.get("ocr_read")
        else:
            row[f"conf_coverage_{field}"] = c
    return row if has_gt else None


def run_from_cache(cache_dir: str, gt_dir: str, out_csv: str):
    # Score confidence offline from a cached OCR+LLM dump -- no API calls.
    sys.path.insert(0, "transcription")
    from confidence import build_confidence

    gt = load_ground_truth(gt_dir)
    rows = []
    for fn in sorted(os.listdir(cache_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(cache_dir, fn), encoding="utf-8") as f:
            rec = json.load(f)
        # detail=True so both coverage and ocr_read are available for analysis.
        conf = build_confidence(rec["fields"], rec["words"], detail=True)
        row = _score_row(rec["occid"], rec["fields"], conf, gt)
        if row:
            rows.append(row)
        print(f"  scored {rec['occid']}")
    _finish(rows, out_csv)


def run(pipeline: str, image_dir: str, gt_dir: str, out_csv: str):
    # Run the pipeline end-to-end (API cost) and analyze. Turns on detail mode
    # so both coverage and ocr_read are available for comparison.
    os.environ["CONFIDENCE_DETAIL"] = "1"
    sys.path.insert(0, "transcription")
    if pipeline == "azure":
        from doc_intelligence import run_doc_intell_pipeline as run_pipeline
    elif pipeline == "google":
        from google_vision import run_google_vision_pipeline as run_pipeline
    elif pipeline == "tesseract":
        from tesseract_pipeline import run_tesseract_pipeline as run_pipeline
    else:
        raise SystemExit(f"unknown pipeline: {pipeline}")

    gt = load_ground_truth(gt_dir)
    rows = []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        occid = os.path.splitext(fname)[0]
        result = run_pipeline(os.path.join(image_dir, fname))
        try:
            data = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            print(f"  skipped {fname}: {str(result)[:80]}")
            continue
        row = _score_row(occid, data, data.get("confidence", {}), gt)
        if row:
            rows.append(row)
        print(f"  processed {fname}")
    _finish(rows, out_csv)


def _finish(rows: list, out_csv: str):
    df = pd.DataFrame(rows)
    # Drop rows where no field accuracy was computed (failed/blank pipeline run).
    acc_cols = [f"acc_{f}" for f in FIELD_TO_GT if f"acc_{f}" in df.columns]
    if acc_cols:
        df = df.dropna(subset=acc_cols, how="all").reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} rows)\n")
    analyze(df)


def analyze(df: pd.DataFrame):
    for field in FIELD_TO_GT:
        acc_col = f"acc_{field}"
        if acc_col not in df.columns:
            continue
        kinds = [k for k in CONF_KINDS if f"conf_{k}_{field}" in df.columns]
        print(f"=== {field} ===")
        baseline = df[acc_col].mean()
        for kind in kinds:
            conf_col = f"conf_{kind}_{field}"
            sub = df[[conf_col, acc_col]].dropna()
            if len(sub) < 3:
                print(f"  {kind:3}: too few rows ({len(sub)})")
                continue
            # Spearman = Pearson of ranks (avoids a scipy dependency)
            rho = sub[conf_col].rank().corr(sub[acc_col].rank())
            print(f"  {kind:3}: Spearman rho={rho:+.3f} (n={len(sub)})")
            # Reliability: is mean accuracy actually higher in higher conf bins?
            binned = pd.cut(sub[conf_col], RELIABILITY_BINS, right=False)
            for interval, grp in sub.groupby(binned, observed=True):
                print(f"        conf [{interval.left:.2f},{interval.right:.2f}): "
                      f"mean acc={grp[acc_col].mean():.3f}  n={len(grp)}")
            # Threshold/coverage: what auto-accepting at each cutoff would yield.
            for cut in (0.6, 0.7, 0.8, 0.9):
                kept = sub[sub[conf_col] >= cut]
                cov = len(kept) / len(sub)
                acc = kept[acc_col].mean() if len(kept) else float("nan")
                print(f"        >= {cut:.1f}: mean acc={acc:.3f}  coverage={cov:.0%}")
        print(f"  baseline mean acc (no filter) = {baseline:.3f}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=["azure", "google", "tesseract"], default="azure")
    ap.add_argument("--cache-dir",
                    help="score confidence offline from an OCR cache dir (no API calls)")
    ap.add_argument("--image-dir", default=SAMPLE_DIR)
    ap.add_argument("--gt-dir", default=SAMPLE_DIR)
    ap.add_argument("--out", default="transcription/results/confidence_eval.csv")
    ap.add_argument("--analyze-csv", help="skip scoring; analyze an existing CSV")
    args = ap.parse_args()
    if args.analyze_csv:
        analyze(pd.read_csv(args.analyze_csv))
    elif args.cache_dir:
        run_from_cache(args.cache_dir, args.gt_dir, args.out)
    else:
        run(args.pipeline, args.image_dir, args.gt_dir, args.out)
