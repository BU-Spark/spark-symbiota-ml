"""Validate that the new confidence scores predict accuracy.

Runs a pipeline over the New England sample images, then for the three fields
that have ground truth (scientificName, location, eventDate) it:
  - records the per-field confidence (ocr / llm / their min),
  - scores the prediction against ground truth with a 0-1 string ratio,
  - reports Spearman correlation (confidence vs. accuracy) and a threshold table
    showing, at each confidence cutoff, the mean accuracy and coverage of the
    fields that would be auto-accepted.

Usage (from repo root, or inside the container at /app):
    python nbs/confidence_eval.py --pipeline azure
    python nbs/confidence_eval.py --pipeline azure --out results/conf_eval.csv

Stage incurs API cost on the team's keys: one OCR + one GPT call per image.
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
}


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


def run(pipeline: str, image_dir: str, gt_dir: str, out_csv: str):
    os.environ["CONFIDENCE_INCLUDE_LLM"] = "1"
    sys.path.insert(0, "transcription")
    if pipeline == "azure":
        from doc_intelligence import run_doc_intell_pipeline as run_pipeline
    elif pipeline == "google":
        from google_vision import run_google_vision_pipeline as run_pipeline
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

        conf = data.get("confidence", {})
        row = {"occid": occid}
        for field in FIELD_TO_GT:
            truth = gt[field].get(occid)
            if truth is None:
                continue
            row[f"acc_{field}"] = ratio(data.get(field, ""), truth)
            c = conf.get(field, {})
            ocr, llm = c.get("ocr"), c.get("llm")
            row[f"conf_ocr_{field}"] = ocr
            row[f"conf_llm_{field}"] = llm
            present = [v for v in (ocr, llm) if v is not None]
            row[f"conf_min_{field}"] = min(present) if present else None
        rows.append(row)
        print(f"  processed {fname}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} rows)\n")
    analyze(df)


def analyze(df: pd.DataFrame):
    for field in FIELD_TO_GT:
        acc_col = f"acc_{field}"
        if acc_col not in df.columns:
            continue
        print(f"=== {field} ===")
        baseline = df[acc_col].mean()
        for kind in ("ocr", "llm", "min"):
            conf_col = f"conf_{kind}_{field}"
            sub = df[[conf_col, acc_col]].dropna()
            if len(sub) < 3:
                print(f"  {kind:3}: too few rows ({len(sub)})")
                continue
            # Spearman = Pearson of ranks (avoids a scipy dependency)
            rho = sub[conf_col].rank().corr(sub[acc_col].rank())
            print(f"  {kind:3}: Spearman rho={rho:+.3f} (n={len(sub)})")
            for cut in (0.6, 0.7, 0.8, 0.9):
                kept = sub[sub[conf_col] >= cut]
                cov = len(kept) / len(sub) if len(sub) else 0
                acc = kept[acc_col].mean() if len(kept) else float("nan")
                print(f"        >= {cut:.1f}: mean acc={acc:.3f}  coverage={cov:.0%}")
        print(f"  baseline mean acc (no filter) = {baseline:.3f}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=["azure", "google"], default="azure")
    ap.add_argument("--image-dir", default=SAMPLE_DIR)
    ap.add_argument("--gt-dir", default=SAMPLE_DIR)
    ap.add_argument("--out", default="transcription/results/confidence_eval.csv")
    ap.add_argument("--analyze-csv", help="skip the pipeline; analyze an existing CSV")
    args = ap.parse_args()
    if args.analyze_csv:
        analyze(pd.read_csv(args.analyze_csv))
    else:
        run(args.pipeline, args.image_dir, args.gt_dir, args.out)
