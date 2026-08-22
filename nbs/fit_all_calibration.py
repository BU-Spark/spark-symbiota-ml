"""Rebuild every shipped calibration map, in the order they must be run.

The per-pipeline fitters replace their whole map file, so the recordedBy fitter has
to run after them or its field is dropped. This runs them in order and reports what
each produced.

    python nbs/fit_all_calibration.py
    python nbs/fit_all_calibration.py --dry-run

Free: reads cached outputs.
"""
import argparse
import json
import os
import subprocess
import sys

STEPS = [
    ("Azure, all fields", ["nbs/fit_calibration.py"]),
    ("Anthropic, all fields", ["nbs/fit_calibration_anthropic.py"]),
    ("Anthropic recordedBy",
     ["nbs/fit_recordedby_calibration.py", "--anthropic",
      "--ocr", "transcription/results/model_cache/claude-sonnet-5",
      "--map", "calibration_anthropic.json"]),
    ("Google, all fields", ["nbs/fit_calibration_google.py", "--min-count", "12"]),
    ("Google recordedBy",
     ["nbs/fit_recordedby_calibration.py", "--map", "calibration_google.json",
      "--ocr", "transcription/results/ocr_cache/google", "--sc", ""]),
    ("Threshold table", ["nbs/build_threshold_table.py"]),
]

MAPS = ["calibration.json", "calibration_google.json", "calibration_anthropic.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the order, run nothing")
    args = ap.parse_args()

    for i, (label, cmd) in enumerate(STEPS, 1):
        print(f"[{i}/{len(STEPS)}] {label}")
        if args.dry_run:
            print(f"        {' '.join(cmd)}")
            continue
        r = subprocess.run([sys.executable, "-W", "ignore"] + cmd,
                           capture_output=True, text=True)
        if r.returncode != 0:
            # A fitter refusing to write is a valid outcome; a crash is not.
            tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or ["failed"]
            print(f"        {tail[0]}")
        else:
            wrote = [l for l in r.stdout.splitlines() if "wrote" in l]
            print(f"        {wrote[-1] if wrote else 'ok'}")

    if args.dry_run:
        return
    print("\nfields covered per map:")
    for m in MAPS:
        p = os.path.join("transcription", m)
        cal = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        missing = {"scientificName", "eventDate", "recordedBy", "barcode", "location"} - set(cal)
        print(f"  {m:<30}{len(cal)}/5" + (f"  missing: {sorted(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
