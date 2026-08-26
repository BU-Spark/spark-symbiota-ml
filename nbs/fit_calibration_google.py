"""Fit the confidence calibration map for the Google pipeline.

Isotonic (PAVA) with repeated holdout; a map ships only if it beats the raw score
on nearly every split. Writes transcription/calibration_google.json.

    python nbs/fit_calibration_google.py

Free: reads cached outputs.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, "transcription")
sys.path.insert(0, os.path.dirname(__file__))
from confidence import build_confidence, GbifTaxonMatcher  # noqa: E402
from fit_calibration_anthropic import admin_accuracy, ece, isotonic, over_ece  # noqa: E402
from sonnet5_hallucination import correct, is_unknown, load_gt  # noqa: E402

GT_DIR = os.environ.get("HERBARIA_GT_DIR", "transcription/data/gbif-ne-500")
OCR = "transcription/results/ocr_cache/google"
SC = "transcription/results/self_consistency_cache/google"
VIS = "transcription/results/vision_cache"
OUT = "transcription/calibration_google.json"

FIELDS = ["scientificName", "location", "eventDate", "recordedBy", "barcode"]
GT_FILE = {"scientificName": "taxons.txt", "eventDate": "dates.txt",
           "recordedBy": "collectors.txt", "barcode": "catalognumbers.txt"}


def load(root, occid):
    p = os.path.join(root, f"{occid}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=15)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--point-estimate", action="store_true",
                    help="ship observed accuracy instead of its lower bound")
    # Defaults to the signal set google_vision.py calls. Fitting a signal the
    # pipeline does not use produces a map for a configuration that never runs.
    ap.add_argument("--with-sc", action="store_true",
                    help="also fit on self-consistency; only correct if the pipeline "
                         "is changed to call it")
    args = ap.parse_args()

    admin = json.load(open(os.path.join(GT_DIR, "gbif_admin_location.json"), encoding="utf-8"))
    gts = {f: load_gt(v) for f, v in GT_FILE.items()}
    matcher = GbifTaxonMatcher()
    occ = sorted(f[:-5] for f in os.listdir(OCR) if f.endswith(".json"))
    print(f"google cache: {len(occ)} specimens, "
          f"signals: grounding + GBIF + vision{' + self-consistency' if args.with_sc else ''}\n")

    pairs = {f: [] for f in FIELDS}
    for o in occ:
        rec = load(OCR, o)
        if not rec:
            continue
        conf = build_confidence(rec["fields"], rec["words"],
                                sc_samples=load(SC, o) if args.with_sc else None,
                                vision_fields=load(VIS, o),
                                taxon_matcher=matcher,
                                calibration=None)      # raw signal, not remapped
        for f in FIELDS:
            c, pred = conf.get(f), rec["fields"].get(f, "")
            if c is None or is_unknown(pred):
                continue
            if f == "location":
                acc = admin_accuracy(pred, admin[o]) if o in admin else None
                ok = None if acc is None else acc >= 0.5
            else:
                gt = gts[f].get(o)
                if not gt:
                    continue
                shipped = pred
                if f == "scientificName":
                    # Fit the name the pipeline ships, not the raw read -- otherwise
                    # the map learns a target the runtime never produces.
                    fixed, _ = matcher.correct(pred)
                    shipped = fixed or pred
                ok = correct(f, shipped, gt)
            if ok is not None:
                pairs[f].append((c, 1 if ok else 0))

    out = {}
    print(f"  {'field':16} {'n':>5} {'holdout raw':>12} {'holdout cal':>12} "
          f"{'splits won':>11}  ship?")
    for f in FIELDS:
        p = pairs[f]
        if len(p) < 30:
            print(f"  {f:16} {len(p):>5}   too few rows to fit")
            continue
        wins, raws, cals, bad = 0, [], [], None
        for s in range(args.splits):
            rng = random.Random(args.seed + s)
            idx = list(range(len(p)))
            rng.shuffle(idx)
            cut = int(len(p) * (1 - args.holdout))
            train = [p[i] for i in idx[:cut]]
            test = [p[i] for i in idx[cut:]]
            knots = isotonic(train, args.min_count, not args.point_estimate)
            if len({y for _, y in knots}) < 2 or len(test) < 20:
                bad = "fit collapsed" if len({y for _, y in knots}) < 2 else "holdout too small"
                break
            score = ece if args.point_estimate else over_ece
            h0, h1 = score(test), score(test, knots)
            raws.append(h0)
            cals.append(h1)
            # Conservative mode also wants the ceiling the map imposes, so ship
            # unless the map makes over-statement worse.
            wins += (h1 <= h0 + 0.005) if not args.point_estimate else (h1 < h0)
        if bad:
            print(f"  {f:16} {len(p):>5} {'--':>12} {'--':>12} {'--':>11}  no ({bad})")
            continue
        keep = wins >= args.splits - 1
        print(f"  {f:16} {len(p):>5} {sum(raws)/len(raws):>12.3f} "
              f"{sum(cals)/len(cals):>12.3f} {wins:>6}/{args.splits}"
              f"    {'YES' if keep else 'no (keep raw)'}")
        if keep:
            out[f] = isotonic(p, args.min_count, not args.point_estimate)

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {OUT}  (calibrated: {sorted(out) or 'none'})")
    for f, k in out.items():
        print(f"  {f}: " + "  ".join(f"{x:g}->{y:.2f}" for x, y in k))


if __name__ == "__main__":
    main()
