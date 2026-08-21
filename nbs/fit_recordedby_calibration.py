"""Fit a `recordedBy` calibration map, validated on the top of the range.

recordedBy's raw score is well calibrated on average, so an overall-ECE test
never ships a map -- but a raw 1.00 still reads as certainty and is worth ~91%.
This scores only the band at or above --cutoff, which is what a reviewer acts on.

    python nbs/fit_recordedby_calibration.py --dry-run
    python nbs/fit_recordedby_calibration.py --map calibration_google.json
        --ocr transcription/results/ocr_cache/google
    python nbs/fit_recordedby_calibration.py --anthropic
        --ocr transcription/results/model_cache/claude-sonnet-5
        --map calibration_anthropic.json

Free: reads cached outputs.
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, "transcription")
sys.path.insert(0, os.path.dirname(__file__))
from confidence import build_confidence, GbifTaxonMatcher  # noqa: E402
from fit_calibration_anthropic import isotonic  # noqa: E402
from sonnet5_hallucination import correct, is_unknown, load_gt  # noqa: E402

VIS = "transcription/results/vision_cache"
FIELD = "recordedBy"


def load(root, occid):
    if not root:
        return None
    p = os.path.join(root, f"{occid}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def apply_knots(knots, s):
    if not knots:
        return s
    if s <= knots[0][0]:
        return knots[0][1]
    if s >= knots[-1][0]:
        return knots[-1][1]
    for i in range(1, len(knots)):
        x1, y1 = knots[i]
        if s <= x1:
            x0, y0 = knots[i - 1]
            return y0 if x1 == x0 else y0 + (y1 - y0) * (s - x0) / (x1 - x0)
    return knots[-1][1]


def top_band_error(pairs, knots, cutoff):
    """|claimed - actual| among items the map places at or above `cutoff`."""
    kept = [(apply_knots(knots, s) if knots else s, ok) for s, ok in pairs]
    kept = [(v, ok) for v, ok in kept if v >= cutoff]
    if len(kept) < 10:
        return None
    claimed = sum(v for v, _ in kept) / len(kept)
    actual = sum(ok for _, ok in kept) / len(kept)
    return max(0.0, claimed - actual)   # only over-statement can mislead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", default="transcription/results/ocr_cache/azure")
    ap.add_argument("--sc", default="transcription/results/self_consistency_cache/azure")
    ap.add_argument("--map", default="calibration.json")
    ap.add_argument("--cutoff", type=float, default=0.90)
    ap.add_argument("--min-count", type=int, default=25)
    ap.add_argument("--splits", type=int, default=25)
    ap.add_argument("--win-rate", type=float, default=0.70,
                    help="fraction of splits the map must improve")
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--point-estimate", action="store_true",
                    help="ship observed accuracy instead of its lower bound")
    ap.add_argument("--anthropic", action="store_true",
                    help="score via claude_sonnet.build_confidence, not the OCR-word path")
    ap.add_argument("--checker", default="transcription/results/model_cache/claude-sonnet-4-6")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gts = load_gt("collectors.txt")
    matcher = GbifTaxonMatcher()
    pairs = []
    for fn in sorted(os.listdir(args.ocr)):
        if not fn.endswith(".json"):
            continue
        occid = fn[:-5]
        rec = load(args.ocr, occid)
        if not rec:
            continue
        if args.anthropic:
            import claude_sonnet as cs
            if not isinstance(rec, dict):
                continue
            fields = {f: rec.get(f, "") for f in ("scientificName", "location",
                                                  "eventDate", "recordedBy", "barcode")}
            azure = load("transcription/results/ocr_cache/azure", occid)
            conf = cs.build_confidence(fields, load(args.checker, occid),
                                       azure["fields"] if azure else None,
                                       calibrate=False)
        else:
            fields = rec["fields"]
            conf = build_confidence(fields, rec["words"],
                                    sc_samples=load(args.sc, occid),
                                    vision_fields=load(VIS, occid),
                                    taxon_matcher=matcher, calibration=None)
        c, pred, gt = conf.get(FIELD), fields.get(FIELD, ""), gts.get(occid)
        if c is None or is_unknown(pred) or not gt:
            continue
        if re.search(r"no data|not available", str(gt), re.I):   # placeholder truth
            continue
        ok = correct(FIELD, pred, gt)
        if ok is not None:
            pairs.append((c, 1 if ok else 0))

    print(f"{len(pairs)} specimens; overall accuracy {sum(o for _, o in pairs)/len(pairs):.1%}")
    top = [ok for s, ok in pairs if s >= args.cutoff]
    print(f"at raw >= {args.cutoff}: n={len(top)}, actually correct {sum(top)/len(top):.1%}\n")

    wins, e_raw, e_cal = 0, [], []
    for s in range(args.splits):
        rng = random.Random(args.seed + s)
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        cut = int(len(pairs) * (1 - args.holdout))
        train = [pairs[i] for i in idx[:cut]]
        test = [pairs[i] for i in idx[cut:]]
        knots = isotonic(train, args.min_count, not args.point_estimate)
        a = top_band_error(test, None, args.cutoff)
        b = top_band_error(test, knots, args.cutoff)
        if a is None or b is None:
            continue
        e_raw.append(a)
        e_cal.append(b)
        wins += b < a
    n = len(e_raw)
    if n == 0:
        raise SystemExit("no split had enough values above the cutoff to judge")
    mean_raw, mean_cal = sum(e_raw) / n, sum(e_cal) / n
    print(f"top-band error   raw {mean_raw:.3f} -> calibrated {mean_cal:.3f}"
          f"   ({wins}/{n} splits improved, {wins/n:.0%})")
    if wins / n < args.win_rate or mean_cal >= mean_raw:
        raise SystemExit("does not generalise -- not written")

    knots = isotonic(pairs, args.min_count, not args.point_estimate)
    print("\nnew map: " + "  ".join(f"{x:g}->{y:.2f}" for x, y in knots))
    print(f"ceiling: {max(y for _, y in knots):.2f}  (was 1.00, uncalibrated)")

    path = os.path.join("transcription", args.map)
    cal = json.load(open(path, encoding="utf-8"))
    if args.dry_run:
        print("\n--dry-run: not written")
        return
    cal[FIELD] = knots
    json.dump(cal, open(path, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {path} ({FIELD} only)")


if __name__ == "__main__":
    main()
