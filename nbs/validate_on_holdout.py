"""Score the shipped calibration maps on specimens they were never fitted on.

Every other calibration figure comes from repeated splits of the training set, and
min_count was chosen by looking at those splits, so they are optimistic. This scores
the maps as they ship against transcription/data/holdout-200, which is disjoint from
gbif-ne-500, hand-50, NE-50 and raw-images.

Run once per set. Refitting to improve a result here burns the holdout; fix against
the training splits and draw a fresh set with nbs/sample_from_export.py.

    python nbs/validate_on_holdout.py

Free: reads cached outputs.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "transcription")
sys.path.insert(0, os.path.dirname(__file__))
import confidence_compare as cc  # noqa: E402


def shipped(occid, matcher):
    """Each pipeline with the signals and map it actually ships."""
    return {
        "azure": cc.azure_conf(occid, matcher),
        "google": cc.google_vis_conf(occid, matcher),
        "anthropic": cc.anthropic_conf(occid, matcher),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="transcription/data/holdout-200")
    ap.add_argument("--cut", type=float, default=0.90)
    args = ap.parse_args()

    occids = sorted(os.path.splitext(f)[0] for f in os.listdir(args.holdout)
                    if f.endswith(".jpeg"))
    gts = {f: {} for f in cc.GT_FILE}
    for f, fn in cc.GT_FILE.items():
        p = os.path.join(args.holdout, fn)
        if os.path.exists(p):
            for line in open(p, encoding="utf-8", errors="replace"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    gts[f][k.strip()] = v.strip()
    admin = json.load(open(os.path.join(args.holdout, "gbif_admin_location.json"),
                           encoding="utf-8"))
    matcher = cc.GbifTaxonMatcher()

    pairs = {p: {f: [] for f in cc.FIELDS} for p in ("azure", "google", "anthropic")}
    for occid in occids:
        for name, (fields, conf) in shipped(occid, matcher).items():
            if not fields:
                continue
            for f in cc.FIELDS:
                c, pred = conf.get(f), fields.get(f, "")
                if c is None or cc.is_unknown(pred):
                    continue
                if f == "location":
                    acc = cc.admin_accuracy(pred, admin[occid]) if occid in admin else None
                    ok = None if acc is None else acc >= 0.5
                else:
                    gt = gts[f].get(occid)
                    if not gt:
                        continue
                    shipped_v = pred
                    if f == "scientificName":
                        fixed, _ = matcher.correct(pred)
                        shipped_v = fixed or pred
                    ok = cc.correct(f, shipped_v, gt)
                if ok is not None:
                    pairs[name][f].append((c, 1 if ok else 0))

    names = list(pairs)
    print(f"{len(occids)} specimens, never fitted on\n")

    print("===== calibration error on unseen data =====")
    print(f"  {'field':16}" + "".join(f"{n:>14}" for n in names))
    for f in cc.FIELDS:
        row = f"  {f:16}"
        for n in names:
            p = pairs[n][f]
            e = cc.ece(p) if len(p) >= 15 else None
            row += f"{e:>10.3f}{len(p):>4}" if e is not None else f"{'--':>14}"
        print(row)

    print(f"\n===== auto-accept at {args.cut:.2f}: claimed vs delivered =====")
    print(f"  {'field':16}" + "".join(f"{n:>22}" for n in names))
    print(f"  {'':16}" + "".join(f"{'n  claims  delivers':>22}" for _ in names))
    for f in cc.FIELDS:
        row = f"  {f:16}"
        for n in names:
            top = [(s, ok) for s, ok in pairs[n][f] if s >= args.cut]
            if len(top) < 5:
                row += f"{'--':>22}"
                continue
            claimed = sum(s for s, _ in top) / len(top)
            actual = sum(o for _, o in top) / len(top)
            flag = "!" if claimed - actual > 0.05 else " "
            row += f"{len(top):>8}{claimed:>8.2f}{actual:>5.2f}{flag}"
        print(row)
    print("\n  ! = claims more than it delivers by over 5 points")


if __name__ == "__main__":
    main()
