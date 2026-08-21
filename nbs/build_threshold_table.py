"""Emit coverage and precision at each review threshold, per pipeline and field.

Writes transcription/threshold_table.json:

    {pipeline: {field: {"ceiling": float, "calibrated": bool,
                        "levels": [{"cut", "coverage", "precision", "n"}, ...]}}}

  coverage  -- fraction of values that would skip review at this cut
  precision -- fraction of those that are actually correct
  ceiling   -- highest confidence the field can emit; a cut above it covers nothing

    python nbs/build_threshold_table.py

Free: reads cached outputs.
"""
import json
import os
import sys

sys.path.insert(0, "transcription")
sys.path.insert(0, os.path.dirname(__file__))
import confidence_compare as cc  # noqa: E402

CUTS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
OUT = "transcription/threshold_table.json"
MAPS = {"azure": "calibration.json",
        "google": "calibration_google.json",
        "anthropic": "calibration_anthropic.json"}
BUILDERS = {"azure": cc.azure_conf, "google": cc.google_full_conf,
            "anthropic": cc.anthropic_conf}


def main():
    occ = sorted(f[:-5] for f in os.listdir(cc.GOOGLE_OCR) if f.endswith(".json"))
    gts = {f: cc.load_gt(v) for f, v in cc.GT_FILE.items()}
    admin = json.load(open(os.path.join(cc.GT_DIR, "gbif_admin_location.json"),
                           encoding="utf-8"))
    matcher = cc.GbifTaxonMatcher()

    out = {}
    for name, build in BUILDERS.items():
        cal = json.load(open(os.path.join("transcription", MAPS[name]), encoding="utf-8"))
        pairs = {f: [] for f in cc.FIELDS}
        for o in occ:
            fields, conf = build(o, matcher)
            if not fields:
                continue
            for f in cc.FIELDS:
                c, pred = conf.get(f), fields.get(f, "")
                if c is None or cc.is_unknown(pred):
                    continue
                if f == "location":
                    acc = cc.admin_accuracy(pred, admin[o]) if o in admin else None
                    ok = None if acc is None else acc >= 0.5
                else:
                    gt = gts[f].get(o)
                    if not gt:
                        continue
                    shipped = pred
                    if f == "scientificName":
                        fixed, _ = matcher.correct(pred)
                        shipped = fixed or pred
                    ok = cc.correct(f, shipped, gt)
                if ok is not None:
                    pairs[f].append((c, 1 if ok else 0))

        out[name] = {}
        for f in cc.FIELDS:
            knots = cal.get(f)
            ceiling = max(y for _, y in knots) if knots else 1.0
            levels = []
            for cut in CUTS:
                cov, prec = cc.at_cutoff(pairs[f], cut)
                levels.append({"cut": cut, "coverage": round(cov, 4),
                               "precision": None if prec is None else round(prec, 4),
                               "n": len(pairs[f])})
            out[name][f] = {"ceiling": round(ceiling, 4),
                            "calibrated": bool(knots), "levels": levels}

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"wrote {OUT}\n")
    for name in out:
        print(f"  {name}")
        print(f"    {'field':16}{'ceiling':>9}" + "".join(f"{c:>13.2f}" for c in CUTS))
        for f in cc.FIELDS:
            d = out[name][f]
            row = f"    {f:16}{d['ceiling']:>9.2f}"
            for lv in d["levels"]:
                row += (f"{lv['coverage']:>7.0%}{lv['precision']:>6.0%}"
                        if lv["precision"] is not None else f"{'--':>13}")
            print(row)
        print()


if __name__ == "__main__":
    main()
