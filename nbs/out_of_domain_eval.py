"""Score the pipelines on specimens unlike the calibration set.

All other figures come from New England, pre-1950 sheets. This runs the
globally-distributed set in transcription/data/raw-images and reports accuracy
and, per confidence band, claimed vs delivered.

Ground truth is transcription/data/gt-labels: scientificName and recordedBy only.

    python nbs/out_of_domain_eval.py

Free: reads cached outputs.
"""
import json
import os
import sys

sys.path.insert(0, "transcription")
sys.path.insert(0, os.path.dirname(__file__))
from confidence import build_confidence, GbifTaxonMatcher  # noqa: E402
from sonnet5_hallucination import correct, is_unknown  # noqa: E402

IMAGES = "transcription/data/raw-images"
AZURE_OCR = "transcription/results/ocr_cache/azure"
GOOGLE_OCR = "transcription/results/ocr_cache/google"
SONNET5 = "transcription/results/model_cache/claude-sonnet-5"
CHECKER = "transcription/results/model_cache/claude-sonnet-4-6"
SC = "transcription/results/self_consistency_cache/azure"
VIS = "transcription/results/vision_cache"
GT = "transcription/data/gt-labels"
GT_FILE = {"scientificName": "taxon_gt.txt", "recordedBy": "collector_gt.txt"}


def load_gt(fn):
    d = {}
    p = os.path.join(GT, fn)
    for line in open(p, encoding="utf-8", errors="replace"):
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d


def load(root, occid):
    p = os.path.join(root, f"{occid}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def band_report(name, pairs):
    if not pairs:
        print(f"  {name:16} no scored specimens")
        return
    acc = sum(o for _, o in pairs) / len(pairs)
    claimed = sum(s for s, _ in pairs) / len(pairs)
    print(f"  {name:16} n={len(pairs):<4} accuracy {acc:>5.0%}   "
          f"mean confidence {claimed:>5.2f}   gap {claimed - acc:>+5.2f}")
    for lo in (0.90, 0.80, 0.70):
        top = [(s, o) for s, o in pairs if s >= lo]
        if len(top) < 5:
            continue
        c = sum(s for s, _ in top) / len(top)
        a = sum(o for _, o in top) / len(top)
        print(f"      at >= {lo:.2f}: {len(top):>3} values, claims {c:.2f}, "
              f"delivers {a:.2f}   {'OVERSTATED by %.2f' % (c - a) if c - a > 0.05 else 'ok'}")


def azure_conf(occid, matcher):
    rec = load(AZURE_OCR, occid)
    if not rec:
        return None, None
    return rec["fields"], build_confidence(
        rec["fields"], rec["words"], sc_samples=load(SC, occid),
        vision_fields=load(VIS, occid), taxon_matcher=matcher,
        calibration="calibration")


def google_conf(occid, matcher):
    # Grounding + GBIF + vision, matching what google_vision.py ships.
    rec = load(GOOGLE_OCR, occid)
    if not rec:
        return None, None
    return rec["fields"], build_confidence(
        rec["fields"], rec["words"], vision_fields=load(VIS, occid),
        taxon_matcher=matcher, calibration="calibration_google")


def anthropic_conf(occid, matcher):
    import claude_sonnet as cs
    rec = load(SONNET5, occid)
    if not isinstance(rec, dict):
        return None, None
    fields = {f: rec.get(f, "") for f in
              ("scientificName", "location", "eventDate", "recordedBy", "barcode")}
    fields["scientificName"] = rec.get("verbatimScientificName") or fields["scientificName"]
    azure = load(AZURE_OCR, occid)
    return fields, cs.build_confidence(fields, load(CHECKER, occid),
                                       azure["fields"] if azure else None)


PIPELINES = {"azure": azure_conf, "google": google_conf, "anthropic": anthropic_conf}


def main():
    occids = sorted(os.path.splitext(f)[0] for f in os.listdir(IMAGES)
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))
    gts = {f: load_gt(v) for f, v in GT_FILE.items()}
    matcher = GbifTaxonMatcher()
    print(f"{len(occids)} out-of-domain specimens\n")

    for name, build in PIPELINES.items():
        pairs = {f: [] for f in GT_FILE}
        for o in occids:
            fields, conf = build(o, matcher)
            if not fields:
                continue
            for f in GT_FILE:
                c, pred, gt = conf.get(f), fields.get(f, ""), gts[f].get(o)
                if c is None or is_unknown(pred) or not gt:
                    continue
                shipped = pred
                if f == "scientificName":
                    fixed, _ = matcher.correct(pred)
                    shipped = fixed or pred
                ok = correct(f, shipped, gt)
                if ok is not None:
                    pairs[f].append((c, 1 if ok else 0))
        print(f"=== {name.upper()} (shipped confidence) ===")
        for f in GT_FILE:
            band_report(f, pairs[f])
        print()


if __name__ == "__main__":
    main()
