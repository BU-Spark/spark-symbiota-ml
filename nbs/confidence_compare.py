"""Confidence discrimination for all three pipelines on the same specimens.

Each pipeline gets the signals it actually ships: Azure adds self-consistency
and a vision read, Anthropic a checker model, Google neither. nbs/confidence_eval.py
scores grounding alone, which no pipeline ships.

Reports Spearman rho between the confidence score and whether the field was
right. Rank-based, so calibration does not affect it.

    python nbs/confidence_compare.py

Free: reads cached outputs only.
"""
import json
import os
import re
import sys
from difflib import SequenceMatcher

import pandas as pd

sys.path.insert(0, "transcription")
sys.path.insert(0, os.path.dirname(__file__))
import claude_sonnet as cs  # noqa: E402
from confidence import build_confidence, GbifTaxonMatcher, _tokenize  # noqa: E402
from sonnet5_hallucination import correct, get_field, is_unknown, load_gt  # noqa: E402

GT_DIR = os.environ.get("HERBARIA_GT_DIR", "transcription/data/gbif-ne-500")
AZURE_OCR = "transcription/results/ocr_cache/azure"
GOOGLE_OCR = "transcription/results/ocr_cache/google"
SC = "transcription/results/self_consistency_cache/azure"
VIS = "transcription/results/vision_cache"
SONNET5 = "transcription/results/model_cache/claude-sonnet-5"
CHECKER = "transcription/results/model_cache/claude-sonnet-4-6"

FIELDS = ["scientificName", "eventDate", "recordedBy", "barcode", "location"]
GT_FILE = {"scientificName": "taxons.txt", "eventDate": "dates.txt",
           "recordedBy": "collectors.txt", "barcode": "catalognumbers.txt",
           "location": "localities.txt"}


def load(root, occid):
    p = os.path.join(root, f"{occid}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def admin_accuracy(pred, a):
    # location correctness against GBIF's structured admin fields, matching
    # nbs/fit_calibration_anthropic.py. localities.txt is a free-text string and
    # scores a correct read wrong whenever the wording differs.
    parts = [a.get("stateProvince", ""),
             re.sub(r"\bcounty\b", "", a.get("county", ""), flags=re.I),
             a.get("locality", "")]
    g = _tokenize(" ".join(parts))
    if not g:
        return None
    p = _tokenize(pred)
    hits = sum(1 for t in g
               if any(SequenceMatcher(None, t.lower(), q.lower()).ratio() >= 0.85 for q in p))
    return hits / len(g)


def azure_conf(occid, matcher):
    rec = load(AZURE_OCR, occid)
    if not rec:
        return None, None
    conf = build_confidence(rec["fields"], rec["words"],
                            sc_samples=load(SC, occid),
                            vision_fields=load(VIS, occid),
                            taxon_matcher=matcher)
    return rec["fields"], conf


def google_conf(occid, matcher):
    # As google_vision.py ships it: OCR grounding + GBIF, no second read.
    rec = load(GOOGLE_OCR, occid)
    if not rec:
        return None, None
    return rec["fields"], build_confidence(rec["fields"], rec["words"], taxon_matcher=matcher)


def anthropic_conf(occid, matcher):
    rec = load(SONNET5, occid)
    if not isinstance(rec, dict):
        return None, None
    raw = {f: get_field(rec, f) for f in FIELDS}
    raw["scientificName"] = rec.get("verbatimScientificName") or raw["scientificName"]
    azure = load(AZURE_OCR, occid)
    return raw, cs.build_confidence(raw, load(CHECKER, occid),
                                    azure["fields"] if azure else None)


def google_vis_conf(occid, matcher):
    # Google plus the vision cross-read. The vision read is of the image, not of
    # any OCR text, so the existing cache applies to Google unchanged -- this is
    # what Google's confidence would look like with Azure's vision signal added.
    rec = load(GOOGLE_OCR, occid)
    if not rec:
        return None, None
    return rec["fields"], build_confidence(rec["fields"], rec["words"],
                                           vision_fields=load(VIS, occid),
                                           taxon_matcher=matcher)


PIPELINES = {"Azure": azure_conf, "Google": google_conf,
             "Google+vis": google_vis_conf, "Anthropic": anthropic_conf}


def main():
    occids = sorted(f[:-5] for f in os.listdir(GOOGLE_OCR) if f.endswith(".json"))
    gts = {f: load_gt(v) for f, v in GT_FILE.items()}
    admin = json.load(open(os.path.join(GT_DIR, "gbif_admin_location.json"), encoding="utf-8"))
    matcher = GbifTaxonMatcher()
    print(f"scored on {len(occids)} specimens\n")

    pairs = {p: {f: [] for f in FIELDS} for p in PIPELINES}
    for occid in occids:
        for pname, fn in PIPELINES.items():
            fields, conf = fn(occid, matcher)
            if not fields:
                continue
            for f in FIELDS:
                c, pred = conf.get(f), fields.get(f, "")
                if c is None or is_unknown(pred):
                    continue
                if f == "location":
                    acc = admin_accuracy(pred, admin[occid]) if occid in admin else None
                    ok = None if acc is None else acc >= 0.5
                else:
                    gt = gts[f].get(occid)
                    if not gt:
                        continue
                    shipped = pred
                    if f == "scientificName":
                        fixed, _ = matcher.correct(pred)
                        shipped = fixed or pred
                    ok = correct(f, shipped, gt)
                if ok is not None:
                    pairs[pname][f].append((c, 1 if ok else 0))

    names = list(PIPELINES)
    print(f"  {'field':16}" + "".join(f"{n:>22}" for n in names))
    print(f"  {'':16}" + "".join(f"{'rho      n':>22}" for _ in names))
    for f in FIELDS:
        row = f"  {f:16}"
        for n in names:
            p = pairs[n][f]
            if len(p) < 20 or len({c for c, _ in p}) < 2:
                row += f"{'--':>22}"
                continue
            rho = pd.Series([c for c, _ in p]).corr(
                pd.Series([o for _, o in p]), method="spearman")
            row += f"{rho:>+16.3f}{len(p):>6}"
        print(row)

    print("\n  rho is the score's ability to rank right above wrong.")
    print("  Below about +0.3 the score is too weak to route review on.")


if __name__ == "__main__":
    main()
