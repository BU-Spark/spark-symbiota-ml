"""Run an Anthropic model (image-native) over the gbif-ne-500 specimens and cache
each JSON output for scoring by nbs/model_compare.py.

Same prompt and occid selection as nbs/sonnet5_run.py; only the model changes. The
prompt in transcription/claude_sonnet.py must stay identical across models for the
cached runs to be comparable.

    python nbs/model_run.py --model claude-opus-5 --n 100
    python nbs/model_run.py --model claude-haiku-4-5 --n 100

Caches to transcription/results/model_cache/<model>/<occid>.json (override with
--out). Resumable: occids already cached are skipped.

--match-cache (default: the Sonnet 5 cache) restricts the run to occids that cache
already covers, so every model is scored on the same specimens.

PAID: each call bills the Anthropic API. Cost scales with --n; see docs/cost-projection.md.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, "transcription")
load_dotenv(Path("transcription/.env"))
import claude_sonnet as cs  # noqa: E402

GT = "transcription/data/gbif-ne-500"
OCR = "transcription/results/ocr_cache/azure"
OUT_ROOT = "transcription/results/model_cache"

# Vendor list price, $ per 1M tokens (input, output). Only used to convert measured
# token usage into $ per 1,000 specimens, which is how cost is reported. Sonnet 5 is
# at its introductory rate through 2026-08-31; the standard rate is 3.00/15.00.
PRICES = {
    "claude-opus-5":     (5.00, 25.00),
    "claude-opus-4-8":   (5.00, 25.00),
    "claude-opus-4-6":   (5.00, 25.00),
    "claude-sonnet-5":   (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5":  (1.00,  5.00),
}


def usable_occids():
    """Occids with an image, an Azure OCR cache entry, and taxon ground truth.

    Returned in filename order. Do not slice this directly for a sample -- GBIF
    assigns occurrence keys in batches per published dataset, so filename order
    clusters by institution and the first N come from one herbarium with one label
    format. Use sample_occids().
    """
    tax = set()
    for line in open(os.path.join(GT, "taxons.txt"), encoding="utf-8", errors="ignore"):
        if ":" in line:
            tax.add(line.split(":", 1)[0].strip())
    out = []
    for f in sorted(os.listdir(GT)):
        if f.endswith(".jpeg") and f[:-5] in tax and os.path.exists(os.path.join(OCR, f[:-5] + ".json")):
            out.append(f[:-5])
    return out


def sample_occids(occids, n, seed=0):
    """A deterministic random sample, so the set spans institutions and label
    formats rather than one published batch. Same seed gives the same specimens to
    every model, which is what makes the runs comparable."""
    if n >= len(occids):
        return sorted(occids)
    return sorted(random.Random(seed).sample(sorted(occids), n))


def parse(raw):
    # Uses the pipeline's own extractor so the eval and the pipeline agree on what
    # counts as a parseable response.
    if not raw:
        return None
    data = cs.extract_json(raw)
    if isinstance(data, dict):
        return data
    # Keep enough raw text to diagnose (and to re-parse offline after a fix).
    return {"_parse_error": str(raw)[:2000]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--match-cache", default="transcription/results/sonnet5_cache",
                    help="only run occids this cache already covers; 'none' to run "
                         "every usable occid")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed; the same seed selects the same specimens "
                         "for every model")
    args = ap.parse_args()

    out = args.out or os.path.join(OUT_ROOT, args.model)
    os.makedirs(out, exist_ok=True)

    occids = usable_occids()
    if args.match_cache.strip().lower() not in ("", "none") and os.path.isdir(args.match_cache):
        have = {f[:-5] for f in os.listdir(args.match_cache) if f.endswith(".json")}
        occids = [o for o in occids if o in have]
    occids = sample_occids(occids, args.n, seed=args.seed)

    todo = [o for o in occids if not os.path.exists(os.path.join(out, o + ".json"))]
    print(f"model {args.model}: target {len(occids)}, "
          f"already cached {len(occids) - len(todo)}, to run {len(todo)}", flush=True)
    if not todo:
        return

    in_tok = out_tok = 0
    t0 = time.time()
    for i, occid in enumerate(todo, 1):
        img = os.path.join(GT, occid + ".jpeg")
        try:
            raw, usage = cs.run_claude_pipeline(img, model=args.model, return_usage=True)
            data = parse(raw)
            if usage:
                in_tok += usage["input_tokens"]
                out_tok += usage["output_tokens"]
                if isinstance(data, dict):
                    data["_usage"] = usage
        except Exception as e:
            data = {"_error": str(e)[:300]}
        json.dump(data, open(os.path.join(out, occid + ".json"), "w", encoding="utf-8"),
                  ensure_ascii=False)
        el = time.time() - t0
        bad = data and ("_error" in data or "_parse_error" in data)
        print(f"[{i}/{len(todo)}] {occid} {el/i:4.1f}s/img  {'ERR' if bad else 'ok'}", flush=True)

    pin, pout = PRICES.get(args.model, (None, None))
    elapsed = time.time() - t0
    if pin:
        cost = in_tok / 1e6 * pin + out_tok / 1e6 * pout
        per_1k = cost / max(len(todo), 1) * 1000
        print(f"\ndone in {elapsed:.0f}s   ${cost:.2f} for {len(todo)} specimens"
              f"   =  ${per_1k:.2f} / 1,000 specimens")
    else:
        print(f"\ndone in {elapsed:.0f}s   no price listed for {args.model}; "
              f"add one to PRICES to report cost. tokens in={in_tok:,} out={out_tok:,}")


if __name__ == "__main__":
    main()
