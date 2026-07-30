"""Run Claude Sonnet 5 (image-native) extraction over a sample of the gbif-ne-500
set and cache each JSON output. Resumable: skips occids already cached.

    python nbs/sonnet5_run.py --n 100

Caches to transcription/results/sonnet5_cache/<occid>.json. Paid Anthropic calls.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, "transcription")
load_dotenv(Path("transcription/.env"))
import claude_sonnet as cs  # noqa: E402

GT = "transcription/data/gbif-ne-500"
OCR = "transcription/results/ocr_cache/azure"
OUT = "transcription/results/sonnet5_cache"


def usable_occids():
    tax = set()
    for line in open(os.path.join(GT, "taxons.txt"), encoding="utf-8", errors="ignore"):
        if ":" in line:
            tax.add(line.split(":", 1)[0].strip())
    out = []
    for f in sorted(os.listdir(GT)):
        if f.endswith(".jpeg") and f[:-5] in tax and os.path.exists(os.path.join(OCR, f[:-5] + ".json")):
            out.append(f[:-5])
    return out


def parse(raw):
    if not raw:
        return None
    s = raw.strip().removeprefix("<output_format>").removesuffix("</output_format>").strip()
    # strip a ```json fence if present
    if s.startswith("```"):
        s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(s)
    except Exception:
        return {"_parse_error": raw[:500]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    occids = usable_occids()[: args.n]
    todo = [o for o in occids if not os.path.exists(os.path.join(OUT, o + ".json"))]
    print(f"target {len(occids)}, already cached {len(occids) - len(todo)}, to run {len(todo)}", flush=True)
    t0 = time.time()
    for i, occid in enumerate(todo, 1):
        img = os.path.join(GT, occid + ".jpeg")
        try:
            raw = cs.run_claude_pipeline(img)
            data = parse(raw)
        except Exception as e:
            data = {"_error": str(e)[:300]}
        json.dump(data, open(os.path.join(OUT, occid + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
        el = time.time() - t0
        print(f"[{i}/{len(todo)}] {occid} {el/i:4.1f}s/img  {'ERR' if data and ('_error' in data or '_parse_error' in data) else 'ok'}", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
