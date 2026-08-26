"""Build the LLM self-consistency cache used to score per-field confidence.

Self-consistency = re-run the extraction K times at a higher temperature and
measure how much the K samples agree per field. Agreement is the best confidence
signal for the *interpretive* fields (eventDate, scientificName); verbatim fields
(recordedBy, barcode) saturate and gain nothing.

This reuses the OCR cache (nbs/ocr_cache.py) so it never re-calls Azure -- the OCR
words are reconstructed into the same document string doc_intelligence feeds the
LLM, then extract_info() is sampled K times at --temp. Cost is only the K
gpt-4o-mini calls per specimen.

    # build the OCR cache first (nbs/ocr_cache.py), then:
    python nbs/self_consistency_cache.py --pipeline azure --k 5 --temp 0.8

Writes one record per specimen:
    transcription/results/self_consistency_cache/<pipeline>/<occid>.json
containing a list of the K raw LLM outputs. Already-cached occids are skipped, so
a re-run only fills gaps.
"""

import argparse
import json
import os
import sys

OCR_CACHE_ROOT = "transcription/results/ocr_cache"
SC_CACHE_ROOT = "transcription/results/self_consistency_cache"


def _document_from_words(words: list) -> str:
    # Rebuild the exact string doc_intelligence.run_doc_intell_pipeline feeds the
    # LLM: OCR reading text, then a "Confidence Metrics" block of per-word scores.
    content = " ".join(w.get("content", "") for w in words)
    conf_lines = "".join(
        "'{}' confidence {}\n".format(w.get("content", ""), w.get("confidence"))
        for w in words
    )
    return content + "\n\nConfidence Metrics:\n" + conf_lines


def build(pipeline: str, k: int, temp: float):
    sys.path.insert(0, "transcription")
    from doc_intelligence import extract_info, example_result, example_output

    ocr_dir = os.path.join(OCR_CACHE_ROOT, pipeline)
    if not os.path.isdir(ocr_dir):
        raise SystemExit(f"no OCR cache at {ocr_dir} -- run nbs/ocr_cache.py first")
    out_dir = os.path.join(SC_CACHE_ROOT, pipeline)
    os.makedirs(out_dir, exist_ok=True)

    for fn in sorted(os.listdir(ocr_dir)):
        if not fn.endswith(".json"):
            continue
        occid = fn[:-5]
        out_path = os.path.join(out_dir, f"{occid}.json")
        if os.path.exists(out_path):
            print(f"  skip cached {occid}")
            continue
        with open(os.path.join(ocr_dir, fn), encoding="utf-8") as f:
            rec = json.load(f)
        document = _document_from_words(rec.get("words", []))

        samples, ok = [], True
        for i in range(k):
            raw = extract_info(document, example_result, example_output, temperature=temp)
            try:
                samples.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                print(f"  bad sample {occid}[{i}]: {str(raw)[:80]}")
                ok = False
                break
        if not ok or len(samples) < k:
            continue
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(samples, f)
        print(f"  cached {occid} (K={len(samples)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="azure")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.8)
    args = ap.parse_args()
    build(args.pipeline, args.k, args.temp)
