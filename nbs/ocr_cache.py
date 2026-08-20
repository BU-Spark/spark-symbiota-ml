"""Build an offline OCR + LLM cache so the confidence scoring in
transcription/confidence.py can be iterated on without re-calling the
Azure / Google / OpenAI APIs on every change.

For each sample image it runs the pipeline's OCR + LLM extraction *once* and
writes one JSON record to
    transcription/results/ocr_cache/<pipeline>/<occid>.json
containing the extracted field values, the raw OCR words (content + confidence,
plus polygon when available), and the LLM's per-field self-rating.

Run this once (incurs API cost on the team's keys):
    python nbs/ocr_cache.py --pipeline azure
    python nbs/ocr_cache.py --pipeline google

Afterwards, tune confidence.py for free with:
    python nbs/confidence_eval.py --cache-dir transcription/results/ocr_cache/azure

Records already cached are skipped, so a re-run only fills in gaps (e.g. images
that failed the first time, or after you add credentials for another pipeline).
"""

import argparse
import json
import os
import sys

SAMPLE_DIR = "transcription/data/new-england-samples/output"
CACHE_ROOT = "transcription/results/ocr_cache"

# The six extracted fields, in output order (mirrors confidence.FIELDS).
FIELDS = [
    "recordedBy",
    "location",
    "scientificName",
    "eventDate",
    "barcode",
    "institutionCode",
]


def _clean_words(words: list) -> list:
    # Normalize OCR words to a JSON-serializable {content, confidence, polygon}.
    # Azure polygons are Point objects (with .x/.y); Google has no polygon.
    clean = []
    for w in words:
        poly = w.get("polygon")
        if poly:
            try:
                poly = [[getattr(p, "x", p[0]), getattr(p, "y", p[1])] for p in poly]
            except Exception:
                poly = None
        clean.append({
            "content": w.get("content"),
            "confidence": w.get("confidence"),
            "polygon": poly,
        })
    return clean


def _split_fields(llm_json: str):
    # Parse the LLM's raw JSON into (fields, llm_scores).
    data = json.loads(llm_json)
    llm_scores = data.pop("confidence", {}) or {}
    fields = {k: data.get(k, "") for k in FIELDS}
    return fields, llm_scores


def _azure(image_path: str):
    from doc_intelligence import (
        process_image, get_image_words, extract_info,
        example_result, example_output,
    )
    result = process_image(image_path)
    words, confidence_text = get_image_words(result)
    document_content = result.content + "\n\nConfidence Metrics:\n" + confidence_text
    fields, llm_scores = _split_fields(
        extract_info(document_content, example_result, example_output))
    return fields, _clean_words(words), llm_scores


def _google(image_path: str):
    from google_vision import batch_process_documents, generate_metadata, shots
    extracted_text, words = batch_process_documents(image_path, "image/jpeg")
    fields, llm_scores = _split_fields(generate_metadata(extracted_text, shots))
    return fields, _clean_words(words), llm_scores


def _tesseract(image_path: str):
    from tesseract_pipeline import get_tesseract_words
    from doc_intelligence import extract_info, example_result, example_output
    words = get_tesseract_words(image_path)
    document_text = " ".join(w["content"] for w in words)
    fields, llm_scores = _split_fields(
        extract_info(document_text, example_result, example_output))
    return fields, _clean_words(words), llm_scores


BUILDERS = {"azure": _azure, "google": _google, "tesseract": _tesseract}


def build(pipeline: str, image_dir: str, only: set = None):
    sys.path.insert(0, "transcription")
    builder = BUILDERS[pipeline]
    out_dir = os.path.join(CACHE_ROOT, pipeline)
    os.makedirs(out_dir, exist_ok=True)

    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        occid = os.path.splitext(fname)[0]
        if only is not None and occid not in only:
            continue
        out_path = os.path.join(out_dir, f"{occid}.json")
        if os.path.exists(out_path):
            print(f"  skip cached {occid}")
            continue
        try:
            fields, words, llm_scores = builder(os.path.join(image_dir, fname))
        except Exception as e:
            print(f"  FAIL {occid}: {str(e)[:120]}")
            continue
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "occid": occid,
                "pipeline": pipeline,
                "fields": fields,
                "words": words,
                "llm_scores": llm_scores,
            }, f)
        print(f"  cached {occid} ({len(words)} words)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=list(BUILDERS), default="azure")
    ap.add_argument("--image-dir", default=SAMPLE_DIR)
    ap.add_argument("--occids-from", default=None,
                    help="cache dir whose occids to mirror, so a new pipeline is "
                         "built on the same specimens an existing run covers")
    args = ap.parse_args()
    only = None
    if args.occids_from:
        only = {f[:-5] for f in os.listdir(args.occids_from) if f.endswith(".json")}
        print(f"restricted to {len(only)} occids from {args.occids_from}")
    build(args.pipeline, args.image_dir, only)
