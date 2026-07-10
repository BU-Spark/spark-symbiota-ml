"""Tesseract OCR pipeline (baseline).

Originally a standalone script that dumped document-level text + average
confidence to a CSV. It now also exposes a `run_tesseract_pipeline(image_path)`
mirroring the Azure/Google pipelines, so Tesseract can feed the shared
per-field confidence path in confidence.py.

Two things distinguish Tesseract from the cloud OCR engines:
  - it emits per-*word* confidence on a 0-100 scale (with -1 for tokens it has
    no confidence for), so we rescale to 0-1 and drop the -1 sentinels; and
  - it has no built-in field extraction, so we reuse the Azure pipeline's LLM
    extraction (extract_info) to assemble the six fields from Tesseract's text.
"""

import json
import os
from io import StringIO

import pandas as pd
import pytesseract
from PIL import Image

# Works both as a script (cwd=transcription) and as a package import.
try:
    from transcription.confidence import build_confidence, detail_enabled
    from transcription.doc_intelligence import (
        extract_info, example_result, example_output,
    )
except ImportError:
    from confidence import build_confidence, detail_enabled
    from doc_intelligence import extract_info, example_result, example_output


def get_tesseract_words(image_path: str) -> list:
    # Per-word content + confidence, rescaled to 0-1. Tesseract reports -1 for
    # tokens with no confidence and blanks for layout gaps -- both dropped.
    data = pytesseract.image_to_data(
        Image.open(image_path),
        output_type=pytesseract.Output.DICT,
        timeout=30,
    )
    words = []
    for text, conf in zip(data["text"], data["conf"]):
        text = str(text).strip()
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            continue
        if text and conf >= 0:
            words.append({"content": text, "confidence": conf / 100.0})
    return words


def run_tesseract_pipeline(image_path: str):
    if not image_path.lower().endswith((".png", ".jpg", ".jpeg")):
        return None
    try:
        words = get_tesseract_words(image_path)
        document_text = " ".join(w["content"] for w in words)
        extracted_info = extract_info(document_text, example_result, example_output)

        data = json.loads(extracted_info)
        data.pop("confidence", None)
        data["image_path"] = image_path
        data["confidence"] = build_confidence(data, words, detail=detail_enabled())
        return json.dumps(data)
    except Exception as e:
        return f"An error occurred: {str(e)} on file {image_path}"


def batch_to_csv(img_dir: str = "./data/new-england-samples/output",
                 out_csv: str = "./results/tesseract_results.csv"):
    # Original baseline behavior: document-level text + mean confidence per image.
    results_df = pd.DataFrame()
    for image_path in os.listdir(img_dir):
        if not image_path.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        try:
            full_image_path = os.path.join(img_dir, image_path)
            ocr_output = pytesseract.image_to_data(Image.open(full_image_path), timeout=30)
            ocr_df = pd.read_csv(StringIO(ocr_output), sep="\t")

            ocr_df.dropna(subset=["text"], inplace=True)
            temp_df = ocr_df[ocr_df["text"] != " "]
            cleaned_blocks = temp_df.groupby("block_num").agg({"text": " ".join,
                                                               "conf": "mean"})
            combined_df = pd.DataFrame({
                "image_path": image_path,
                "combined_text": [" ".join(cleaned_blocks["text"].astype(str))],
                "avg_conf": [cleaned_blocks["conf"].mean()],
            })
            results_df = pd.concat((results_df, combined_df), ignore_index=True)
        except RuntimeError as timeout_error:
            print(f"Error: {timeout_error} on image located at: {image_path}")

    results_df.to_csv(out_csv)


if __name__ == "__main__":
    batch_to_csv()
