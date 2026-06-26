"""Shared confidence helpers for the OCR pipelines.

Each pipeline produces six JSON fields assembled by an LLM, but the OCR engine
reports confidence per *word/token*. These helpers bridge that gap: they map an
extracted field value back to the OCR words that produced it and aggregate their
confidence, so the output JSON can carry both an OCR-derived score and the LLM's
own self-rated score per field.
"""

import re
from difflib import SequenceMatcher
from typing import Optional

# The six extracted fields, in output order.
FIELDS = [
    "recordedBy",
    "location",
    "scientificName",
    "eventDate",
    "barcode",
    "institutionCode",
]

# A field token must match an OCR word at least this closely to count.
_MATCH_THRESHOLD = 0.8


def aggregate(confidences: list) -> Optional[float]:
    # A field is only as trustworthy as its weakest word, so use the min.
    # Returns None when there is nothing to aggregate.
    if not confidences:
        return None
    return min(confidences)


def _tokenize(text: str) -> list:
    # Split into lowercased alphanumeric tokens; punctuation is dropped.
    return [t for t in re.split(r"\W+", text.lower()) if t]


def map_field_confidence(field_value: str, ocr_words: list) -> Optional[float]:
    # Map an extracted field value back to its source OCR words and aggregate
    # their confidence. ocr_words is a list of {"content": str, "confidence": float}.
    # Returns None when no token matches any OCR word (e.g. the LLM inferred or
    # reformatted the value, such as a normalized date) -- itself a review signal.
    if not field_value or str(field_value).strip().upper() == "UNKNOWN":
        return None

    word_confs = [
        (str(w["content"]).lower(), w["confidence"])
        for w in ocr_words
        if w.get("content") and w.get("confidence") is not None
    ]
    if not word_confs:
        return None

    matched = []
    for token in _tokenize(str(field_value)):
        best_conf = None
        best_ratio = _MATCH_THRESHOLD
        for content, conf in word_confs:
            ratio = SequenceMatcher(None, token, content).ratio()
            if ratio >= best_ratio:
                best_ratio = ratio
                best_conf = conf
        if best_conf is not None:
            matched.append(best_conf)

    return aggregate(matched)


def build_confidence(fields: dict, ocr_words: list, llm_scores: dict = None,
                     include_llm: bool = False) -> dict:
    # Assemble the per-field confidence object.
    #
    # The scope is OCR confidence, so by default each field maps to a single
    # OCR-derived float (or None when no OCR word matched the value):
    #     {"scientificName": 0.40, ...}
    # The LLM self-rating is still computed (cheap) but kept behind include_llm,
    # which yields the richer {"ocr": ..., "llm": ...} shape for analysis.
    confidence = {}
    for field in FIELDS:
        ocr = map_field_confidence(fields.get(field, ""), ocr_words)
        if include_llm:
            confidence[field] = {
                "ocr": ocr,
                "llm": llm_scores.get(field) if llm_scores else None,
            }
        else:
            confidence[field] = ocr
    return confidence


# Flag (env-driven) for including the LLM self-rating in the output. Off by
# default so the downstream JSON contract stays OCR-only.
def include_llm_enabled() -> bool:
    import os
    return os.environ.get("CONFIDENCE_INCLUDE_LLM", "").lower() in ("1", "true", "yes")
