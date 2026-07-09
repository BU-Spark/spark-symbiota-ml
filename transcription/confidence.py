"""Shared confidence helpers for the OCR pipelines.

Each pipeline produces six JSON fields assembled by an LLM from OCR text. These
helpers score, per field, how much we should trust that value -- derived purely
from the OCR output, never from the LLM's own self-rating (which is subjective
and can be hallucinated).

A single best-match pass maps each field token back to the OCR words and yields
two signals:
  - coverage  -- the fraction of the field's content (by token length) that is
    actually grounded in the OCR words. This is the HEADLINE score: it measures
    whether the LLM's answer is supported by the scan or was invented/normalized,
    and it is the strongest empirical predictor of field correctness.
  - ocr_read  -- the length-weighted mean OCR-engine confidence of the matched
    words (the "OCR-stage" confidence). Computed every time but reported only in
    detail mode, because empirically it does not predict field correctness as
    well as coverage and should not drive the score.

Default output is a single float per field (coverage). Detail mode
(CONFIDENCE_DETAIL=1) yields {"coverage": ..., "ocr_read": ...} for analysis.
"""

import datetime
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional, Tuple

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
# Tokens shorter than this are dropped: short tokens (initials, "de", "var")
# produce jittery fuzzy ratios and match unrelated words. They carry little
# identifying content, so they count toward neither coverage nor ocr_read.
_MIN_TOKEN_LEN = 2


def _strip_accents(text: str) -> str:
    # Fold diacritics so "León" matches OCR "Leon" etc.
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def _tokenize(text: str) -> list:
    # Lowercased, accent-folded alphanumeric tokens; punctuation dropped.
    folded = _strip_accents(str(text).lower())
    return [t for t in re.split(r"\W+", folded) if len(t) >= _MIN_TOKEN_LEN]


def _best_match_conf(token: str, word_confs: list) -> Optional[float]:
    # Confidence of the OCR word best matching `token`, or None if none clears
    # the threshold. Pure-digit tokens (barcodes, date parts) require an *exact*
    # match -- a fuzzy match on digits would happily accept a wrong number.
    if token.isdigit():
        for content, conf in word_confs:
            if content == token:
                return conf
        return None

    best_conf = None
    best_ratio = _MATCH_THRESHOLD
    for content, conf in word_confs:
        ratio = SequenceMatcher(None, token, content).ratio()
        if ratio >= best_ratio:
            best_ratio = ratio
            best_conf = conf
    return best_conf


def match_field(field_value: str, ocr_words: list) -> Tuple[Optional[float], Optional[float]]:
    # One best-match pass. Returns (coverage, ocr_read).
    #   coverage: fraction of the field's tokens (by length) grounded in OCR --
    #             0.0 when the value has content but none of it is in the scan.
    #   ocr_read: length-weighted mean OCR confidence of the matched words, or
    #             None when nothing matched (so there is nothing to average).
    # Returns (None, None) when the value is empty/UNKNOWN, when there are no
    # usable OCR words, or when the value has no scorable tokens.
    if not field_value or str(field_value).strip().upper() == "UNKNOWN":
        return None, None

    word_confs = [
        (_strip_accents(str(w["content"]).lower()), w["confidence"])
        for w in ocr_words
        if w.get("content") and w.get("confidence") is not None
    ]
    if not word_confs:
        return None, None

    tokens = _tokenize(field_value)
    if not tokens:
        return None, None

    total_weight = 0.0     # summed length of all field tokens
    matched_weight = 0.0   # summed length of tokens grounded in OCR
    weighted_conf = 0.0    # sum(len * conf) over matched tokens
    for token in tokens:
        weight = len(token)
        total_weight += weight
        conf = _best_match_conf(token, word_confs)
        if conf is not None:
            matched_weight += weight
            weighted_conf += weight * conf

    coverage = round(matched_weight / total_weight, 4)
    ocr_read = round(weighted_conf / matched_weight, 4) if matched_weight else None
    return coverage, ocr_read


def map_field_confidence(field_value: str, ocr_words: list) -> Optional[float]:
    # Headline per-field score: coverage (grounding). See match_field.
    return match_field(field_value, ocr_words)[0]


# A catalog number: an optional short alpha prefix then a run of digits, e.g.
# "YU.037310", "CBS.008065", "061893".
_BARCODE_PATTERN = re.compile(r"^[A-Za-z]{0,5}[.\-_ ]?\d{3,}$")


def _digits(text: str) -> str:
    return re.sub(r"\D", "", str(text))


def barcode_confidence(field_value: str, ocr_words: list) -> Optional[float]:
    # Barcodes need their own scorer: plain coverage is ANTI-predictive for them
    # (labels carry many numbers, and copying any one verbatim looks "grounded"
    # but is often the wrong number). Instead we score two things that do predict
    # correctness: does the value look like a catalog number, and are its digits
    # actually present as an OCR token on the label.
    #   None -> empty/UNKNOWN         0.0 -> doesn't look like a barcode
    #   0.5  -> valid format, but its digits aren't found on the label (suspect)
    #   1.0  -> valid format and grounded on the label
    v = str(field_value).strip()
    if not v or v.upper() == "UNKNOWN":
        return None
    if not _BARCODE_PATTERN.match(v):
        return 0.0
    d = _digits(v)
    grounded = d and any(
        _digits(w["content"]) == d
        for w in ocr_words if w.get("content")
    )
    return 1.0 if grounded else 0.5


_YEAR_RE = re.compile(r"(1[6-9]\d\d|20[0-2]\d)")


def _event_year(text: str) -> Optional[int]:
    m = _YEAR_RE.search(str(text))
    return int(m.group(1)) if m else None


def _is_valid_date(text: str) -> bool:
    # Parses cleanly as a normalized (Darwin-Core-style) date.
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            datetime.datetime.strptime(str(text).strip(), fmt)
            return True
        except ValueError:
            pass
    return False


def eventdate_confidence(field_value: str, ocr_words: list) -> Optional[float]:
    # Dates are normalized by the LLM, so coverage of the normalized value is a
    # weak signal. What predicts date correctness is validity + plausibility
    # (the errors are malformed/implausible dates), boosted by whether the year
    # is actually printed on the label.
    #   None -> empty/UNKNOWN
    #   0.0  -> doesn't parse as a valid date, or implausible year
    #   0.6  -> valid plausible date, but its year isn't found on the label
    #   1.0  -> valid plausible date and its year is grounded on the label
    v = str(field_value).strip()
    if not v or v.upper() == "UNKNOWN":
        return None
    year = _event_year(v)
    if not _is_valid_date(v) or year is None or not (1600 <= year <= datetime.date.today().year):
        return 0.0
    ys = str(year)
    grounded = any(
        _digits(w["content"]) == ys for w in ocr_words if w.get("content")
    )
    return 1.0 if grounded else 0.6


def _field_confidence(field: str, value: str, ocr_words: list):
    # Dispatch to the per-field scoring strategy. Returns (primary, secondary):
    # for text fields primary=coverage, secondary=ocr_read; the structured
    # fields (barcode, eventDate) use their own validators and have no secondary.
    if field == "barcode":
        return barcode_confidence(value, ocr_words), None
    if field == "eventDate":
        return eventdate_confidence(value, ocr_words), None
    return match_field(value, ocr_words)


def build_confidence(fields: dict, ocr_words: list, detail: bool = False) -> dict:
    # Assemble the per-field confidence object.
    #
    # Default (detail=False): each field maps to a single float -- coverage --
    # or None when the value is empty/UNKNOWN or there are no OCR words:
    #     {"scientificName": 0.83, ...}
    # Detail (detail=True): each field maps to {"coverage": ..., "ocr_read": ...}
    # so the OCR-stage confidence is available for analysis without cluttering
    # the shipped single-number contract.
    confidence = {}
    for field in FIELDS:
        primary, secondary = _field_confidence(field, fields.get(field, ""), ocr_words)
        if detail:
            confidence[field] = {"coverage": primary, "ocr_read": secondary}
        else:
            confidence[field] = primary
    return confidence


# Flag (env-driven) for the richer {coverage, ocr_read} output shape. Off by
# default so the downstream JSON contract stays a single number per field.
def detail_enabled() -> bool:
    import os
    return os.environ.get("CONFIDENCE_DETAIL", "").lower() in ("1", "true", "yes")
