import anthropic
import base64
import os 
import json
from pathlib import Path

from dotenv import load_dotenv
from utils import image_utils

# Load from this file's directory, not the caller's cwd, so the pipeline works
# whether it is run from transcription/ or imported from the repository root.
load_dotenv(Path(__file__).parent / ".env")

OUTPUT_FORMAT = '{"recordedBy": "", "location": "", "scientificName": "", "eventDate": "", "barcode": "", "institutionCode": ""}'

def encode_image_to_base64(image_path):
    # Encode an image located at `image_path` into a Base64 string.
    try:
        # Open and read the image file in binary mode
        with open(image_path, "rb") as image_file:
            # Read the contents of the image file
            image_data = image_file.read()
        
        # Encode the image data into Base64
        base64_encoded_image = base64.standard_b64encode(image_data).decode('utf-8')
        
        return base64_encoded_image
    
    except FileNotFoundError:
        print(f"Error: The file at {image_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

DEFAULT_MODEL = "claude-sonnet-5"

_TAXON_MATCHER = None


def _taxon_matcher():
    global _TAXON_MATCHER
    if _TAXON_MATCHER is None:
        try:
            from transcription.confidence import GbifTaxonMatcher
        except ImportError:
            from confidence import GbifTaxonMatcher
        _TAXON_MATCHER = GbifTaxonMatcher()
    return _TAXON_MATCHER


def apply_taxon_correction(data: dict) -> dict:
    """Replace scientificName with GBIF's accepted species, in place.

    Same step doc_intelligence.py applies to the Azure output. A label carrying a
    synonym (Cypripedium pubescens) or a light misread is snapped onto the accepted
    name (C. parviflorum); a name GBIF cannot match is left alone. Lossless -- the
    original read is kept in verbatimScientificName. Idempotent.
    """
    raw = data.get("scientificName", "")
    if not raw or str(raw).strip().upper() == "UNKNOWN":
        return data
    try:
        corrected, match_type = _taxon_matcher().correct(raw)
    except Exception:
        return data  # lookup unavailable -> keep the model's read
    data["_taxonMatchType"] = match_type
    if corrected and corrected != raw:
        data["verbatimScientificName"] = raw
        data["scientificName"] = corrected
    return data


# Scored from the GBIF backbone and the offline gazetteer; no extra API call.
FREE_FIELDS = ("scientificName", "location")

# No external authority to check against, so these need a second model's read.
CHECKED_FIELDS = ("eventDate", "recordedBy")

# barcode is scored only against the Azure pipeline's read. A second Claude model
# is a weak checker here (rho +0.16); the Azure OCR words alone are useless (+0.00).
AZURE_FIELDS = ("barcode",)


_CALIBRATION = None
_CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), "calibration_anthropic.json")


def _calibration():
    # Separate from the Azure pipeline's calibration.json -- the same raw score
    # means a different probability for a different pipeline. Fit by
    # nbs/fit_calibration_anthropic.py; missing file means uncalibrated scores.
    global _CALIBRATION
    if _CALIBRATION is None:
        try:
            _CALIBRATION = json.load(open(_CALIBRATION_PATH, encoding="utf-8"))
        except Exception:
            _CALIBRATION = {}
    return _CALIBRATION


def _apply_calibration(field, score):
    # Piecewise-linear between the fitted knots, clamped outside them. Monotonic,
    # so ranking is unchanged -- only the numbers become probabilities.
    knots = _calibration().get(field)
    if score is None or not knots:
        return score
    if score <= knots[0][0]:
        return knots[0][1]
    if score >= knots[-1][0]:
        return knots[-1][1]
    for i in range(1, len(knots)):
        x1, y1 = knots[i]
        if score <= x1:
            x0, y0 = knots[i - 1]
            return round(y0 if x1 == x0 else y0 + (y1 - y0) * (score - x0) / (x1 - x0), 4)
    return knots[-1][1]


def build_confidence(data: dict, checker: dict = None, azure_read: dict = None,
                     calibrate: bool = True) -> dict:
    """Per-field confidence for the Anthropic pipeline.

    Two optional independent reads of the same specimen, each unlocking the fields
    it is measurably good at:
      checker    -- a second Claude model. Adds eventDate and recordedBy.
      azure_read -- the Azure pipeline's field dict. Adds barcode.
    Each costs one extra call per specimen.

    Fields with no usable signal are omitted; envelope.py renders a missing field
    as "confidence unavailable" rather than as a low score.

    calibration_anthropic.json maps the raw signal onto observed accuracy, so a
    shipped 0.91 means roughly 91% likely correct. Pass calibrate=False for the raw
    signal; nbs/fit_calibration_anthropic.py needs that to refit without compounding
    the previous fit. Only scientificName has a fitted map so far.
    """
    try:
        from transcription.confidence import (barcode_confidence,
                                              eventdate_confidence,
                                              location_confidence,
                                              recordedby_confidence,
                                              scientificname_confidence)
    except ImportError:
        from confidence import (barcode_confidence, eventdate_confidence,
                                location_confidence, recordedby_confidence,
                                scientificname_confidence)

    fields = list(FREE_FIELDS)
    if checker:
        fields += list(CHECKED_FIELDS)
    if azure_read:
        fields += list(AZURE_FIELDS)
    out = {}
    for field in fields:
        value = data.get(field, "")
        other = (checker or {}).get(field)
        try:
            if field == "scientificName":
                score = scientificname_confidence(value, None, _taxon_matcher(), None)
            elif field == "location":
                score = location_confidence(value, None, None)
            elif field == "eventDate":
                score = eventdate_confidence(value, [], None, other)
            elif field == "barcode":
                score = barcode_confidence(value, [], [azure_read.get("barcode", "")])
            else:
                score = recordedby_confidence(value, None, other)
        except Exception:
            score = None
        if calibrate:
            score = _apply_calibration(field, score)
        if score is not None:
            out[field] = score
    return out


def build_free_confidence(data: dict, calibrate: bool = True) -> dict:
    """build_confidence with no extra reads: the two fields that cost nothing."""
    return build_confidence(data, calibrate=calibrate)


def _balanced_object(s):
    """First balanced {...} in s, ignoring braces inside strings. None if absent."""
    start = depth = None
    in_str = esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if start is None:
                start, depth = i, 0
            depth += 1
        elif ch == "}" and start is not None:
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def extract_json(text):
    """Parse the JSON object out of a model response, or None.

    Models wrap the answer differently under the same prompt -- bare, inside a
    ```json fence, or after a prose preamble -- so pull the object out wherever it
    sits rather than assuming the response is bare JSON.
    """
    if not text:
        return None
    s = str(text).strip().removeprefix("<output_format>").removesuffix("</output_format>").strip()
    for candidate in (s, _balanced_object(s)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None


def _correct_json(text: str, checker: dict = None) -> str:
    # Unparseable input is returned untouched for the caller to handle.
    data = extract_json(text)
    if not isinstance(data, dict):
        return text
    # Score before correcting: correcting first makes every name match GBIF and
    # saturates the signal.
    data["confidence"] = build_confidence(data, checker)
    data = apply_taxon_correction(data)
    return json.dumps(data, ensure_ascii=False)

CHECKER_MODEL = "claude-sonnet-4-6"


def run_claude_pipeline(image_path, model=DEFAULT_MODEL, return_usage=False,
                        checker_model=None):
    # `model` lets the same prompt run across models; keep the prompt identical or
    # cached runs stop being comparable.
    # `return_usage` returns (text, usage) instead of text; usage is None on error.
    # `checker_model` re-reads the image to score eventDate and recordedBy, at
    # double the cost.
    checker = None
    if checker_model:
        try:
            checker = extract_json(run_claude_pipeline(image_path, checker_model))
        except Exception:
            checker = None
    if image_path.lower().endswith((".png", ".jpg", ".jpeg")):
        image_utils.resize_image(image_path)
        encoded_image = encode_image_to_base64(image_path)

        # set up API calls to Claude
        anthropic_key = os.environ["ANTHROPIC_API_KEY"]
        client = anthropic.Anthropic(api_key=anthropic_key)
        
        # api call to claude 
        # maximum 5MB image via API 
        try:
            message = client.messages.create(
                model=model,
                # Caps thinking and response text together; 1024 truncated some
                # replies mid-JSON. Only generated tokens are billed.
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": encoded_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": "You are an expert in herbarium specimens and cursive handwriting. Perform OCR on this image and transcribe six items from the scanned herbarium specimen: the name of the specimen collector (recordedBy), the location the specimen was collected, the scientific name (genus and species, minimally) and/or any identifying information about the specimen, the event date the specimen was collected, the barcode associated with the specimen, and the institution code. Your response should contain only the output as a JSON object in plaintext. For the taxon name, only output recognized species within the identified genus. Only use the information available in the image or insert 'UNKNOWN' if there is none or if you are unsure. Use exactly these JSON keys. Your response should be formatted as such: "
                                f"<output_format>{OUTPUT_FORMAT}</output_format>"
                            }
                        ],
                    }
                ],
            )
            # Newer models can return a thinking block before the text block, so
            # pick the first text block rather than content[0] blindly.
            result = next((b.text for b in message.content if b.type == "text"), "")
            result = _correct_json(result, checker)
            if return_usage:
                u = message.usage
                return result, {"input_tokens": u.input_tokens,
                                "output_tokens": u.output_tokens,
                                "model": message.model}
            return result

        except Exception as e:
            # JSON, not a bare message, so a caller that json.loads the result
            # handles the failure instead of raising. Detect it via "_error".
            err = json.dumps({"_error": str(e)})
            return (err, None) if return_usage else err

    # Not an image we can read; same shape as the error path.
    err = json.dumps({"_error": f"unsupported file type: {image_path}"})
    return (err, None) if return_usage else err
    
if __name__ == "__main__":
    #image_path =  "/Users/mvoong/Desktop/spark-symbiota-ml/transcription/data/new-england-samples/output/1262197442.jpeg"
    #result = run_claude_pipeline(image_path)
    
    import pandas as pd 
    results_df = pd.DataFrame()

    image_folder = "/Users/mvoong/Desktop/spark-symbiota-ml/transcription/data/new-england-samples/output"
    image_files = os.listdir(image_folder)

    for img in image_files:
        image_path = os.path.join(image_folder, img)

        # check if the file is an image 
        if image_path.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                extracted_text = run_claude_pipeline(image_path)

                # build new row for each image and concat to results dataframe
                cleaned_str = extracted_text.strip().removeprefix("<output_format>").removesuffix("</output_format>").strip()
                new_row = json.loads(cleaned_str)
                print(new_row)
                
                new_row["image_path"] = image_path
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

            except Exception as e:
                print(f"An error occurred: {str(e)} on file {image_path}")

    results_df.to_csv("/Users/mvoong/Desktop/spark-symbiota-ml/transcription/results/claude_results_sample_50.csv")
    

