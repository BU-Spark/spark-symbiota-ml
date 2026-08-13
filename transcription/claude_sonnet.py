import anthropic
import base64
import os 
import json
from dotenv import load_dotenv
from utils import image_utils

load_dotenv()

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


# Fields that can be scored with no OCR text and no second model call, and where the
# resulting score actually separates right from wrong (rho +0.42 / +0.43 on the
# 530-specimen set). The other three are omitted rather than shipped weak:
# eventDate's validity check and barcode's digit-run check both collapse to a
# constant without OCR words, and recordedBy's collector gazetteer barely moves
# (rho +0.08, 91% vs 87% between bands). A score that carries no information is
# worse than no score, because the UI presents it as one.
FREE_CONFIDENCE_FIELDS = ("scientificName", "location")


def build_free_confidence(data: dict) -> dict:
    """Per-field confidence from references that cost nothing to consult.

    scientificName -- GBIF taxonomic-backbone match type
    location       -- count of recognized places in the offline gazetteer

    Fields with no free signal are omitted, which envelope.py renders as
    "confidence unavailable" rather than as a low score. Scores are uncalibrated:
    transcription/calibration.json was fit on the Azure pipeline's distributions
    and does not transfer.
    """
    try:
        from transcription.confidence import (location_confidence,
                                              scientificname_confidence)
    except ImportError:
        from confidence import location_confidence, scientificname_confidence

    out = {}
    for field in FREE_CONFIDENCE_FIELDS:
        value = data.get(field, "")
        try:
            if field == "scientificName":
                score = scientificname_confidence(value, None, _taxon_matcher(), None)
            else:
                score = location_confidence(value, None, None)
        except Exception:
            score = None
        if score is not None:
            out[field] = score
    return out


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


def _correct_json(text: str) -> str:
    # Apply the taxon correction and free confidence to a JSON response. Anything
    # that does not parse is returned untouched so the caller's own error handling
    # still sees it.
    data = extract_json(text)
    if not isinstance(data, dict):
        return text
    # Score the raw read BEFORE correcting: a FUZZY match means the model misread
    # the name, which is exactly what confidence should flag. Correcting first would
    # make every name match GBIF and the signal would saturate at 1.0.
    data["confidence"] = build_free_confidence(data)
    data = apply_taxon_correction(data)
    return json.dumps(data, ensure_ascii=False)

def run_claude_pipeline(image_path, model=DEFAULT_MODEL, return_usage=False):
    # `model` lets the same prompt run across models (nbs/model_run.py); the prompt
    # must stay identical for those runs to stay comparable.
    # `return_usage` returns (text, usage) instead of text; usage is None on error.
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
                # Headroom for models that think before answering: max_tokens caps
                # thinking and response text together, and 1024 truncated some
                # replies mid-JSON. Only tokens actually generated are billed.
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
            result = _correct_json(result)
            if return_usage:
                u = message.usage
                return result, {"input_tokens": u.input_tokens,
                                "output_tokens": u.output_tokens,
                                "model": message.model}
            return result

        except Exception as e:
            err = f"An error occurred: {str(e)}"
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
    

