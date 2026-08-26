import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import openai
import os
import typing
import pandas as pd
import json
from dotenv import load_dotenv

# .env file should be in the directory 'transcription'
from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Works both as a script (cwd=transcription) and as a package import (from backend)
try:
    from transcription.confidence import build_confidence, detail_enabled, GbifTaxonMatcher, _tokenize
except ImportError:
    from confidence import build_confidence, detail_enabled, GbifTaxonMatcher, _tokenize

# document intelligence code from fall 2023 team 

example_result = \
    "Confidence Metrics: 'inches' 0.985, '111401' 0.562, '1' 0.572, '39.12' 0.965, '65.43' 0.723, '18.11' 0.931, '2' 0.876, '13.24' 0.753, '15.07' 0.876, '18.72' 0.695, '-22:29' 0.683, '3' 0.895, '4' 0.416, '44.26' 0.89, '13,80' 0.86, '22.85' 0.903, '49.8' 0.926, '5' 0.559, '55.56' 0.874, '9.82' 0.941, '-24.49' 0.934, '70.62' 0.652, '-33' 0.962, '43' 0.722, '0.35' 0.703, '351' 0.837, '34.26' 0.925, '59.60' 0.978, '39.92' 0.666, '11.81' 0.921, '-46.07' 0.646, '10' 0.961, '11' 0.688, '(A)' 0.617, '2.24' 0.748, '97.06' 0.961, '48.55' 0.86, '18.51' 0.947, '12' 0.985, '13' 0.61, '14' 0.792, '18' 0.558, '(B)' 0.531, '19' 0.703, 'Don' 0.552, 'Williams' 0.541, '26' 0.928, '27' 0.984, '1.' 0.513, '29' 0.993, '54.91' 0.831, '38.91' 0.904, '52.79' 0.961, '|50.87' 0.528, '50.88' 0.896, '30.77' 0.983, '28' 0.995, '82.74' 0.976, '3.45' 0.862, '30' 0.546, '43.96' 0.983, '52.00' 0.993, '30.01' 0.889, '-27.17' 0.951, '81.29.1' 0.845, '-12.72' 0.887, '-29.46' 0.584, 'Colors' 0.993, 'by' 0.995, 'Munsell' 0.976, 'Color' 0.993, 'Services' 0.993, 'Lab' 0.995, '-0.40' 0.843, '1.13' 0.79, '92.02' 0.958, '-0.60' 0.748, '-0.75' 0.852, '0.2' 0.736, '0.21' 0.737, '82.14' 0.876, '-1:06' 0.586, '2.06' 0.881, '-1.19' 0.962, '0.28' 0.878, '0.19' 0.948, '15' 0.685, '62.15' 0.614, '-1:07' 0.573, '16' 0.589, '(MA)' 0.518, '49.25' 0.913, '19.25' 0.614, '16.19' 0.579, '-0.05' 0.812, '8:29' 0.666, '20.81' 0.538, '211' 0.549, '3.44.' 0.682, '-0.23' 0.681, '0.49' 0.736, '22' 0.982, '23' 0.906, '24' 0.794, '72.46' 0.844, '24.45' 0.825, '55.93' 0.961, '72.95' 0.924, '16.83' 0.884, '68.80' 0.825, '25' 0.995, '29.3' 0.782, '3.06' 0.969, '-49.4' 0.598, '31.41' 0.961, '20.9' 0.633, '-19.43' 0.704, 'Density' 0.994, '0.15' 0.834, '0.22' 0.925, '0.36' 0.919, '0.51' 0.844, 'Golden' 0.93, 'Thread' 0.619, '0.98' 0.939, '1.24' 0.883, '1.67' 0.835, '2.04' 0.788, '2.42' 0.946, 'SN:' 0.993, 'OL0207' 0.973, 'centimeters' 0.577, 'Smithsonian' 0.905, 'Institution' 0.973, '0.01' 0.769, '17' 0.983, '38.6' 0.781, '0.18' 0.839, '0.54' 0.757, '-0.04' 0.877, '0.60' 0.559, '0.73' 0.94, '0.19' 0.976, 'D50' 0.995, 'Illuminant,' 0.789, '2' 0.995, 'degree' 0.993, 'observer' 0.985, '0.04' 0.86, '0.09' 0.918, '0.75' 0.86, 'US' 0.995, 'TATES' 0.984, 'BERNICE' 0.985, 'PAUAHI' 0.961, 'BISHOP' 0.994, 'MUSEUM' 0.987, 'HERBARIUM.' 0.966, 'HERBAR' 0.994, '1627083' 0.994, 'ED' 0.845, 'Apr' 0.442, '26-' 0.691, 'may' 0.44, '16' 0.971, '-' 0.976, '1912' 0.636, 'Collected' 0.934, 'by' 0.87, 'C.' 0.995, 'N.' 0.995, 'Forbes' 0.993, 'on' 0.995, 'Oahu.' 0.883, 'Flora' 0.992, 'Hawaiiensis.' 0.993, 'UNITED' 0.983, 'STATES' 0.992, 'NATIONAL' 0.983, 'MUSEUM' 0.994, 'Magalina' 0.294, 'Plopsy' 0.215, 'Koula.' 0.594, 'Clermontia' 0.781, 'persicifolia' 0.776, 'Gand.' 0.667, 'ONA!' 0.502, 'T!' 0.751, 'No.' 0.992, '1815.0.' 0.721, 'Image' 0.858, 'No.' 0.995, '00427028' 0.993."

example_output = {"recordedBy": "C. N. Forbes",
                  "location": "Oahu",
                  "scientificName": "Clermontia persicifolia Gaud.",
                  "eventDate": "1912-05-16", # Apr 26 - May 16, 1912
                  "barcode": "00427028",
                  "institutionCode": "Smithsonian Institution"}

def process_image(image_path: str):
    # Load Azure Cognitive Services (Document Intelligence) endpoint and key
    # Returns extracted text from processed image from Document Intelligence. 

    endpoint = "https://cursive-handwritings.cognitiveservices.azure.com/"
    key = os.environ["AZURE_DOCUMENT_KEY"]
    try:
        with open(image_path, "rb") as f:
            image_stream = f.read()

        document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, 
                                                          credential=AzureKeyCredential(key))

        poller = document_analysis_client.begin_analyze_document("prebuilt-read", image_stream)
        result = poller.result()
        
        return result      

    except Exception as e:
        return f"An error occurred while processing {image_path}: {e}"
    
def get_image_words(result: str) -> typing.Tuple[list, str]:
    # Collect words, their polygon data, and confidence scores. 
    # Returns the words and confidence scores.

    words = []
    confidence_text = ""
    for page in result.pages:
        for word in page.words:
            words.append({'content': word.content,
                          'polygon': word.polygon,
                          'confidence': word.confidence
            })
            confidence_text += "'{}' confidence {}\n".format(word.content, word.confidence)
    return words, confidence_text

def extract_info(text: str, example_result: str, example_output: str, temperature: float = 0.1):
    # Given text output from Document Intelligence, extract relevant information from text using LLM.
    # Returns response from GPT.
    # temperature: sampling temperature; raise it (e.g. 0.8) to draw diverse
    # samples for LLM self-consistency confidence, default 0.1 for production.

    # Set your OpenAI API key
    openai.api_key = os.environ["OPENAI_API_KEY"]

    prompt = f"Your goal is to translate (if necessary) and then extract six items from a string of text: the name of the specimen collector, the location the specimen was collected (as STRUCTURED administrative geography -- stateProvince, county, and the specific locality/place; if the state or county is not written explicitly, INFER it from the locality and region using your knowledge of US geography), the taxon name (genus and species, minimally) and/or any identifying information about the specimen, the date the specimen was collected, the barcode associated with the specimen, and the collection/institution code. For the taxon name, only output recognized species within the identified genus. Give the county WITHOUT the word 'County'. Use the best information available or insert 'UNKNOWN' if there is none. Here is an example input \n{example_result} and example output \n{example_output}.\n\nReturn ONLY a valid JSON object (no markdown, no commentary) with the six fields plus a 'confidence' object giving your confidence from 0 to 1 that each field is correct. Format:\n{{\"recordedBy\": ..., \"location\": {{\"stateProvince\": ..., \"county\": ..., \"locality\": ...}}, \"scientificName\": ..., \"eventDate\": ..., \"barcode\": ..., \"institutionCode\": ..., \"confidence\": {{\"recordedBy\": 0.0, \"location\": 0.0, \"scientificName\": 0.0, \"eventDate\": 0.0, \"barcode\": 0.0, \"institutionCode\": 0.0}}}}\n\nHere is your attempt: \n{text}"

    try:
        # Send the request to the API
        client = openai.OpenAI() 

        response = client.chat.completions.create(
            model="gpt-4o-mini", # gpt-4o-mini
            messages=[{"role": "system", "content": "You are a helpful assistant"},
                      {"role": "user", "content":prompt}],
            temperature=temperature, # default 0.1 for production; raise for self-consistency sampling
            response_format={"type": "json_object"} # enforce valid JSON output
        )

        # Extract the response
        if response.choices:
            return response.choices[0].message.content
        else:
            return "No response from the API."

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Vision confidence signal: an independent read of the six fields straight from
# the image (used for the eventDate + recordedBy ensembles). Behind a flag
# because it costs an extra gpt-4o-mini vision call per specimen.
VISION_PROMPT = (
    "You are reading a herbarium specimen sheet. From the image, extract exactly "
    "these fields and return ONLY a JSON object (no markdown): recordedBy "
    "(collector), location as {stateProvince, county, locality} (infer state/county "
    "from the town + region if not written; county WITHOUT the word 'County'), "
    "scientificName (genus species), eventDate (ISO), barcode (catalog number), "
    "institutionCode. Use 'UNKNOWN' if not legible."
)

_TAXON_MATCHER = None


def _taxon_matcher():
    # Lazily-built, cached GBIF backbone matcher for the scientificName signal.
    global _TAXON_MATCHER
    if _TAXON_MATCHER is None:
        _TAXON_MATCHER = GbifTaxonMatcher()
    return _TAXON_MATCHER


def _binomial(name: str) -> str:
    # genus + species pair (first two significant tokens), used to compare a text
    # read against a vision read for the P3 tiebreak. Matches the policy eval.
    return " ".join([t for t in _tokenize(name) if len(t) >= 3][:2])


def _flatten_location(loc):
    # Structured {stateProvince, county, locality} -> the middleware's flat
    # "locality, county, state" string (Option A). Plain strings pass through.
    if isinstance(loc, dict):
        def c(k):
            v = loc.get(k)
            return "" if (v is None or str(v).strip().upper() == "UNKNOWN") else str(v).strip()
        return ", ".join(p for p in (c("locality"), c("county"), c("stateProvince")) if p)
    return loc if loc is not None else ""


def _enhanced_enabled() -> bool:
    # The paid confidence signals (self-consistency + vision) are ON BY DEFAULT:
    # they are what gives every field a strong rho (~+0.37..+0.47). Without them
    # eventDate/recordedBy/barcode discrimination roughly thirds (rho +0.12/+0.15/
    # +0.20) -- too weak to route review on. Opt OUT for a cheaper, weaker score
    # (e.g. a scientificName-only use case) with CONFIDENCE_ENHANCED=0. When off,
    # eventDate falls back to date validity and recordedBy to the collector
    # gazetteer alone; scientificName/location/barcode keep their free signals.
    return os.environ.get("CONFIDENCE_ENHANCED", "1").strip().lower() not in (
        "0", "false", "no", "off", "")


def _self_consistency_samples(document_content: str, k: int = 3):
    # K re-extractions at temp 0.8; their agreement is the eventDate signal.
    out = []
    for _ in range(k):
        raw = extract_info(document_content, example_result, example_output, temperature=0.8)
        try:
            out.append(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            pass
    return out or None


def vision_extract_fields(image_path: str):
    # Independent vision read of the six fields (no OCR text). Returns None on any
    # failure so confidence scoring degrades gracefully.
    try:
        import base64
        import io
        from PIL import Image
        im = Image.open(image_path).convert("RGB")
        im.thumbnail((1536, 1536), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
            ]}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return None


def run_doc_intell_pipeline(image_path: str):
    # Check if the file is an image 
    if image_path.lower().endswith((".png", ".jpg", ".jpeg")):
        # Processing an image
        image_result = process_image(image_path)
        words, confidence_text = get_image_words(image_result)
        document_content = image_result.content + "\n\nConfidence Metrics:\n" + confidence_text
        extracted_info = extract_info(document_content, example_result, example_output)

        try:
            # Parse the LLM's JSON output (json.loads is safer than eval)
            data = json.loads(extracted_info)

            # Discard the LLM's subjective self-rating; confidence is derived from
            # reference checks (gazetteer / taxonomic backbone) and, when enabled,
            # an independent self-consistency + vision read.
            data.pop("confidence", None)
            data["image_path"] = image_path

            # Structured location -> preserve the parts in _meta and flatten to the
            # middleware's single "locality, county, state" string. The flat string
            # also lets location_confidence parse a state for the vision cross-check.
            if isinstance(data.get("location"), dict):
                data["_locationStructured"] = data["location"]
                data["location"] = _flatten_location(data["location"])

            sc_samples = vision_fields = None
            if _enhanced_enabled():
                sc_samples = _self_consistency_samples(document_content, k=3)
                vision_fields = vision_extract_fields(image_path)
                if vision_fields and isinstance(vision_fields.get("location"), dict):
                    vision_fields["location"] = _flatten_location(vision_fields["location"])

            matcher = _taxon_matcher()
            # Confidence is scored on the RAW read (so a FUZZY read gets ~0.5 and
            # is flagged) -- before we snap the value to GBIF's canonical name.
            data["confidence"] = build_confidence(
                data, words,
                sc_samples=sc_samples,
                vision_fields=vision_fields,
                taxon_matcher=matcher,
                detail=detail_enabled(),
            )

            # scientificName correction: replace the read with GBIF's accepted
            # species, preserving the verbatim read + match type. EXACT keeps a
            # valid/synonym name's verbatim as meaningful; FUZZY means the verbatim
            # was a misread. Lossless: nothing is discarded.
            raw_sci = data.get("scientificName", "")
            if raw_sci and str(raw_sci).strip().upper() != "UNKNOWN":
                corrected, match_type = matcher.correct(raw_sci)

                # P3 vision tiebreak: only when GBIF can't resolve the text read at
                # all do we fall back to the independent vision read -- and only if
                # GBIF resolves THAT to a different species. Conservative by design:
                # it recovers misreads that GBIF correction alone can't touch
                # (75.4% -> 80.5%, 27 fixed / 4 regressed on the GBIF-NE set) while
                # never overriding a name GBIF already validated. No-op without the
                # vision signal (CONFIDENCE_ENHANCED off).
                if match_type not in ("EXACT", "FUZZY"):
                    vis_sci = (vision_fields or {}).get("scientificName", "")
                    if vis_sci and str(vis_sci).strip().upper() != "UNKNOWN":
                        vis_corr, vis_mt = matcher.correct(vis_sci)
                        if vis_mt in ("EXACT", "FUZZY") and _binomial(vis_corr) != _binomial(raw_sci):
                            corrected, match_type = vis_corr, vis_mt
                            data["_scientificNameSource"] = "vision_tiebreak"

                data["_taxonMatchType"] = match_type
                if corrected and corrected != raw_sci:
                    data["verbatimScientificName"] = raw_sci
                    data["scientificName"] = corrected

            return json.dumps(data)
        except Exception as e:
            return f"An error occurred while processing {image_path}: {e}"

if __name__ == "__main__":
    input_img_path = "data/raw-images/437659994.jpg" # test image
    print(run_doc_intell_pipeline(input_img_path))