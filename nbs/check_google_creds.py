"""Verify the Google Document AI credentials without spending OpenAI credit.

run_google_vision_pipeline calls Document AI and then gpt-4o-mini, so a failure
there is ambiguous. This runs the OCR half only.

    python nbs/check_google_creds.py
    python nbs/check_google_creds.py --image transcription/data/raw-images/437804966.jpg

Cost: one Document AI page, about $0.0015.
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / "transcription" / ".env")
sys.path.insert(0, str(ROOT / "transcription"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(ROOT / "transcription/data/raw-images/437804966.jpg"))
    args = ap.parse_args()

    project = os.environ.get("GOOGLE_PROJECT_ID", "")
    processor = os.environ.get("GOOGLE_PROCESSOR_ID", "")
    location = os.environ.get("GOOGLE_LOCATION", "us")
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

    print(f"project    {project or '(unset)'}")
    print(f"processor  {processor or '(unset)'}")
    print(f"location   {location}")
    print(f"creds      {creds or '(unset -- will fall back to gcloud ADC)'}")
    if creds and not os.path.exists(creds):
        raise SystemExit(f"\nFAIL: credential file not found at {creds}")
    if not project or not processor:
        raise SystemExit("\nFAIL: set GOOGLE_PROJECT_ID and GOOGLE_PROCESSOR_ID in transcription/.env")
    if not os.path.exists(args.image):
        raise SystemExit(f"\nFAIL: no image at {args.image}")

    from google.api_core.client_options import ClientOptions
    from google.cloud import documentai_v1 as documentai
    from google.cloud.documentai_v1.types import RawDocument

    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    name = client.processor_path(project, location, processor)
    print(f"\ncalling    {name}")

    with open(args.image, "rb") as f:
        raw = RawDocument(content=f.read(), mime_type="image/jpeg")
    doc = client.process_document(
        request=documentai.ProcessRequest(name=name, raw_document=raw), timeout=60
    ).document

    tokens = [t for p in doc.pages for t in p.tokens]
    confs = [t.layout.confidence for t in tokens]
    print(f"\nOK: {len(doc.text)} chars, {len(tokens)} tokens, "
          f"mean token confidence {sum(confs)/len(confs):.3f}" if confs else "\nOK but no tokens")
    print("\nfirst 300 chars:")
    print(doc.text.replace("\n", " ")[:300])


if __name__ == "__main__":
    main()
