import json
import requests
from PIL import Image
import os
from fastapi import FastAPI, HTTPException, Query
from transcription.doc_intelligence import run_doc_intell_pipeline
from transcription.envelope import to_middleware_envelope

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "OCR service is running"}

@app.post("/azure")
async def evaluate(url: str = Query(...)):
    # Create a temporary file
    temp_filename = 'temp.jpg'

    # Download the image
    def download_image():
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("Failed to download image")
        with open(temp_filename, "wb") as f:
            f.write(response.content)

    try:
        download_image()
    except Exception as e:
        os.remove(temp_filename)
        raise HTTPException(status_code=400, detail=str(e))

    # Verify that the downloaded file is a valid image
    try:
        with Image.open(temp_filename) as img:
            img.verify()
    except Exception:
        os.remove(temp_filename)
        raise HTTPException(status_code=400, detail="Downloaded file is not a valid image")

    # Downloaded file path
    print("Downloaded image path:", temp_filename)

    azure_result = run_doc_intell_pipeline(temp_filename)

    # Clean up by deleting the temporary file
    os.remove(temp_filename)

    # Reshape into the middleware's flat DWC + _confidence envelope.
    try:
        data = json.loads(azure_result)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=502, detail=f"OCR pipeline error: {str(azure_result)[:200]}")
    return to_middleware_envelope(data, model="azure")