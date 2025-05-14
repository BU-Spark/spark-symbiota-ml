import requests
from PIL import Image
import os
from fastapi import FastAPI, HTTPException, Query
from transcription.doc_intelligence import run_doc_intell_pipeline

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

    return azure_result