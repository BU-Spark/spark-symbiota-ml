# Transcription Scripts

## Overview
- `/data/gt-labels` - contains ground truth `.txt` files (collector, taxon, location) for sample images, organized by occurrence id. 
- `/data/raw-images` - contains sample images from GBIF  to  test with 
- `/utils/image_utils.py` - script for resizing images to be processed by vision tools

This repository contains two Python scripts for document intelligence tasks:
1. `doc_intelligence.py` - Uses Azure Document Intelligence and gpt4o to extract text from herbarium specimens. 
2. `google_vision.py` - Uses Google Document AI and gpt4o to extract text from herbarium specimens. 
   
## Dependencies
Ensure that the reuired API credentials are set up before running the scripts. 
- `Azure Document Intelligence`
- `OpenAI API`
- `Google Document AI`

Set up the following environment variables:
- `AZURE_FORM_RECOGNIZER_KEY`
- `AZURE_FORM_RECOGNIZER_ENDPOINT`
- `GOOGLE_PROJECT_ID`
- `GOOGLE_PROCESSOR_ID`
- `OPENAI_API_KEY`

## Usage

### `doc_intelligence.py` and `google_vision.py`
These scripts extract structured information from herbarium specimens. They both take an `image_path` as an input and will return the information in `JSON` as seen below 

```
{ 
  "recordedBy": "C. N. Forbes",
  "location": "Oahu",
  "scientificName": "Clermontia persicifolia Gaud.",
  "eventDate": "Apr 26 - May 16, 1912",
  "barcode": "00427028",
  "institutionCode": "Smithsonian Institution"
}
