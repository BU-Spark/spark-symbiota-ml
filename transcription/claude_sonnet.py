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

def run_claude_pipeline(image_path):
    
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
                model="claude-3-7-sonnet-20250219",
                max_tokens=1024,
                temperature = 0.1,
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
                                "text": "You are an expert in herbarium specimens and cursive handwriting. Perform OCR on this image and transcribe six items from the scanned herbarium specimen: the name of the specimen collector (recordedBy), the location the specimen was collected, the scientific name (genus and species, minimally) and/or any identifying information about the specimen, the event date the specimen was collected, the barcode associated with the specimen, and the institution code. Your response should contain only the output as a JSON object in plaintext. For the taxon name, only output recognized species within the identified genus. Only use the information available in the image or insert 'UNKNOWN' if there is none or if you are unsure. Your response should be formatted as such: "
                                "<output_format>{{OUTPUT_FORMAT}}</output_format>. "
                                "Here is the image you need to perform OCR on <input_image>{{encoded_image}}</input_image>"
                            }
                        ],
                    }
                ],
            )
            result = message.content[0].text
            return result
    
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
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
    

