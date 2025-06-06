import anthropic
import base64
import os 
from dotenv import load_dotenv
from utils import image_utils

load_dotenv()


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

# encoding image to base64 
image_path =  "/Users/mvoong/Desktop/spark-symbiota-ml/transcription/data/new-england-samples/output/1262197442.jpeg"
image_utils.resize_image(image_path)
base64_image = encode_image_to_base64(image_path)


# api call to claude 
# maximum 5MB image via API 

anthropic_key = os.environ["ANTHROPIC_API_KEY"]

client = anthropic.Anthropic(api_key=anthropic_key)
message = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": "Your goal is to transcribe and translate (if necessary) to English six items from the scanned herbarium specimen: the name of the specimen collector, the location the specimen was collected, the taxon name (genus and species, minimally) and/or any identifying information about the specimen, the date the specimen was collected, the barcode associated with the specimen, and the collection/institution code. Your response should contain only the output in JSON format. For the taxon name, only output recognized species within the identified genus. Only use the information available in the image or insert 'UNKNOWN' if there is none or if you are unsure."
                }
            ],
        }
    ],
)
print(message)


# to do:
# check image size 
# write prompt 
# few shot examples 
# Just as with document-query placement, Claude works best when images come before text. Images placed after text or interpolated with text will still perform well, but if your use case allows it, we recommend an image-then-text structure.

"""
Message(id='msg_01CFWCPPBZvkZDw7wa1PcHXB', content=[TextBlock(citations=None, text='Collector: Nathaniel Thayer Kidder\nLocation: UNKNOWN\nTaxon: Koelreuteria paniculata\nDate: July 20\nBarcode: UNKNOWN\nCollection/Institution: Herbarium Nathaniel Thayer Kidder', type='text')], model='claude-3-7-sonnet-20250219', role='assistant', stop_reason='end_turn', stop_sequence=None, type='message', usage=Usage(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=1428, output_tokens=64))
"""

"""
I want to specify that in the image sometimes there are multiple handwritten or typed labels that can be transcribed. If there is more than one label on the plant specimen that means that it has been updated by someone. I want the updated taxon name but everything else can be from the original label. You can tell which label is the original label because it will be dated and the date on the original label will be for before the updated label. 
"""