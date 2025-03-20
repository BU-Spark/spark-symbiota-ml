from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from google.cloud.documentai_v1.types import RawDocument
import os
import json
import openai
import random

from utils import image_utils

# code adapted from spring 2024 ml team 
openai.api_key = os.environ["OPENAI_API_KEY"] 

# DOCUMENT AI DETAILS
project_id = os.environ["GOOGLE_PROJECT_ID"]
processor_id = os.environ["GOOGLE_PROCESSOR_ID"]
location = "us"

# few shot examples 
shots = \
{
    "Chinese National Herbarium (PE) Plants of Xizang CHINA, Xizang, Lhoka City, Lhozhag County, Lhakang Town, Kharchhu Gompa vicinity 西藏自治区山南市洛扎县拉康镇卡久寺附近 28°5'37.15N, 91°7'24.74″E; 3934 m Herbs. Slopes near roadsides. PE-Xizang Expedition #PE6657 14 September 2017 M4 5 6 7 8 NCIL 中国数字植物标本馆 N° 2604988 西藏 TIBET 中国科学院 植物研究所 标本馆 PE CHINESE NATIONAL HERBARIUM (PE) 02334122 #PE6657 ASTERACEAE 菊科 Anaphalis contorta (D. Don) Hook. f. 鉴定人:张国进 Guo-Jin ZHANG 旋叶香青 17 May 2018"
    :{"recordedBy":"Guo-Jin, Zhang",
      "location":"Xizang Autonomous Region, Shannan City, Lhozhag County, Lhakang Town, Kharchhu Gompa vincinity, Slopes near roadsides",
      "scientificName":"Asteraceae; Anaphalis contorta (D. Don) Hook. f.",
      "eventDate":"14 September 2017",
      "barcode":"2604988",
      "institutionCode":"Chinese National Herbarium"
    },

    "PE-Xizang Expedition #PE6673 9 NSIT Chinese National Herbarium (PE) Plants of Xizang CHINA, Xizang, Lhoka City, Lhozhag County, Lhakang Town, Kharchhu Gompa vicinity 28°5&#39;37.15&quot;N, 91°7&#39;24.74&quot;E; 3934 m Herbs. Slopes near roadsides. PE-Xizang Expedition #PE6673 9 NSIT Chinese National Herbarium (PE) Plants of Xizang CHINA, Xizang, Lhoka City, Lhozhag County, Lhakang Town, Kharchhu Gompa vicinity 28°5&#39;37.15&quot;N, 91°7&#39;24.74&quot;E; 3934 m Herbs. Slopes near roadsides. PE-Xizang Expedition #PE6673 9 NSIT Chinese National Herbarium (PE) Plants of Xizang Spiral Leaf Green 17 May 2018"
    :{"recordedBy":"PE-Xizang Expedition #PE6673",
      "location":"Xizang Autonomous Region, Lhoka City, Lhozhag County, Lhakang Town, Kharchhu Gompa vicinity, Slopes near roadsides",
      "scientificName":"Spiral Leaf Green",
      "eventDate":"17 May 2018",
      "barcode":"UNKNOWN",
      "institutionCode":"Chinese National Herbarium"
    },

    "M.E. Naturalis Biodiversity Center Cimicifu SA EUROPAER ScHipcz. MEI 1975 4TT 3 AMD.115083 HUGO DE VRIES-LABORATORIU HORTUS BOTANICUS PLANTAGE MIDDENLAAN AMSTERDAM-C. 042566 RANLINELLACEAE Comicifuge foetida + Moravia: Frain. 18.711. 1888 A Obong. 22.85 2449 0.19 D50 Illuminant, 2 degree observer Density -0.04 0.09 0.15 0.22 0.36 0.51 Golden Thread 0.75 0.98 1.24 2.04 242 Colors by Munsell Color Services Lab"
    :{"recordedBy": "Oborny A",
      "location": "UNKNOWN",
      "scientificName": "Cimicifuga europaea Schipcz.",
      "eventDate": "11/18/1888",
      "barcode": "AMD.115083",
      "institutionCode": "Naturalis Biodiversity Center"
    },

}

# few-shot randomizer
def get_random_pairs_list(input_dict, num_pairs=3):
    if len(input_dict) < num_pairs:
        return "Not enough elements in the dictionary to select the requested number of pairs"
    keys = random.sample(list(input_dict.keys()), num_pairs)
    return [(key, input_dict[key]) for key in keys]

# using openai gpt4o to extract/generate metadata 
def generate_metadata(input_text, shots):
    # FEW-SHOT RANDOMIZER
    random_pairs = get_random_pairs_list(shots)
    
    # PROMPT FORMATTING
    prompt = \
    """
    Your goal is to translate (if necessary) and then extract six items from a string of text: the name of the specimen collector, the location the specimen was collected, the taxon name (genus and species, minimally) and/or any identifying information about the specimen, the date the specimen was collected, the barcode associated with the specimen, and the collection/institution code. . Your response should contain only the output in string format. For the taxon name, only output recognized species within the identified genus. Use the best information available or insert 'UNKNOWN' if there is none.

    Examples:

    Input 1:
    {shot1_input}
    Output 1:
    {{"recordedBy":"{shot1_output_collector}","location":"{shot1_output_location}","scientificName":"{shot1_output_taxon}","eventDate":"{shot1_output_date}","barcode":"{shot1_output_barcode}","institutionCode":"{shot1_output_code}"}}

    Input 2:
    {shot2_input}
    Output 2:
    {{"recordedBy":"{shot2_output_collector}","location":"{shot2_output_location}","scientificName":"{shot2_output_taxon}","eventDate":"{shot2_output_date}","barcode":"{shot2_output_barcode}","institutionCode":"{shot2_output_code}"}}

    Input 3:
    {shot3_input}
    Output 3:
    {{"recordedBy":"{shot3_output_collector}","location":"{shot3_output_location}","scientificName":"{shot3_output_taxon}","eventDate":"{shot3_output_date}","barcode":"{shot3_output_barcode}","institutionCode":"{shot3_output_code}"}}

    Your attempt:
    Input:
    {input_text}
    Output:

    """.format(
    shot1_input = random_pairs[0][0],
    shot1_output_collector = random_pairs[0][1]['recordedBy'],
    shot1_output_location = random_pairs[0][1]['location'],
    shot1_output_taxon = random_pairs[0][1]['scientificName'],
    shot1_output_date = random_pairs[0][1]['eventDate'],
    shot1_output_barcode = random_pairs[0][1]['barcode'],
    shot1_output_code = random_pairs[0][1]['institutionCode'],

    shot2_input = random_pairs[1][0],
    shot2_output_collector = random_pairs[1][1]['recordedBy'],
    shot2_output_location = random_pairs[1][1]['location'],
    shot2_output_taxon = random_pairs[1][1]['scientificName'],
    shot2_output_date = random_pairs[1][1]['eventDate'],
    shot2_output_barcode = random_pairs[1][1]['barcode'],
    shot2_output_code = random_pairs[1][1]['institutionCode'],

    shot3_input = random_pairs[2][0],
    shot3_output_collector = random_pairs[2][1]['recordedBy'],
    shot3_output_location = random_pairs[2][1]['location'],
    shot3_output_taxon = random_pairs[2][1]['scientificName'],
    shot3_output_date = random_pairs[2][1]['eventDate'],
    shot3_output_barcode = random_pairs[2][1]['barcode'],
    shot3_output_code = random_pairs[2][1]['institutionCode'],

    input_text = input_text
    )

    try:
        # Send the request to the API
        client = openai.OpenAI() 

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful assistant"},
                      {"role": "user", "content":prompt}],
            temperature=0.1 # default temperature = 1
        )

        # Extract the response
        if response.choices:
            return response.choices[0].message.content # get the content 
        else:
            return "No response from the API."

    except Exception as e:
        return f"An error occurred: {str(e)}"
    

# main document AI processor
def batch_process_documents(file_path: str, file_mime_type: str) -> tuple:
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    with open(file_path, "rb") as file_stream:
        raw_document = RawDocument(content=file_stream.read(), mime_type=file_mime_type)

    name = client.processor_path(project_id, location, processor_id)
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = client.process_document(request=request)

    extracted_text = result.document.text.replace('\n', ' ')
    return extracted_text

def run_google_vision_pipeline(image_path: str):
    if image_path.lower().endswith((".png", ".jpg", ".jpeg")):
        # check image resolution 
        # document ai has a restriction on image resolution to 40 megapixels per page
        image_utils.resize_image(image_path)

        try:
            # document ai processing 
            extracted_text = batch_process_documents(image_path, "image/jpeg")
            processed_text = generate_metadata(extracted_text, shots)

            # build result 
            result = eval(processed_text)
            result["image_path"] = image_path
            return json.dumps(result)

        except Exception as e:
            return f"An error occurred: {str(e)} on file {image_path}"

if __name__ == "__main__":
    input_image_path = "data/raw-images/1927995752.jpg" # test image
    print(run_google_vision_pipeline(input_image_path))
    