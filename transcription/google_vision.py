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
    :{"Collector":"Guo-Jin, Zhang",
      "Location":"Xizang Autonomous Region, Shannan City, Lhozhag County, Lhakang Town, Kharchhu Gompa vincinity, Slopes near roadsides",
      "Taxon":"Asteraceae; Anaphalis contorta (D. Don) Hook. f.",
      "Date":"14 September 2017",
      "Barcode":"2604988",
      "Institution_code":"Chinese National Herbarium"
    },

    "PE-Xizang Expedition #PE6673 9 NSIT Chinese National Herbarium (PE) Plants of Xizang CHINA, Xizang, Lhoka City, Lhozhag County, Lhakang Town, Kharchhu Gompa vicinity 28°5&#39;37.15&quot;N, 91°7&#39;24.74&quot;E; 3934 m Herbs. Slopes near roadsides. PE-Xizang Expedition #PE6673 9 NSIT Chinese National Herbarium (PE) Plants of Xizang CHINA, Xizang, Lhoka City, Lhozhag County, Lhakang Town, Kharchhu Gompa vicinity 28°5&#39;37.15&quot;N, 91°7&#39;24.74&quot;E; 3934 m Herbs. Slopes near roadsides. PE-Xizang Expedition #PE6673 9 NSIT Chinese National Herbarium (PE) Plants of Xizang Spiral Leaf Green 17 May 2018"
    :{"Collector":"PE-Xizang Expedition #PE6673",
      "Location":"Xizang Autonomous Region, Lhoka City, Lhozhag County, Lhakang Town, Kharchhu Gompa vicinity, Slopes near roadsides",
      "Taxon":"Spiral Leaf Green",
      "Date":"17 May 2018",
      "Barcode":"UNKNOWN",
      "Institution_code":"Chinese National Herbarium"
    },

    "Honey Plants Research Institute of the Chinese Academy of Agricultural Sciences Collection No.: 13687. May 7, 1993 Habitat Roadside Altitude: 1600 * Characters Shrub No. Herbarium of the Institute of Botany, Chinese Academy of Sciences Collector 3687 Scientific Name Height: m (cm) Diameter at breast height m (cm) Flower: White Fruit: Notes Blooming period: from January to July Honey: Scientific Name: Rosa Sericea Lindl. Appendix: Collector: cm 1 2 3 4 25 CHINESE NATIONAL HERBARUM ( 01833954 No 1479566 * Herbarium of the Institute of Botany, Chinese Academy of Sciences Sichuan SZECHUAN DET. Rosa sercea Lindl. var. Various Zhi 2009-02-16"
    :{"Collector":"UNKNOWN",
      "Location":"Sichuan, China, Roadside, Altitude: 1600",
      "Taxon":"Rosa sericea",
      "Date":"7 May 1993",
      "Barcode": "01833954",
      "Institution_code": "Chinese Academy of Agricultural Sciences Collection"
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
    {{"Collector":"{shot1_output_collector}","Location":"{shot1_output_location}","Taxon":"{shot1_output_taxon}","Date":"{shot1_output_date}","Barcode":"{shot1_output_barcode}","Institution_code":"{shot1_output_code}"}}

    Input 2:
    {shot2_input}
    Output 2:
    {{"Collector":"{shot2_output_collector}","Location":"{shot2_output_location}","Taxon":"{shot2_output_taxon}","Date":"{shot2_output_date}","Barcode":"{shot2_output_barcode}","Institution_code":"{shot2_output_code}"}}

    Input 3:
    {shot3_input}
    Output 3:
    {{"Collector":"{shot3_output_collector}","Location":"{shot3_output_location}","Taxon":"{shot3_output_taxon}","Date":"{shot3_output_date}","Barcode":"{shot3_output_barcode}","Institution_code":"{shot3_output_code}"}}

    Your attempt:
    Input:
    {input_text}
    Output:

    """.format(
    shot1_input = random_pairs[0][0],
    shot1_output_collector = random_pairs[0][1]['Collector'],
    shot1_output_location = random_pairs[0][1]['Location'],
    shot1_output_taxon = random_pairs[0][1]['Taxon'],
    shot1_output_date = random_pairs[0][1]['Date'],
    shot1_output_barcode = random_pairs[0][1]['Barcode'],
    shot1_output_code = random_pairs[0][1]['Institution_code'],

    shot2_input = random_pairs[1][0],
    shot2_output_collector = random_pairs[1][1]['Collector'],
    shot2_output_location = random_pairs[1][1]['Location'],
    shot2_output_taxon = random_pairs[1][1]['Taxon'],
    shot2_output_date = random_pairs[1][1]['Date'],
    shot2_output_barcode = random_pairs[1][1]['Barcode'],
    shot2_output_code = random_pairs[1][1]['Institution_code'],

    shot3_input = random_pairs[2][0],
    shot3_output_collector = random_pairs[2][1]['Collector'],
    shot3_output_location = random_pairs[2][1]['Location'],
    shot3_output_taxon = random_pairs[2][1]['Taxon'],
    shot3_output_date = random_pairs[2][1]['Date'],
    shot3_output_barcode = random_pairs[2][1]['Barcode'],
    shot3_output_code = random_pairs[2][1]['Institution_code'],

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
    input_image_path = "data/raw-images/3734813368.jpg" # test image
    print(run_google_vision_pipeline(input_image_path))
    