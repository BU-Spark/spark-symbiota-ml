from PIL import Image
import pytesseract
import os
import pandas as pd
from io import StringIO

img_dir = "./data/new-england-samples/output"
img_path_list = os.listdir(img_dir) 
results_df = pd.DataFrame()

for image_path in img_path_list:
    # timeout/terminate the tesseract job after a period of time
    if image_path.lower().endswith((".png", ".jpg", ".jpeg")):
        try:
            full_image_path = os.path.join(img_dir, image_path)
            ocr_output = pytesseract.image_to_data(Image.open(full_image_path), timeout=30) # timeout after 30 seconds
            ocr_df = pd.read_csv(StringIO(ocr_output), sep="\t")

            # drop rows where there is no text
            ocr_df.dropna(subset=["text"], inplace=True)
            temp_df = ocr_df[ocr_df["text"] != " "] # remove rows where there is an empty string in "text" column 
            # joining all extracted text and aggregating the confidence score by mean
            cleaned_blocks = temp_df.groupby("block_num").agg({"text": " ".join,
                                                               "conf": "mean"})
            

            combined_df = pd.DataFrame({"image_path": image_path,
                                        "combined_text": [' '.join(cleaned_blocks["text"].astype(str))],
                                        "avg_conf": [cleaned_blocks["conf"].mean()]})
            
            results_df = pd.concat((results_df, combined_df), ignore_index=True)
            
        except RuntimeError as timeout_error:
            # tesseract processing is terminated
            print(f"Error: {timeout_error} on image located at: {image_path}")

results_df.to_csv("./results/tesseract_results.csv")