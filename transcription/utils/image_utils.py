from PIL import Image, ImageOps
import os 
import shutil 
import re 

# functions to resize images 
def resize_with_aspect_ratio(image_path, output_path, size=(1000, 1000)) -> bool:
    
    with Image.open(image_path) as image:
        # skip is size is already 1000x1000
        if image.size == size:
            return False

        # calculate the new size while maintaining the aspect ratio
        image.thumbnail(size, Image.Resampling.LANCZOS)
        
        # calculate the padding to make the image 1000x1000
        delta_w = size[0] - image.size[0]
        delta_h = size[1] - image.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        
        # add padding to the image
        new_image = ImageOps.expand(image, padding)
        
        # save the resized image
        new_image.save(output_path)

        return True

def resize_image(image_path: str) -> None:
    # resized images replace original image 
    i = os.path.basename(image_path)
    
    orig_size = os.path.getsize(image_path) / (1024 * 1024)

    if orig_size > 40: # only resize if the original size is above 40mb 
        print(f"Resizing {i} from {orig_size:.2f} MB...")
        resize_with_aspect_ratio(image_path, image_path)

# functions to check downloaded gbif images 
# check if downloaded images are in ground truth files because sometimes gbif images are downloaded from dwca, but ground truth metadata is mising. 
def build_gt_dictionary(gt_txt_file: str) -> dict:
    # build ground truth dictionaries 
    # takes an image path as input 
    # key: occurrence_id 
    # value: ground truth taxon, date, or geography 
    # returns a dictionary 
    data_dict = {}
    with open(gt_txt_file, "r") as f: 
        lines = f.readlines()
        for line in lines:
            try:
                pattern = r"(?<=\d):"  # Match ':' only if preceded by a digit
                key, value = re.split(pattern, line.strip())
                data_dict[key] = value
            except Exception as e:
                return f"Error reading line in file: {line}: {e}"
    return data_dict

def check_if_image_in_gt(gt_dictionary, input_dir_path):
    # using image directory path, check if images are in the ground truth dictionary 
    # return a list of image names with missing data 
    input_images = os.listdir(input_dir_path)
    missing_gt_data = []
    for image_name in input_images:
        occid = str(image_name.split("/")[-1].split(".jpeg")[0])       
        try:
            gt_value = gt_dictionary[occid]
        except Exception as e:
            missing_gt_data.append(occid)
    
    return missing_gt_data

def move_images(image_names: list, input_dir_path, dest_dir_path):
    # takes a list of image names (only contains occurrence id)
    # moves them to a separate folder - dest_dir_path

    for name in image_names:
        try:
            source_file_name = name + ".jpeg"
            source_path = os.path.join(input_dir_path, source_file_name)
            shutil.move(source_path, dest_dir_path)
            print(f"Moved image: {source_file_name}")
        except FileNotFoundError as fnfe:
            print(f"Error recieved: {fnfe} on file: {name}")