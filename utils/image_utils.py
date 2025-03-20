from PIL import Image, ImageOps
import os 

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

    if orig_size > 30: # only resize if the original size is above 30mb 
        print(f"Resizing {i} from {orig_size:.2f} MB...")
        resize_with_aspect_ratio(image_path, image_path)