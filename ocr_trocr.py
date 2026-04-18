import cv2
import torch
import re
import numpy as np
import pandas as pd
import sys
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr

# 1. Setup Device and Load Models
# Using TrOCR (Vision Transformer + GPT-2)
print("Loading Open Source Models (TrOCR & EasyOCR)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(device)

# Initialize EasyOCR for label detection (Scout)
# We disable GPU for EasyOCR if you only have one GPU to save VRAM for TrOCR
scout = easyocr.Reader(['en'], gpu=(device == "cuda"))

def recognize_handwriting(image_crop):
    """Uses Microsoft TrOCR to turn an image of ink into text."""
    try:
        pixel_values = processor(images=image_crop, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    except Exception:
        return ""

def extract_spatial_open(image_path):
    orig_img = cv2.imread(image_path)
    if orig_img is None: return None
    h, w, _ = orig_img.shape

    # CROP: Standardize to the area where your form fields live
    # Adjust h//6 (top offset) and h//2 (bottom limit) as needed
    top_half = orig_img[h//6:h//2, 0:w]
    top_half_rgb = cv2.cvtColor(top_half, cv2.COLOR_BGR2RGB)

    # FIND LABELS using EasyOCR
    detections = scout.readtext(top_half)
    
    extracted_data = {
        "Filename": image_path.split('/')[-1] if '/' in image_path else image_path.split('\\')[-1],
        "App No.": "Not Found",
        "Name": "Not Found", 
        "Mobile": "Not Found"
    }

    for (bbox, text, conf) in detections:
        clean_text = text.lower()
        
        target_key = None
        if "application" in clean_text: target_key = "App No."
        elif "name" in clean_text: target_key = "Name"
        elif "mob" in clean_text or "mobile" in clean_text: target_key = "Mobile"

        if target_key:
            # bbox coordinates: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            x_end = int(bbox[1][0])
            y_top = int(bbox[0][1])
            y_bot = int(bbox[2][1])

            # CROP HANDWRITING: Take 600px to the right of the label
            # We add a 10px padding for better Transformer vision
            handwriting_crop = top_half_rgb[max(0, y_top-10):min(h, y_bot+10), 
                                          x_end:min(w, x_end+600)]

            if handwriting_crop.size > 0:
                pil_crop = Image.fromarray(handwriting_crop)
                val = recognize_handwriting(pil_crop)
                
                if target_key == "App No.":
                    # Find exactly 5 digits for Application Number
                    digits = re.search(r'(\d{5})', val)
                    extracted_data["App No."] = digits.group(1) if digits else "Not Found"
                
                elif target_key == "Mobile":
                    digits = re.sub(r'\D', '', val)
                    extracted_data["Mobile"] = digits[-10:] if len(digits) >= 10 else "Not Found"
                
                else:
                    extracted_data["Name"] = val.strip().title()

    return extracted_data

def main():
    # Accept image paths from shell script arguments
    image_list = sys.argv[1:]
    
    if not image_list:
        print("No images provided.")
        return

    results = []
    for p in image_list:
        print(f"Processing: {p}")
        data = extract_spatial_open(p)
        if data:
            results.append(data)
    
    if results:
        df = pd.DataFrame(results)
        print("\n" + df.to_markdown(index=False))
        df.to_csv("open_source_results.csv", index=False)
        print("\nResults saved to open_source_results.csv")

if __name__ == "__main__":
    main()