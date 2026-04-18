import re
import cv2
import numpy as np
import pandas as pd
import sys
import io
from google.cloud import vision_v1 as vision

def extract_tight_spatial(image_path):
    # Local ADC will be picked up automatically
    client = vision.ImageAnnotatorClient()

    img = cv2.imread(image_path)
    if img is None: 
        print(f"Warning: Could not read {image_path}")
        return None
        
    h, w, _ = img.shape

    # Crop: Remove header (10%) and keep until middle (55%)
    header_offset = int(h * 0.10)
    bottom_limit = int(h * 0.55)
    cropped_img = img[header_offset:bottom_limit, 0:w]

    _, encoded_image = cv2.imencode('.jpg', cropped_img)
    content = encoded_image.tobytes()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    annotations = response.text_annotations
    if not annotations: return None

    words = annotations[1:]

    def get_text_tight(label_keyword, y_thresh=25):
        anchor_y_top = None
        anchor_y_bottom = None
        anchor_x_end = None

        for word in words:
            if label_keyword.lower() in word.description.lower():
                v = word.bounding_poly.vertices
                anchor_y_top = v[0].y
                anchor_y_bottom = v[2].y
                anchor_x_end = v[1].x
                anchor_center_y = (anchor_y_top + anchor_y_bottom) / 2
                break

        if anchor_y_top is None: return None

        line_elements = []
        for word in words:
            v = word.bounding_poly.vertices
            word_center_y = (v[0].y + v[2].y) / 2
            if abs(word_center_y - anchor_center_y) < y_thresh and v[0].x > anchor_x_end:
                line_elements.append((v[0].x, word.description))

        line_elements.sort()
        return " ".join([t for x, t in line_elements])

    # --- Extraction ---
    raw_appl = get_text_tight("Application") or get_text_tight("No.:")
    appl_no = "Not Found"
    if raw_appl:
        match = re.search(r'(\d{5})', raw_appl)
        if match: appl_no = match.group(1)

    name_val = get_text_tight("Name", y_thresh=25)

    raw_mob = get_text_tight("Mob")
    mob_no = "Not Found"
    if raw_mob:
        clean_mob = re.sub(r'\D', '', raw_mob)
        if len(clean_mob) >= 10: mob_no = clean_mob[-10:]

    return {
        "Filename": image_path.split('/')[-1] if '/' in image_path else image_path.split('\\')[-1],
        "Appl No": appl_no,
        "Name": name_val.strip().title() if name_val else "Not Found",
        "Mob No": mob_no
    }

def main():
    # Take image list from system arguments (passed by the shell script)
    image_list = sys.argv[1:]
    
    if not image_list:
        print("No images provided.")
        return

    all_data = []
    for p in image_list:
        result = extract_tight_spatial(p)
        if result:
            all_data.append(result)
    
    if all_data:
        df = pd.DataFrame(all_data)
        print("\n--- FINAL EXTRACTION RESULTS ---")
        print(df.to_markdown(index=False))
        # Optional: Save to csv automatically
        df.to_csv("extracted_data.csv", index=False)
    else:
        print("No data extracted.")

if __name__ == "__main__":
    main()