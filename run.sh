#!/bin/bash

# 1. Install necessary Python packages
echo "--- Installing Python Packages ---"
pip install --upgrade google-cloud-vision opencv-python pandas tabulate numpy

# 2. Run Authentication Setup
echo "--- Setting up Google Cloud Authentication ---"
# We run this as a separate step. This will trigger the browser login.
python3 auth_setup.py

# 3. Prompt user for image paths
echo "------------------------------------------------"
echo "Enter image paths separated by space (e.g., 1.jpg 2.png):"
read -a user_images

# 4. Run the OCR script and pass the images as arguments
echo "--- Running OCR Extraction ---"
python3 ocr.py "${user_images[@]}"
