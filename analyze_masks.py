import os
import json
from PIL import Image
import numpy as np
from collections import defaultdict
import glob
import pathlib

# --- 1. Configuration ---
print("Configuring script...")

# Get the directory where this script is located
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
print(f"Script location (base path): {SCRIPT_DIR}")

# Paths to your *original* data folders
IMAGE_DIR = os.path.join(SCRIPT_DIR, "images")
MASK_DIR = os.path.join(SCRIPT_DIR, "masks")

IMAGE_EXTENSION = ".jpg"
MASK_EXTENSION = ".png"

# Your hex codes converted to (R, G, B) tuples
# This MUST match the classes in your dataset.py
CLASSES = {
    # Hex: #9B9B9B
    (155, 155, 155): "Unlabeled",
    # Hex: #3C1098
    (60, 16, 152): "Building",
    # Hex: #8429F6
    (132, 41, 246): "Land",
    # Hex: #6EC1E4
    (110, 193, 228): "Road",
    # Hex: #FEDD3A
    (254, 221, 58): "Vegetation",
    # Hex: #E2A929
    (226, 169, 41): "Water",
    # Add any other classes here
}

# --- 2. Main Script ---
print("Starting global mask analysis...")

# This will hold the grand totals for the global ratio
global_pixel_counts = defaultdict(int)

# Find all *images* first
search_pattern = os.path.join(IMAGE_DIR, f"image*{IMAGE_EXTENSION}")
all_image_paths = sorted(glob.glob(search_pattern))

if not all_image_paths:
    print(f"FATAL ERROR: No images found at {search_pattern}")
    print("Please make sure your 'images' folder is in the same directory as this script.")
    exit()

print(f"Found {len(all_image_paths)} images. Now finding and analyzing their masks...")

for img_path in all_image_paths:
    # 1. Get filename: "image0001.jpg"
    img_filename_with_ext = os.path.basename(img_path)
    # 2. Get filename without ext: "image0001"
    img_filename_no_ext = os.path.splitext(img_filename_with_ext)[0]
    # 3. Create mask filename: "mask0001"
    mask_filename_no_ext = img_filename_no_ext.replace("image", "mask")
    # 4. Add mask extension: "mask0001.png"
    mask_filename_with_ext = f"{mask_filename_no_ext}{MASK_EXTENSION}"
    # 5. Create full path
    mask_path = os.path.join(MASK_DIR, mask_filename_with_ext)

    if not os.path.exists(mask_path):
        print(f"Warning: Mask not found, skipping: {mask_path}")
        continue

    # --- This is the analysis part from the old project ---
    try:
        img = Image.open(mask_path).convert("RGB")
    except Exception as e:
        print(f"Error opening {mask_path}: {e}")
        continue
        
    img_data = np.array(img)
    
    # Reshape 3D array (H, W, 3) to 2D (H*W, 3)
    pixels = img_data.reshape(-1, 3)
    
    # Get all unique colors (pixels) and their counts
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    for color_tuple, count in zip(map(tuple, colors), counts):
        class_name = CLASSES.get(color_tuple, "Other")
        
        # Add to the *global* stats
        global_pixel_counts[class_name] += int(count)

    print(f"Processed {mask_filename_with_ext}")

# --- 3. Final Output ---
print("Analysis finished. Compiling final JSON...")

# This is the simplified output, we only need global stats
final_output = {
    "global_stats": global_pixel_counts,
    "image_data": {} # Empty, as our ML app doesn't need the per-image breakdown
}

# Save the analysis to a file
output_filename = "analysis.json"
with open(output_filename, "w") as f:
    json.dump(final_output, f, indent=2)

print(f"\nSUCCESS! Global stats saved to {output_filename}")
print("You can now copy this file to your 'app/' folder.")