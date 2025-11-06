import os
import glob
import shutil
from sklearn.model_selection import train_test_split
import pathlib
import ntpath # Import for better path handling

# --- 1. Configuration ---

# Get the directory where this script is located
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
print(f"Script location (base path): {SCRIPT_DIR}")

# !! NEW: We now have two extension variables !!
IMAGE_EXTENSION = ".jpg"
MASK_EXTENSION = ".png"

# Make paths absolute by joining them with the script's directory
IMAGE_DIR = os.path.join(SCRIPT_DIR, "images")
MASK_DIR = os.path.join(SCRIPT_DIR, "masks")

# Define the new base directory for our ML data
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")

# Define the split ratio (e.g., 80% train, 20% test)
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42

print("Script started. Configuring paths...")
print(f"Image directory: {IMAGE_DIR}")
print(f"Mask directory: {MASK_DIR}")

# --- 2. Define Output Directories ---
train_dir = os.path.join(BASE_OUTPUT_DIR, "train")
test_dir = os.path.join(BASE_OUTPUT_DIR, "test")

train_img_dir = os.path.join(train_dir, "images")
train_mask_dir = os.path.join(train_dir, "masks")
test_img_dir = os.path.join(test_dir, "images")
test_mask_dir = os.path.join(test_dir, "masks")

# Create these directories
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(test_mask_dir, exist_ok=True)

print(f"Created/verified directory structure at: {BASE_OUTPUT_DIR}")

# --- 3. Find and Pair All Files ---

# Define the search pattern for IMAGES
search_pattern = os.path.join(IMAGE_DIR, f"image*{IMAGE_EXTENSION}")
print(f"Searching for images using pattern: {search_pattern}")

# Get a list of all image files, sorted
all_image_paths = sorted(glob.glob(search_pattern))

# Check if we found any images
if not all_image_paths:
    print("\n--- FATAL ERROR ---")
    print(f"No images found matching pattern: {search_pattern}")
    print("Please check the following:")
    print(f"1. Does the folder '{IMAGE_DIR}' actually exist?")
    print(f"2. Are your images named 'imageXXXX{IMAGE_EXTENSION}' (e.g., image0001.jpg)?")
    print(f"3. Is the IMAGE_EXTENSION variable ('{IMAGE_EXTENSION}') correct?")
    print("Script cannot continue. Exiting.")
    exit()

print(f"Found {len(all_image_paths)} images.")

# --- NEW: Updated pairing logic ---
print("Pairing images with masks...")
paired_files = []
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
    
    # 6. Verify both files exist
    if os.path.exists(img_path) and os.path.exists(mask_path):
        paired_files.append((img_path, mask_path))
    else:
        print(f"Warning: Missing pair. Looked for '{mask_path}' but it wasn't found. Skipping.")

# --- End of new pairing logic ---

total_files = len(paired_files)
print(f"Found {total_files} valid image/mask pairs.")

# --- 4. Split the Data ---

# Check for 0 files
if total_files == 0:
    print("--- FATAL ERROR ---")
    print("Found 0 pairs. This means no masks matched the image names.")
    print("Please check your file naming convention. (e.g., 'image0001.jpg' -> 'mask0001.png')")
    print("Exiting.")
    exit()

print(f"Splitting data: {int(total_files * (1-TEST_SPLIT_SIZE))} training pairs, {int(total_files * TEST_SPLIT_SIZE)} test pairs.")
train_pairs, test_pairs = train_test_split(
    paired_files,
    test_size=TEST_SPLIT_SIZE,
    random_state=RANDOM_SEED
)

# --- 5. Copy Files to New Directories ---

def copy_files(file_pairs, dest_img_dir, dest_mask_dir):
    """Helper function to copy pairs to their destination."""
    count = 0
    for img_path, mask_path in file_pairs:
        img_name = os.path.basename(img_path)
        mask_name = os.path.basename(mask_path)
        
        dest_img_path = os.path.join(dest_img_dir, img_name)
        dest_mask_path = os.path.join(dest_mask_dir, mask_name)
        
        shutil.copy(img_path, dest_img_path)
        shutil.copy(mask_path, dest_mask_path)
        count += 1
    return count

# Copy training files
print("Copying training files...")
train_count = copy_files(train_pairs, train_img_dir, train_mask_dir)
print(f"Copied {train_count} training pairs.")

# Copy test files
print("Copying test files...")
test_count = copy_files(test_pairs, test_img_dir, test_mask_dir)
print(f"Copied {test_count} test pairs.")

print("\n--- Success! ---")
print(f"Data preparation complete. Your new data is in the '{BASE_OUTPUT_DIR}' folder.")