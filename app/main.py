import os
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from collections import defaultdict
import io
# --- ML Model Imports ---
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from model import UNet  # Import the UNet class from model.py

# --- 1. App & Model Configuration ---
app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (256, 256)
MODEL_PATH = "unet_model.pth"

# Define the class mapping (MUST match dataset.py)
# 0 is Unlabeled, 1 Building, 2 Land, 3 Road, 4 Veg, 5 Water
CLASSES = {
    0: "Unlabeled",
    1: "Building",
    2: "Land",
    3: "Road",
    4: "Vegetation",
    5: "Water",
}
NUM_CLASSES = len(CLASSES)

# Define the exact image transformations from training
image_transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 2. Load The Model (on startup) ---
print(f"Loading model from {MODEL_PATH} onto {DEVICE}...")
model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except FileNotFoundError:
    print(f"FATAL ERROR: {MODEL_PATH} not found.")
    print("Please copy your trained unet_model.pth into the 'app/' folder.")
    exit()
model.eval()  # Set model to evaluation mode
print("--- Model Loaded. Server Ready. ---")


# --- 3. Core Logic Functions ---

def run_model_prediction(image_path: str) -> np.ndarray:
    """Loads an image, runs it through the model, and returns a 2D numpy array of class indices."""
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return None
        
    # Apply transformations and add batch dimension
    img_tensor = image_transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(img_tensor)  # Output is [1, C, H, W]
    
    # Get the class with highest probability
    pred_mask = torch.argmax(logits, dim=1) # Output is [1, H, W]
    
    # Send to CPU and convert to numpy
    return pred_mask[0].cpu().numpy() # Output is [H, W]


def analyze_predicted_mask(mask_array: np.ndarray, sub_plot_id: int, image_id: int):
    """
    Analyzes a predicted mask (numpy array) based on the "Local Health" rules.
    VERSION 3: Allows building ON vegetation if neighborhood is healthy.
    """
    
    # --- Define our fixed "Local Health" thresholds ---
    MIN_IMAGE_VEG_RATIO = 0.15  # 15%
    MIN_IMAGE_WATER_RATIO = 0.0049 # 0.5%
    
    height, width = mask_array.shape
    sub_height = height // 3
    sub_width = width // 3
    
    # 1. Get stats for the specific sub-plot
    r = (sub_plot_id - 1) // 3
    c = (sub_plot_id - 1) % 3
    
    r_start = r * sub_height
    c_start = c * sub_width
    r_end = (r + 1) * sub_height if r < 2 else height
    c_end = (c + 1) * sub_width if c < 2 else width
    
    sub_plot_data = mask_array[r_start:r_end, c_start:c_end]
    
    # Count pixels in the sub-plot
    sub_plot_stats = defaultdict(int)
    indices, counts = np.unique(sub_plot_data, return_counts=True)
    for idx, count in zip(indices, counts):
        class_name = CLASSES.get(int(idx), "Other")
        sub_plot_stats[class_name] += int(count)

    # 2. Get stats for the *entire* predicted mask
    image_stats = defaultdict(int)
    indices, counts = np.unique(mask_array, return_counts=True)
    for idx, count in zip(indices, counts):
        class_name = CLASSES.get(int(idx), "Other")
        image_stats[class_name] += int(count)
        
    image_total = max(1, sum(image_stats.values())) # Avoid division by zero
    image_veg_ratio = image_stats.get("Vegetation", 0) / image_total
    image_water_ratio = image_stats.get("Water", 0) / image_total

    # 3. Create Debug Data Packet
    debug_data = {
        "Sub-Plot Vegetation": sub_plot_stats.get("Vegetation", 0),
        "Sub-Plot Water": sub_plot_stats.get("Water", 0),
        "Sub-Plot Land": sub_plot_stats.get("Land", 0),
        "Image Veg Ratio": image_veg_ratio,
        "Required Veg Ratio": MIN_IMAGE_VEG_RATIO,
        "Image Water Ratio": image_water_ratio,
        "Required Water Ratio": MIN_IMAGE_WATER_RATIO,
    }

    # --- 4. Run Rules ---
    # Rule 1: Check the sub-plot itself (Modified)
    
    # *** THE VEGETATION CHECK IS NOW REMOVED ***
    
    if sub_plot_stats.get("Water", 0) > 0:
        return {"status": "denied", "message": f"Denied: Sub-plot {sub_plot_id} contains Water.", "data": debug_data}

    if sub_plot_stats.get("Land", 0) == 0 and sub_plot_stats.get("Vegetation", 0) == 0:
        # This rule now says: you can't build on a road, building, or unlabeled area.
        # It MUST be either "Land" or "Vegetation".
        return {"status": "denied", "message": f"Denied: Sub-plot {sub_plot_id} is not developable (e.g., it's a road or existing building).", "data": debug_data}

    # Rule 2: Check the "Local Health" of the entire neighborhood (image)
    if image_veg_ratio < MIN_IMAGE_VEG_RATIO:
        msg = f"Denied: Neighborhood Veg ratio ({image_veg_ratio:.2%}) is below the required minimum ({MIN_IMAGE_VEG_RATIO:.2%})."
        return {"status": "denied", "message": msg, "data": debug_data}
    
    if image_water_ratio < MIN_IMAGE_WATER_RATIO:
        msg = f"Denied: Neighborhood Water ratio ({image_water_ratio:.2%}) is below the required minimum ({MIN_IMAGE_WATER_RATIO:.2%})."
        return {"status": "denied", "message": msg, "data": debug_data}

    # --- All checks passed! ---
    return {"status": "approved", "message": f"Approved: Sub-plot {sub_plot_id} is developable and neighborhood meets green-space requirements.", "data": debug_data}



# --- 4. API Endpoints ---

# This endpoint just gives the frontend the image paths
@app.get("/api/image_data/{image_id}")
async def get_image_data(image_id: int):
    return {
        "image_id": image_id,
        "image_url": f"/static/images/image{image_id:04d}.jpg",
        "mask_url": f"/static/masks/mask{image_id:04d}.png",
    }

# The main ML-powered endpoint
@app.get("/api/check_plot/{image_id}/{sub_plot_id}")
async def check_plot(image_id: int, sub_plot_id: int):
    if not (1 <= sub_plot_id <= 9):
        raise HTTPException(status_code=400, detail="Sub-plot ID must be 1-9")
        
    image_path = f"static/images/image{image_id:04d}.jpg"
    
    # 1. Run ML Model
    predicted_mask_array = run_model_prediction(image_path)
    
    if predicted_mask_array is None:
        raise HTTPException(status_code=404, detail=f"Image file not found: {image_path}")

    # 2. Run Analysis on the result
    result = analyze_predicted_mask(predicted_mask_array, sub_plot_id, image_id)
    
    return result
@app.post("/api/analyze_upload/{sub_plot_id}")
async def analyze_uploaded_plot(sub_plot_id: int, file: UploadFile = File(...)):
    """
    Analyzes a user-uploaded image.
    Receives a file and a sub_plot_id.
    """
    
    # 1. Read image from upload
    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # 2. Run ML Model (similar to run_model_prediction)
    # Apply transformations and add batch dimension
    img_tensor = image_transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(img_tensor)  # Output is [1, C, H, W]
    
    # Get the class with highest probability
    predicted_mask_array = torch.argmax(logits, dim=1)[0].cpu().numpy()

    # 3. Run Analysis
    # We pass image_id=0 as a placeholder, since it's not a pre-loaded image
    # The image_id is not used in the logic, only in debug messages.
    result = analyze_predicted_mask(predicted_mask_array, sub_plot_id, image_id=0)
    
    return result

# --- 5. Static Files and Root ---

# Serve the main index.html file
@app.get("/", response_class=FileResponse)
async def read_root():
    return FileResponse('index.html')

# Serve the CSS file
@app.get("/style.css", response_class=FileResponse)
async def read_style():
    return FileResponse('style.css')

# Serve the JS file
@app.get("/script.js", response_class=FileResponse)
async def read_script():
    return FileResponse('script.js')

# Mount the 'static' folder to serve images and masks
app.mount("/static", StaticFiles(directory="static"), name="static")