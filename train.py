# import torch
# import torch.optim as optim
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm # For a nice progress bar

# # Import our custom files
# from dataset import SegmentationDataset, SegmentationDataset
# from model import UNet

# # --- 1. Hyperparameters ---
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {DEVICE}")

# LEARNING_RATE = 1e-4
# BATCH_SIZE = 4       # Adjust based on your GPU memory
# NUM_EPOCHS = 10      # Start with a small number
# IMG_SIZE = (256, 256)
# TRAIN_IMG_DIR = "data/train/images/"
# TRAIN_MASK_DIR = "data/train/masks/"
# TEST_IMG_DIR = "data/test/images/"
# TEST_MASK_DIR = "data/test/masks/"
# MODEL_SAVE_PATH = "unet_model.pth"

# # Get number of classes from our dataset file
# NUM_CLASSES = SegmentationDataset.NUM_CLASSES

# # --- 2. Training Function ---
# def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
#     model.train() # Set model to training mode
    
#     loop = tqdm(loader, desc="Training")
#     total_loss = 0

#     for batch_idx, (data, targets) in enumerate(loop):
#         data = data.to(DEVICE)
#         targets = targets.to(DEVICE, dtype=torch.long) # Ensure targets are Long

#         # Forward pass
#         # Using autocast for mixed precision (faster training, less memory)
#         with torch.cuda.amp.autocast():
#             predictions = model(data)
#             loss = loss_fn(predictions, targets)

#         # Backward pass
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         # Update progress bar
#         total_loss += loss.item()
#         loop.set_postfix(loss=loss.item())
        
#     avg_loss = total_loss / len(loader)
#     print(f"Epoch avg. training loss: {avg_loss:.4f}")


# # --- 3. Validation Function ---
# def check_accuracy(loader, model, loss_fn):
#     model.eval() # Set model to evaluation mode
    
#     total_loss = 0
#     num_correct = 0
#     num_pixels = 0
    
#     print("Running validation...")
#     with torch.no_grad():
#         for data, targets in loader:
#             data = data.to(DEVICE)
#             targets = targets.to(DEVICE)
            
#             predictions = model(data)
#             loss = loss_fn(predictions, targets)
#             total_loss += loss.item()
            
#             # Check pixel accuracy
#             preds_max = torch.argmax(predictions, dim=1) # Get class index
#             num_correct += (preds_max == targets).sum().item()
#             num_pixels += torch.numel(targets)

#     avg_loss = total_loss / len(loader)
#     pixel_acc = (num_correct / num_pixels) * 100
#     print(f"Validation Loss: {avg_loss:.4f}, Pixel Accuracy: {pixel_acc:.2f}%")
#     model.train() # Set model back to train mode
#     return avg_loss


# # --- 4. Main Training Script ---
# def main():
#     # Initialize Model, Loss, Optimizer
#     # n_channels=3 (RGB), n_classes=NUM_CLASSES
#     model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
    
#     # We use CrossEntropyLoss.
#     # Note: 'Unlabeled' (index 0) will be penalized like any other class.
#     # We could ignore it by setting:
#     # loss_fn = nn.CrossEntropyLoss(ignore_index=0)
#     # But for now, let's train on all pixels.
#     loss_fn = nn.CrossEntropyLoss()
    
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
#     # Gradient scaler for mixed precision
#     scaler = torch.cuda.amp.GradScaler()

#     # Load Data
#     train_dataset = SegmentationDataset(
#         images_dir=TRAIN_IMG_DIR,
#         masks_dir=TRAIN_MASK_DIR,
#         img_size=IMG_SIZE
#     )
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=2, # Use background workers to load data
#         pin_memory=True,
#     )

#     test_dataset = SegmentationDataset(
#         images_dir=TEST_IMG_DIR,
#         masks_dir=TEST_MASK_DIR,
#         img_size=IMG_SIZE
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=2,
#         pin_memory=True,
#     )

#     # --- Start Training Loop ---
#     print(f"Starting training for {NUM_EPOCHS} epochs...")
#     best_val_loss = float('inf')

#     for epoch in range(NUM_EPOCHS):
#         print(f"\n--- Epoch {epoch+1} / {NUM_EPOCHS} ---")
#         train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)
        
#         # Check validation accuracy
#         val_loss = check_accuracy(test_loader, model, loss_fn)

#         # Save model if it has the best validation loss so far
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             print(f"New best model! Saving to {MODEL_SAVE_PATH}")
#             torch.save(model.state_dict(), MODEL_SAVE_PATH)

#     print("--- Training Complete ---")

# if __name__ == "__main__":
#     main()

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm # For a nice progress bar
import json # <-- NEW IMPORT

# Import our custom files
from dataset import SegmentationDataset
from model import UNet

# --- 1. Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

LEARNING_RATE = 1e-4
BATCH_SIZE = 4       
NUM_EPOCHS = 10      # Let's increase to 20 for a better graph
IMG_SIZE = (256, 256)
TRAIN_IMG_DIR = "data/train/images/"
TRAIN_MASK_DIR = "data/train/masks/"
TEST_IMG_DIR = "data/test/images/"
TEST_MASK_DIR = "data/test/masks/"
MODEL_SAVE_PATH = "unet_model.pth"
HISTORY_SAVE_PATH = "training_history.json" # <-- NEW

# Get number of classes from our dataset file
NUM_CLASSES = SegmentationDataset.NUM_CLASSES

# --- 2. Training Function ---
def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    model.train() # Set model to training mode
    
    loop = tqdm(loader, desc="Training")
    total_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE, dtype=torch.long)

        # Forward pass
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update progress bar
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    avg_loss = total_loss / len(loader)
    print(f"Epoch avg. training loss: {avg_loss:.4f}")
    return avg_loss # <-- MODIFIED

# --- 3. Validation Function ---
def check_accuracy(loader, model, loss_fn):
    model.eval() # Set model to evaluation mode
    
    total_loss = 0
    num_correct = 0
    num_pixels = 0
    
    print("Running validation...")
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
            
            # Check pixel accuracy
            preds_max = torch.argmax(predictions, dim=1) # Get class index
            num_correct += (preds_max == targets).sum().item()
            num_pixels += torch.numel(targets)

    avg_loss = total_loss / len(loader)
    pixel_acc = (num_correct / num_pixels) * 100
    print(f"Validation Loss: {avg_loss:.4f}, Pixel Accuracy: {pixel_acc:.2f}%")
    model.train() # Set model back to train mode
    return avg_loss, pixel_acc # <-- MODIFIED

# --- 4. Main Training Script ---
def main():
    model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    # Load Data
    train_dataset = SegmentationDataset(
        images_dir=TRAIN_IMG_DIR,
        masks_dir=TRAIN_MASK_DIR,
        img_size=IMG_SIZE
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = SegmentationDataset(
        images_dir=TEST_IMG_DIR,
        masks_dir=TEST_MASK_DIR,
        img_size=IMG_SIZE
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Start Training Loop ---
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    best_val_loss = float('inf')
    
    # <-- NEW: History tracking -->
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1} / {NUM_EPOCHS} ---")
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)
        val_loss, val_acc = check_accuracy(test_loader, model, loss_fn)

        # <-- NEW: Store history -->
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best model! Saving to {MODEL_SAVE_PATH}")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("--- Training Complete ---")
    
    # <-- NEW: Save history to file -->
    with open(HISTORY_SAVE_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {HISTORY_SAVE_PATH}")

if __name__ == "__main__":
    main()