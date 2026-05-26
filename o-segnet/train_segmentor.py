import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Import your custom O-SegNet model from models.py
from models import OSegNet

# =====================================================================
# HYPERPARAMETERS & CONFIGURATION
# =====================================================================
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 5
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_SAVE_PATH = "/home/greatgilbertsoco/WolfDetect/code/osegnet_weights.pth"

# =====================================================================
# DATA PIPELINE (COCO TARGETED SEGMENTATION)
# =====================================================================
def get_segmentation_dataloader():
    """
    Downloads and prepares a localized semantic segmentation dataset.
    Normalizes images to 224x224 to match our structural verification test.
    """
    # Image transformations: Resize and convert to tensor
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Target mask transformations: Nearest neighbor resize to preserve class integers (0 or 1)
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    
    print("[Data] Initializing open-source animal segmentation loader (VOC/COCO standard)...")
    
    # Using Pascal VOC Segmentation as a lightweight, fast-downloading alternative to heavy COCO
    # It contains explicit pixel-level mask tags for dogs, horses, sheep, cows, and cats
    train_dataset = datasets.VOCSegmentation(
        root='/home/greatgilbertsoco/WolfDetect/data/benchmark_cache',
        year='2012',
        image_set='train',
        download=True,
        transform=img_transform,
        target_transform=mask_transform
    )
    
    loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        drop_last=True
    )
    return loader

# =====================================================================
# TARGET TRANSFORMATION LAYER
# =====================================================================
def preprocess_targets(target_masks):
    """
    Converts multi-class dataset masks into clean binary matrices:
    0 = Background / Environment Terrain
    1 = Foreground Animal Subject
    """
    # VOC contains background (0), specific classes (1-20), and border outlines (255)
    # Squeeze the channel dimension out [B, 1, H, W] -> [B, H, W]
    masks = (target_masks * 255).squeeze(1).long()
    
    # Map all valid animal classes (e.g., bird, cat, cow, dog, horse, sheep) to foreground (1)
    # For VOC 2012: bird=3, cat=8, cow=10, dog=12, horse=13, sheep=14
    animal_indices = [3, 8, 10, 12, 13, 14]
    
    binary_masks = torch.zeros_like(masks)
    for idx in animal_indices:
        binary_masks[masks == idx] = 1
        
    # Treat everything else (including background and borders) as background (0)
    return binary_masks.to(DEVICE)

# =====================================================================
# CORE TRAINING LOOP
# =====================================================================
def train_engine():
    print(f"=====================================================================")
    print(f"        O-SEGNET GUIDED-ATTENTION TRAINING PIPELINE                 ")
    print(f"=====================================================================")
    print(f"[Hardware] Active Compute Target: {DEVICE}")
    
    # 1. Initialize custom architecture
    model = OSegNet(num_classes=2).to(DEVICE)
    
    # 2. Define standard optimization and loss functions
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 3. Fetch dataloader stream
    train_loader = get_segmentation_dataloader()
    
    # 4. Process execution loops
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for images, raw_masks in progress_bar:
            images = images.to(DEVICE)
            targets = preprocess_targets(raw_masks)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass through Guided-Attention and Pyramid Pooling layers
            outputs = model(images) # Expected shape: [Batch, 2, 224, 224]
            
            # Compute segmentation loss matrix
            loss = criterion(outputs, targets)
            
            # Backward pass and weight optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        epoch_loss = running_loss / len(train_loader)
        print(f"[Epoch Summary] Epoch {epoch} completed. Average Loss: {epoch_loss:.4f}")
        
    # 5. Serialize custom weights to disk
    print(f"\n[Saving] Serialization process starting...")
    torch.save(model.state_dict(), WEIGHTS_SAVE_PATH)
    print(f"[Success] Custom O-SegNet weights securely saved to: {WEIGHTS_SAVE_PATH}")

if __name__ == "__main__":
    train_engine()