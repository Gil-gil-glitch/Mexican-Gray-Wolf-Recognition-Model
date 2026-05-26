import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models import OSegNet

# =====================================================================
# CONFIGURATION
# =====================================================================
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PSEUDO_TRAIN_MANIFEST = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_final_train_manifest.csv")
MASK_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_masks")
IWILDCAM_RAW_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/train_images")
IDAHO_RAW_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/wolf_images")
WEIGHTS_SAVE_PATH = "/home/greatgilbertsoco/WolfDetect/code/osegnet_weights.pth"

class CustomWildlifeSegmentationDataset(Dataset):
    def __init__(self, manifest_path):
        self.df = pd.read_csv(manifest_path)
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['file_name']
        source = row['dataset_source']
        mask_name = row['pseudo_mask_name']
        
        # Load Image
        base_dir = IWILDCAM_RAW_DIR if source == 'iwildcam' else IDAHO_RAW_DIR
        img_path = base_dir / file_name
        with Image.open(img_path).convert('RGB') as img:
            img_tensor = self.img_transform(img)
            
        # Load Pre-computed Pseudo Mask
        mask_path = MASK_DIR / mask_name
        mask_array = np.load(mask_path).astype(np.int64)
        mask_tensor = torch.from_numpy(mask_array) # Shape: [224, 224]
        
        return img_tensor, mask_tensor

def train_wildlife_domain():
    print("=====================================================================")
    print("        O-SEGNET WILDLIFE DOMAIN PSEUDO-LABEL TRAINING LOOP          ")
    print("=====================================================================")
    
    dataset = CustomWildlifeSegmentationDataset(PSEUDO_TRAIN_MANIFEST)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    
    model = OSegNet(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for images, targets in progress_bar:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        print(f"[Epoch Summary] Epoch {epoch} completed. Average Loss: {running_loss / len(loader):.4f}")
        
    torch.save(model.state_dict(), WEIGHTS_SAVE_PATH)
    print(f"[Success] Domain-adapted weights exported safely to: {WEIGHTS_SAVE_PATH}")

if __name__ == "__main__":
    train_wildlife_domain()