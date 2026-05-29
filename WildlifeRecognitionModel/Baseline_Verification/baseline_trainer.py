#
## baseline_trainer.py
#
#   This script serves as the foundational training loop for our wildlife classification model, utilizing a MobileNetV3-Small architecture. It is 
# designed to establish a performance baseline using the provided dataset of hybrid crops and raw images, with a specific focus on ensuring a 
# balanced representation of empty frames to enhance the model's ability to distinguish between presence and absence of wildlife.
#
#
#
#

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#
# Resources
MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_final_train_manifest.csv")
CROPS_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/hybrid_crops_v2")
RAW_IWILDCAM_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/train_images")
RAW_IDAHO_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/wolf_images")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

CLASS_MAP = {
    0: 0,   # Empty -> Class 0
    15: 1,  # Wolf -> Class 1
    11: 2,  # Coyote -> Class 2
    18: 3   # Dog -> Class 3
}

class BalancedWildlifeDataset(Dataset):
    def __init__(self, manifest_df, crops_dir, transform=None):
        self.transform = transform
        self.samples = []
        
        print("[Dataset] Indexing valid foreground crops and injecting negative controls...")
        manifest_lookup = {Path(row['file_name']).stem: int(row['category_id']) for _, row in manifest_df.iterrows()}
        
        # Gather all successfully generated foreground crops
        crop_files = list(crops_dir.glob("*.png"))
        for file_path in crop_files:
            name_str = file_path.name
            original_stem = name_str.replace("hybrid_wildlife_subject_", "").replace("hybrid_wolf_", "").split('.')[0]
            
            matched_id = manifest_lookup.get(original_stem)
            if matched_id in CLASS_MAP and matched_id != 0: # Pull positive profiles
                self.samples.append((file_path, CLASS_MAP[matched_id]))
                
        total_foregrounds = len(self.samples)
        
        # Inject a controlled 15% ratio of raw empty background samples (Class 0), giving  the model a clear concept of what an empty forest looks like
        target_empty_count = int(total_foregrounds * 0.15)
        empty_rows = manifest_df[manifest_df['category_id'] == 0]
        
        print(f"[Dataset] Found {total_foregrounds} crops. Injecting {target_empty_count} negative empty frames...")
        
        injected_empty = 0
        for _, row in empty_rows.iterrows():
            if injected_empty >= target_empty_count:
                break
            base_dir = RAW_IWILDCAM_DIR if row['dataset_source'] == 'iwildcam' else RAW_IDAHO_DIR
            raw_path = base_dir / row['file_name']
            
            if raw_path.exists():
                self.samples.append((raw_path, 0)) # 0 is the label for empty
                injected_empty += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path).convert('RGB') as img:
                if self.transform:
                    img = self.transform(img)
                return img, label
        except Exception as e:
            # Fallback to zero tensor if image reading fails momentarily during high disk load
            return torch.zeros(3, 224, 224), label


# Data augmentation and normalization for training, and just normalization for validation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_baseline_training():
    print(f"Using execution device: {DEVICE}")
    
    # Load Master Manifest Spreadsheet
    master_df = pd.read_csv(MANIFEST_PATH)
    
    # Split manifest row references to ensure no data leakage between train and validation sets
    train_df, val_df = train_test_split(master_df, test_size=0.20, random_state=42)
    
    train_dataset = BalancedWildlifeDataset(train_df, CROPS_DIR, transform=train_transforms)
    val_dataset = BalancedWildlifeDataset(val_df, CROPS_DIR, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Base Model Initialization: MobileNetV3-Small with pre-trained weights, modified for our 4-class output
    print("[Initialization] Loading MobileNetV3-Small pre-trained base model...")
    model = models.mobilenet_v3_small(pretrained=True)
    
    # Reconfigure final linear layer for our specific 4 outputs: [Empty, Wolf, Coyote, Dog]
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 4)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Model Optimization and Training Loop
    print("\n[Execution] Commencing baseline verification training loops...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Training")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.0 * correct_train / total_train)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * correct_train / total_train
        
        # Validation Assessment Block
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_acc = 100.0 * correct_val / total_val
        print(f" >> Summary Epoch {epoch+1}: Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
    # Save base weights to serve as comparison checkpoint
    torch.save(model.state_dict(), "baseline_mobilenetv3_small.pth")
    print("[Complete] Baseline weights successfully written to disk as 'baseline_mobilenetv3_small.pth'")

if __name__ == "__main__":
    run_baseline_training()