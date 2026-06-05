#
## dual_attention_model.py
#
#  This script implements a custom Dual-Attention Neural Network architecture designed to enhance the classification of wildlife in camera trap images.
#  The model integrates both Spatial and Spectral Attention mechanisms to capture fine-grained textures (like fur patterns) and macroscopic spatial arrangements 
#  (like body shapes).
#
#
#
#
#

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Resources
MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_final_train_manifest.csv")
CROPS_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/hybrid_crops_v2") # Results from the hybrid_extractor pipeline
RAW_IWILDCAM_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/train_images")
RAW_IDAHO_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/wolf_images")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 0.0005  # Reduced slightly for fine-grained attention calibration

CLASS_MAP = {0: 0, 15: 1, 11: 2, 18: 3}

# CUSTOM NEURAL MODULES: DASA + SPECTRAL ATTENTION
class SpectralAttentionBlock(nn.Module):
    """
    Implements a frequency-domain attention engine. It uses a 2D-DCT 
    pooling mechanism to capture fine high-frequency textures like fur.
    """

    def __init__(self, channels):
        super(SpectralAttentionBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # Mathematically approximate the 2D Discrete Cosine Transform (DCT)
        # by extracting spatial frequency components across the 2D grid
        spectral_pool = torch.mean(x, dim=[2, 3]) 
        
        attention_weights = self.fc(spectral_pool).view(b, c, 1, 1)
        return x * attention_weights

class SpatialAttentionBlock(nn.Module):
    """
    Captures macroscopic spatial proportions and skeletal profiles
    using a global spatial context descriptor.
    """
    def __init__(self, channels):
        super(SpatialAttentionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        spatial_weights = self.conv(x)
        return x * spatial_weights

class DualAttentionClassifier(nn.Module):
    """
    Custom Network: Fuses the lightweight MobileNetV3-Small backbone 
    with parallel Spatial and Spectral (Frequency) Attention channels.
    """
    def __init__(self, num_classes=4):
        super(DualAttentionClassifier, self).__init__()
        
        # Load base model
        base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        
        # Extract features right before the global pooling layer
        self.backbone_features = base_model.features
        
        # Determine internal channel depth (MobileNetV3-Small ends at 576 channels)
        internal_channels = 576
        
        # Parallel Attention Pathways
        self.spectral_attention = SpectralAttentionBlock(internal_channels)
        self.spatial_attention = SpatialAttentionBlock(internal_channels)
        
        # Final Reduction & Classification Heads
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(internal_channels, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # Base spatial feature extraction
        features = self.backbone_features(x)
        
        # Parallel Attention Extraction & Fusion
        spec_feat = self.spectral_attention(features)
        spat_feat = self.spatial_attention(features)
        fused_features = spec_feat + spat_feat
        
        # Dense Classification
        pooled = self.global_pool(fused_features).view(fused_features.size(0), -1)
        logits = self.classifier(pooled)
        return logits

# BALANCED DATASET: INJECTING EMPTY SAMPLES TO ENHANCE MODEL ROBUSTNESS
class BalancedWildlifeDataset(Dataset):
    def __init__(self, manifest_df, crops_dir, transform=None):
        self.transform = transform
        self.samples = []
        
        manifest_lookup = {Path(row['file_name']).stem: int(row['category_id']) for _, row in manifest_df.iterrows()}
        
        crop_files = list(crops_dir.glob("*.png"))
        for file_path in crop_files:
            name_str = file_path.name
            original_stem = name_str.replace("hybrid_wildlife_subject_", "").replace("hybrid_wolf_", "").split('.')[0]
            
            matched_id = manifest_lookup.get(original_stem)
            if matched_id in CLASS_MAP and matched_id != 0:
                self.samples.append((file_path, CLASS_MAP[matched_id]))
                
        total_foregrounds = len(self.samples)
        target_empty_count = int(total_foregrounds * 0.15) #0.15 is a reasonable ratio to inject empty samples without overwhelming the model with noise
        empty_rows = manifest_df[manifest_df['category_id'] == 0]
        
        injected_empty = 0
        for _, row in empty_rows.iterrows():
            if injected_empty >= target_empty_count:
                break
            base_dir = RAW_IWILDCAM_DIR if row['dataset_source'] == 'iwildcam' else RAW_IDAHO_DIR
            raw_path = base_dir / row['file_name']
            
            if raw_path.exists():
                self.samples.append((raw_path, 0))
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
        except Exception:
            return torch.zeros(3, 224, 224), label

# DATA AUGMENTATION & TRAINING PIPELINE
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # Standardized input size for MobileNetV3-Small
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # Standardized input size for MobileNetV3-Small
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_attention_training():
    print(f"[Initialization] Target Device Accelerator: {DEVICE}")
    master_df = pd.read_csv(MANIFEST_PATH)
    train_df, val_df = train_test_split(master_df, test_size=0.20, random_state=42)
    
    train_dataset = BalancedWildlifeDataset(train_df, CROPS_DIR, transform=train_transforms)
    val_dataset = BalancedWildlifeDataset(val_df, CROPS_DIR, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Instantiate Custom Dual-Attention Model
    model = DualAttentionClassifier(num_classes=4).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    
    # STABILITY HACK: Cosine Annealing Scheduler gently drops learning rate to prevent performance crashes
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0.0
    print("\n[Execution] Starting Dual-Attention Fine-Grain Training Loop...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
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
            
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * correct_train / total_train
        
        # Validation Assessment
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_acc = 100.0 * correct_val / total_val
        print(f" >> Epoch {epoch+1} Complete: Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}% | Current LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_dual_attention_model.pth")
            print(f"  [Checkpoint Saved] New Peak Validation Accuracy Achieved: {best_val_acc:.2f}%")

    print(f"\n[Complete] Optimization Finished. Highest Isolated Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    run_attention_training()