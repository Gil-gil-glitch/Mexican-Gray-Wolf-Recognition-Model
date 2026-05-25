#
## Unified Wildlife Dataset
#
#  This file creates a unified wildlife dataset by combining the iWildCam2019 and Gray Wolf images from the LILAC BC dataset.    
#  Taking into account the findings from the EDAs, this dataset is designed to handle splitting by sequence frames instead 
#  of individual images, ensuring that all frames from a sequence are kept together in either the training or testing set. 
#  This approach helps to prevent data leakage and ensures that the model is evaluated on truly unseen data. The dataset 
#  is structured to facilitate training and evaluation of machine learning models for wildlife classification tasks, 
#  with a focus on maintaining the integrity of the sequences while providing a comprehensive set of images for both 
#  training and testing. The program will also convert all images into a consistent format of RGB. Image resizing must
#  be done separately since the datasets have different image dimensions and resizing may lead to loss of important features. 
#
#
## Result
#
# Total training images tracked: 160944
# Total validation images tracked: 40049


import os
import json
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit


# STEP 1 & 2: LEAK-PROOF SPLITTING & UNIFICATION
def generate_unified_manifests(iwild_df, idaho_df):
    """
    This function takes in the iWildCam and Idaho Wolf dataframes, performs a leak-proof split by sequence, and then
    combines the resulting training and validation sets into unified manifests. The iWildCam dataset is split based 
    on its 'seq_id' to ensure that all frames from a sequence are kept together. The Idaho Wolf dataset is similarly 
    split by its 'seq_id'. After splitting, the function concatenates the training and validation sets from both 
    datasets, ensuring that each entry is labeled with its source dataset for later reference in the custom PyTorch 
    Dataset class.
    """

    # 1. Split iWildCam by sequence
    gss_iwild = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    iwild_train_idx, iwild_val_idx = next(gss_iwild.split(iwild_df, groups=iwild_df['seq_id']))
    
    iwild_train = iwild_df.iloc[iwild_train_idx][['file_name', 'category_id']].copy()
    iwild_val = iwild_df.iloc[iwild_val_idx][['file_name', 'category_id']].copy()
    iwild_train['dataset_source'] = 'iwildcam'
    iwild_val['dataset_source'] = 'iwildcam'
    
    # 2. Split Idaho Wolf by sequence
    gss_wolf = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    wolf_train_idx, wolf_val_idx = next(gss_wolf.split(idaho_df, groups=idaho_df['seq_id']))
    
    idaho_train = idaho_df.iloc[wolf_train_idx][['local_file_name']].copy()
    idaho_val = idaho_df.iloc[wolf_val_idx][['local_file_name']].copy()
    
    idaho_train.columns = ['file_name']
    idaho_val.columns = ['file_name']
    idaho_train['category_id'] = 15 # Set explicit Wolf Category ID
    idaho_val['category_id'] = 15
    idaho_train['dataset_source'] = 'idaho_wolf'
    idaho_val['dataset_source'] = 'idaho_wolf'
    
    # 3. Concatenate Manifests
    train_manifest = pd.concat([iwild_train, idaho_train], ignore_index=True)
    val_manifest = pd.concat([iwild_val, idaho_val], ignore_index=True)
    
    return train_manifest, val_manifest


# STEP 3: CUSTOM PYTORCH WRAPPER

class UnifiedWildlifeDataset(Dataset):
    """
    A custom PyTorch Dataset class that loads images from both the iWildCam and Idaho Wolf datasets based on a unified manifest.
    """
    def __init__(self, manifest_df, iwild_dir, idaho_dir, transform=None):
        self.df = manifest_df.reset_index(drop=True)
        self.iwild_dir = Path(iwild_dir)
        self.idaho_dir = Path(idaho_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['file_name']
        source = row['dataset_source']
        label = int(row['category_id'])
        
        # Path resolution based on source dataset
        if source == 'iwildcam':
            img_path = self.iwild_dir / file_name
        elif source == 'idaho_wolf':
            img_path = self.idaho_dir / file_name
        else:
            raise ValueError(f"Unknown dataset source: {source}")
            
        # Error-resistant image loading and RGB enforcement
        try:
            with Image.open(img_path) as img:
                image = img.convert('RGB')
        except Exception as e:
            # Fallback canvas to protect heavy training loops from mid-run disk dropouts
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)



# STEP 4: EXECUTION BLOCK 

IWILDCAM_DIR = "/home/greatgilbertsoco/WolfDetect/data/train_images"
WOLF_DIR = "/home/greatgilbertsoco/WolfDetect/data/wolf_images"

CSV_PATH = "/home/greatgilbertsoco/WolfDetect/data/train.csv"
df = pd.read_csv(CSV_PATH)

JSON_PATH = "/home/greatgilbertsoco/WolfDetect/data/idaho-camera-traps.json"
with open(JSON_PATH, "r") as f:
    data = json.load(f)
df_idaho = pd.DataFrame(data, columns=["id", "location", "timestamp", "species", "image_path"])

if "images" in data:
    df_wolf_metadata = pd.DataFrame(data["images"])
else:
    df_wolf_metadata = pd.DataFrame(data)

local_files = set(os.listdir(WOLF_DIR)) #Filter local files to only those in the directory

df_wolf_metadata['local_file_name'] = df_wolf_metadata['file_name'].apply(lambda x: os.path.basename(x))
df_wolf = df_wolf_metadata[df_wolf_metadata['local_file_name'].isin(local_files)].copy()



train_manifest, val_manifest = generate_unified_manifests(df, df_wolf)


train_dataset = UnifiedWildlifeDataset(train_manifest, iwild_dir=IWILDCAM_DIR, idaho_dir=WOLF_DIR)
val_dataset = UnifiedWildlifeDataset(val_manifest, iwild_dir=IWILDCAM_DIR, idaho_dir=WOLF_DIR)

print("--- PIPELINE STRUCTURAL SUCCESS ---")
print(f"Total training images tracked: {len(train_dataset)}")
print(f"Total validation images tracked: {len(val_dataset)}")

sample_img, sample_label = train_dataset[0]
print(f"Dataset Test: Retrieved sample image format: {type(sample_img)} | Parsed Target Class Label: {sample_label.item()}")

train_manifest.to_csv("/home/greatgilbertsoco/WolfDetect/data/final_train_manifest.csv", index=False)
val_manifest.to_csv("/home/greatgilbertsoco/WolfDetect/data/final_val_manifest.csv", index=False)
print("Saved final standardized data manifests to data directory.")