#
## generate_psuedolabels.py
#
#  This script generates pseudo-label masks for the training and validation datasets using a pre-trained YOLOv8-seg model. It reads 
#  the unified manifest CSV files, processes each image to detect animals, and creates binary masks indicating the presence of animals. 
#  The generated masks are saved as .npy files, and the manifest is updated with references to these masks for later use in training of
#  the custom O-SegNet model.
#
#
#
#
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


TRAIN_MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/final_train_manifest.csv")
VAL_MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/final_val_manifest.csv")

IWILDCAM_RAW_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/train_images")
IDAHO_RAW_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/wolf_images")

MASK_OUTPUT_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_masks")
MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_masks_for_manifest(manifest_path):
    print(f"\n[Initialization] Loading YOLOv8-seg to generate pseudo-labels for: {manifest_path.name}")
    model = YOLO("yolov8n-seg.pt").to(DEVICE)
    animal_classes = [15, 16, 17, 18, 19, 20, 21, 22, 23] # COCO Animals
    
    df = pd.read_csv(manifest_path)
    mask_paths = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating Pseudo-Labels"):
        file_name = row['file_name']
        source = row['dataset_source']
        base_dir = IWILDCAM_RAW_DIR if source == 'iwildcam' else IDAHO_RAW_DIR
        full_path = base_dir / file_name
        
        # Output destination for this specific image's mask
        mask_filename = f"mask_{source}_{Path(file_name).stem}.npy"
        mask_save_path = MASK_OUTPUT_DIR / mask_filename
        
        if mask_save_path.exists():
            mask_paths.append(mask_filename)
            continue
            
        try:
            with Image.open(full_path).convert('RGB') as img:
                w, h = img.size
                results = model(img, verbose=False, device=DEVICE)
                
                # Default to an empty background mask (0s) if nothing is found
                final_mask = np.zeros((224, 224), dtype=np.uint8)
                
                for result in results:
                    if result.masks is not None:
                        # Iterate through boxes to find valid animal indices
                        for box, mask_data in zip(result.boxes, result.masks):
                            cls_id = int(box.cls[0].item())
                            if cls_id in animal_classes:
                                # Get binary mask resized down to O-SegNet training shape (224x224)
                                m_resized = mask_data.data[0].cpu().numpy()
                                m_pil = Image.fromarray((m_resized * 255).astype(np.uint8))
                                m_final = np.array(m_pil.resize((224, 224), resample=Image.NEAREST))
                                
                                # Combine overlapping masks if multiple animals exist
                                final_mask = np.maximum(final_mask, (m_final > 0).astype(np.uint8))
                
                # Serialize the 224x224 mask array efficiently to disk
                np.save(mask_save_path, final_mask)
                mask_paths.append(mask_filename)
                
        except Exception as e:
            mask_paths.append(None)
            
    # Add mask reference pointer to your manifest file tracking matrix
    df['pseudo_mask_name'] = mask_paths
    df = df.dropna(subset=['pseudo_mask_name'])
    
    updated_manifest_path = manifest_path.parent / f"pseudo_{manifest_path.name}"
    df.to_csv(updated_manifest_path, index=False)
    print(f"[Success] Generated pseudo-label tracking manifest: {updated_manifest_path}")

if __name__ == "__main__":
    generate_masks_for_manifest(TRAIN_MANIFEST_PATH)
    generate_masks_for_manifest(VAL_MANIFEST_PATH)