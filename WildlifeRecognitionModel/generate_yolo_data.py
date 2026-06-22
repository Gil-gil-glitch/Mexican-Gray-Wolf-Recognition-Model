#
## generate_yolo_data.py
#
#
#   This script generates YOLOv8-compatible training and validation datasets from the unified 
#   wildlife dataset. It extracts bounding boxes for target canid species (Mexican Gray Wolf, 
#   Coyote, Domestic Dog) and converts them into YOLO format.   
#
#
#


import os
import shutil
import json
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Input paths
MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_final_train_manifest.csv")
RAW_IWILDCAM_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/train_images")
RAW_IDAHO_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/wolf_images")

# Source original JSON annotations to extract the pixel bounding boxes
# Adjust paths if your raw json files are named slightly differently
IWILDCAM_JSON = Path("/home/greatgilbertsoco/WolfDetect/data/iwildcam2019_train_annotations.json")
IDAHO_JSON = Path("/home/greatgilbertsoco/WolfDetect/data/idaho-camera-traps.json")

# Output root
OUTPUT_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/yolo_gatekeeper")

def load_bbox_lookup_maps():
    """Builds an efficient filename -> bbox lookup from original source COCO JSONs"""
    bbox_map = {}
    
    # Parse iWildCam BBoxes if json exists
    if IWILDCAM_JSON.exists():
        print("[Setup] Indexing iWildCam bounding boxes...")
        with open(IWILDCAM_JSON, 'r') as f:
            data = json.load(f)
        # Map image_id to its bboxes
        img_id_to_file = {img['id']: img['file_name'] for img in data['images']}
        for ann in data['annotations']:
            if 'bbox' in ann and ann['image_id'] in img_id_to_file:
                f_name = img_id_to_file[ann['image_id']]
                bbox_map[f"iwildcam_{f_name}"] = ann['bbox']

    # Parse Idaho BBoxes if json exists
    if IDAHO_JSON.exists():
        print("[Setup] Indexing Idaho Camera Trap bounding boxes...")
        with open(IDAHO_JSON, 'r') as f:
            data = json.load(f)
        img_id_to_file = {img['id']: img['file_name'] for img in data['images']}
        for ann in data['annotations']:
            if 'bbox' in ann and ann['image_id'] in img_id_to_file:
                f_name = img_id_to_file[ann['image_id']]
                bbox_map[f"idaho_{f_name}"] = ann['bbox']
                
    return bbox_map

def convert_to_yolo_bbox(img_width, img_height, coco_bbox):
    """
    Converts COCO pixel format [xmin, ymin, width, height] 
    to normalized YOLO format [x_center, y_center, width, height].
    """
    xmin, ymin, box_w, box_h = coco_bbox
    
    # Calculate absolute center coordinates
    x_center = xmin + (box_w / 2.0)
    y_center = ymin + (box_h / 2.0)
    
    # Normalize values between 0.0 and 1.0
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = box_w / img_width
    height_norm = box_h / img_height
    
    return [x_center_norm, y_center_norm, width_norm, height_norm]

def build_yolo_dataset():
    if not MANIFEST_PATH.exists():
        print(f"[CRITICAL] Manifest not found at {MANIFEST_PATH}")
        return

    # Build bbox index mapping
    bbox_lookup = load_bbox_lookup_maps()

    print("[Loading] Reading unified manifest records...")
    df = pd.read_csv(MANIFEST_PATH)
    
    # Target only rows containing our key canids
    canid_df = df[df['category_id'].isin([11, 15, 18])].copy()
    
    # Perform a reproducible split (85% train, 15% validation)
    train_df = canid_df.sample(frac=0.85, random_state=42)
    val_df = canid_df.drop(train_df.index)
    
    splits = {"train": train_df, "val": val_df}
    
    for split_name, split_df in splits.items():
        print(f"\n[Processing] Extracting {len(split_df)} files for '{split_name}' split...")
        
        img_out_dir = OUTPUT_DIR / split_name / "images"
        lbl_out_dir = OUTPUT_DIR / split_name / "labels"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)
        
        copied_count = 0
        
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Writing {split_name}"):
            file_name = row['file_name']
            source = row['dataset_source']
            
            base_dir = RAW_IWILDCAM_DIR if source == 'iwildcam' else RAW_IDAHO_DIR
            img_path = base_dir / file_name
            
            if not img_path.exists():
                continue
                
            # Construct names matching lookup indexes
            lookup_key = f"{source}_{file_name}"
            unique_filename = f"{source}_{Path(file_name).name}"
            
            # Fetch coordinates
            coco_box = bbox_lookup.get(lookup_key)
            if coco_box is None:
                continue # Skip if no bounding box annotation found
                
            dest_img_path = img_out_dir / unique_filename
            dest_lbl_path = lbl_out_dir / f"{Path(unique_filename).stem}.txt"
            
            try:
                # 1. Get image boundaries
                with Image.open(img_path) as img:
                    w, h = img.size
                
                # 2. Normalize to YOLO layout
                yolo_box = convert_to_yolo_bbox(w, h, coco_box)
                
                # 3. Write out the structural text file
                # Class 0 designates 'Canid'
                with open(dest_lbl_path, "w") as f:
                    f.write(f"0 {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n")
                
                # 4. Copy corresponding photo
                shutil.copy(img_path, dest_img_path)
                copied_count += 1
                
            except Exception:
                continue
                
        print(f"[Finished] Successfully populated {copied_count} files into {split_name}.")

if __name__ == "__main__":
    build_yolo_dataset()