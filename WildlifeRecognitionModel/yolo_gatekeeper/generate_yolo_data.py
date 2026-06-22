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
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Setup paths
MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/final_train_manifest.csv")
VAL_MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/final_val_manifest.csv")
RAW_IWILDCAM_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/train_images")
RAW_IDAHO_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/wolf_images")

OUTPUT_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/yolo_gatekeeper")

def convert_to_yolo_bbox(img_width, img_height, raw_box):
    """
    Convert raw pixel coordinates [xmin, ymin, xmax, ymax] or [xmin, ymin, width, height]
    to normalized YOLO format [x_center, y_center, width, height].
    Modify this function based on your exact source annotation format!
    """
    # Assuming standard COCO format: [xmin, ymin, width, height]
    xmin, ymin, box_w, box_h = raw_box
    
    x_center = xmin + (box_w / 2.0)
    y_center = ymin + (box_h / 2.0)
    
    # Normalize by image boundaries
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = box_w / img_width
    height_norm = box_h / img_height
    
    return [x_center_norm, y_center_norm, width_norm, height_norm]

def build_yolo_subset(manifest_path, split_name):
    print(f"Processing {split_name} split...")
    df = pd.read_csv(manifest_path)
    
    # Filter for your target canids (excluding empty images since we need bounding boxes)
    canid_df = df[df['category_id'].isin([11, 15, 18])]
    
    img_out_dir = OUTPUT_DIR / split_name / "images"
    lbl_out_dir = OUTPUT_DIR / split_name / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(canid_df.iterrows(), total=len(canid_df)):
        file_name = row['file_name']
        source = row['dataset_source']
        
        base_dir = RAW_IWILDCAM_DIR if source == 'iwildcam' else RAW_IDAHO_DIR
        img_path = base_dir / file_name
        
        if not img_path.exists():
            continue
            
        # Unique naming to prevent naming collisions across datasets
        unique_name = f"{source}_{Path(file_name).name}"
        dest_img_path = img_out_dir / unique_name
        dest_lbl_path = lbl_out_dir / f"{Path(unique_name).stem}.txt"
        
        try:
            # 1. Get dimensions for normalization
            with Image.open(img_path) as img:
                w, h = img.size
            
            # 2. Copy image over
            shutil.copy(img_path, dest_img_path)
            
            # 3. GET YOUR BOUNDING BOX HERE
            # Note: You need to pull the raw bounding box from your source json/csv.
            # Example placeholder: raw_box = get_box_from_metadata(row) 
            # For this example, let's assume 'bbox' column exists or you fetch it here.
            
            # Write out the YOLO text file
            with open(dest_lbl_path, "w") as f:
                # class_id is 0 for binary detector
                # yolo_box = convert_to_yolo_bbox(w, h, raw_box)
                # f.write(f"0 {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n")
                pass
                
        except Exception as e:
            continue

if __name__ == "__main__":
    # Run for both splits
    # build_yolo_subset(MANIFEST_PATH, "train")
    # build_yolo_subset(VAL_MANIFEST_PATH, "val")
    print("YOLO data exporter ready. Ensure your source bbox parsing is linked.")