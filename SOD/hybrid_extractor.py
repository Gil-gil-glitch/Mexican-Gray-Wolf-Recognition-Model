#
## hybrid_pipeline.py
#
#  This script implements a hybrid pipeline that combines metadata-driven filtering with SOD-based foreground extraction to create high-fidelity 
#  crops of wildlife subjects from camera-trap images. The pipeline reads from a manifest CSV to identify valid animal profiles, applies SOD to 
#  extract the foreground subject, and saves the resulting crops while retaining the original class labels for each image. This approach ensures 
#  that we leverage both the cognitive metadata and structural information to produce cleaner training data for our models.
#
#
#
#

import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from rembg import remove, new_session
from tqdm import tqdm

# =====================================================================
# SYSTEM CONFIGURATION
# =====================================================================
MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_final_train_manifest.csv")
IWILDCAM_RAW_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/train_images")
IDAHO_RAW_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/wolf_images")

# VERSIONED OUTPUT: Automatically outputs to v2 to prevent overwriting/clashing
HYBRID_OUTPUT_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/hybrid_crops_v2")
HYBRID_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def execute_upgraded_production_pipeline():
    print("=====================================================================")
    print("      UPGRADED PRODUCTION PIPELINE: AGNOSTIC CASCADED MATTING       ")
    print("=====================================================================")
    
    if not MANIFEST_PATH.exists():
        print(f"[CRITICAL] Manifest file does not exist at: {MANIFEST_PATH}")
        return
        
    df = pd.read_csv(MANIFEST_PATH)
    total_rows = len(df)
    print(f"[Manifest] Loaded {total_rows} total rows to process.")
    
    # 1. Initialize YOLOv8 for spatial gating
    print("[Initialization] Loading YOLOv8 localization weights...")
    yolo_model = YOLO("yolov8n.pt")
    
    # Native COCO biological IDs that YOLO pre-trained weights recognize as living creatures
    COCO_ANIMAL_CLASSES = [15, 16, 17, 18, 19, 20, 21, 22, 23]
    
    # 2. Initialize BiRefNet for edge matting
    print("[Initialization] Staging local BiRefNet model on GPU accelerator...")
    provider_options = [{"device_id": "0"}]
    sod_session = new_session(
        "birefnet-general", 
        providers=["CUDAExecutionProvider"], 
        provider_options=provider_options
    )
    
    saved_count = 0
    skipped_empty_count = 0
    
    print(f"\n[Pipeline] Running extraction loop. Saving results to: {HYBRID_OUTPUT_DIR}")
    
    # Progress bar tracks manifest row index
    pbar = tqdm(total=total_rows, desc="Processing Dataset Rows")
    
    for idx, row in df.iterrows():
        pbar.update(1)
        
        file_name = row['file_name']
        source = row['dataset_source']
        
        # --- TECHNICAL GUARDRAIL 1: MANIFEST-BASED METADATA FILTER ---
        # Assuming your column is named 'id' or 'category_id'. Change 'id' to match your column name if needed.
        # If class mapping is 0, it is known empty. Skip instantly to save GPU cycles.
        try:
            class_id = int(row['category_id'])
            if class_id == 0:
                skipped_empty_count += 1
                continue
        except (KeyError, ValueError):
            # If the specific 'id' column isn't found, safely fall back entirely to YOLO's visual filter
            pass
            
        animal_class_label = "wolf" if source == "idaho_wolf" else "wildlife_subject"
        base_dir = IWILDCAM_RAW_DIR if source == 'iwildcam' else IDAHO_RAW_DIR
        img_path = base_dir / file_name
        
        if not img_path.exists():
            continue
            
        try:
            with Image.open(img_path).convert('RGB') as img:
                w, h = img.size
                
                # --- STAGE 1: CLASS-AGNOSTIC LOCALIZATION ---
                yolo_results = yolo_model(img, verbose=False)
                best_box = None
                highest_conf = -1.0
                
                for result in yolo_results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = box.conf[0].item()
                            
                            # Checks if YOLO recognizes ANY biological animal, ignoring custom CSV values
                            if cls_id in COCO_ANIMAL_CLASSES and conf > highest_conf:
                                highest_conf = conf
                                best_box = box.xyxy[0].cpu().numpy()
                
                # --- TECHNICAL GUARDRAIL 2: YOLO CONFIDENCE FILTER ---
                # Trashes image dynamically if no biological subject is found above 40% confidence
                if best_box is None or highest_conf < 0.40:
                    skipped_empty_count += 1
                    continue
                
                # --- STAGE 2: TIGHT CROP WITH CONTEXT BUFFER ---
                xmin, ymin, xmax, ymax = best_box
                pad_w = (xmax - xmin) * 0.10
                pad_h = (ymax - ymin) * 0.10
                
                crop_box = (
                    max(0, int(xmin - pad_w)),
                    max(0, int(ymin - pad_h)),
                    min(w, int(xmax + pad_w)),
                    min(h, int(ymax + pad_h))
                )
                coarse_crop = img.crop(crop_box)
                
                # --- STAGE 3: FINE MATTING & EDGE HARDENING ---
                rgba_output = remove(coarse_crop, session=sod_session)
                r, g, b, a = rgba_output.split()
                
                # Convert alpha mask to NumPy array for threshold operations
                alpha_array = np.array(a)
                
                # STEP FUNCTION: Binarizes mask to force sharp borders and delete fuzzy edges
                hard_alpha_array = np.where(alpha_array > 128, 255, 0).astype(np.uint8)
                hard_alpha_channel = Image.fromarray(hard_alpha_array)
                
                crisp_rgba = Image.merge("RGBA", (r, g, b, hard_alpha_channel))
                final_bbox = crisp_rgba.getbbox()
                
                if final_bbox:
                    tight_crop = crisp_rgba.crop(final_bbox)
                    save_name = f"hybrid_{animal_class_label}_{Path(file_name).name}"
                    save_path = HYBRID_OUTPUT_DIR / Path(save_name).with_suffix('.png')
                    tight_crop.save(save_path)
                    saved_count += 1
                    
        except Exception as e:
            continue
            
    pbar.close()
    print("\n=====================================================================")
    print("                PRODUCTION EXTRACTOR PROCESSING COMPLETE             ")
    print("=====================================================================")
    print(f"Total Profiles Saved to Disk (v2): {saved_count}")
    print(f"Total Skipped (Blank/Noise/ID 0):  {skipped_empty_count}")
    print(f"Output Directory Location:         {HYBRID_OUTPUT_DIR}")
    print("=====================================================================")

if __name__ == "__main__":
    execute_upgraded_production_pipeline()