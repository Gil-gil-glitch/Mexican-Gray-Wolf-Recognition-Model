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
HYBRID_OUTPUT_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/hybrid_crops")
HYBRID_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def execute_non_empty_pipeline(target_count=20):
    print("=====================================================================")
    print("      NON-EMPTY CASCADED PIPELINE: VALIDATED WILDLIFE SUBJECTS      ")
    print("=====================================================================")
    
    if not MANIFEST_PATH.exists():
        print(f"[CRITICAL] Manifest file does not exist at: {MANIFEST_PATH}")
        return
        
    df = pd.read_csv(MANIFEST_PATH)
    print(f"[Manifest] Loaded {len(df)} total rows.")
    
    # 1. Initialize YOLOv8 for spatial validation
    print("[Initialization] Loading YOLOv8 localization weights...")
    yolo_model = YOLO("yolov8n.pt")
    animal_classes = [15, 16, 17, 18, 19, 20, 21, 22, 23] # COCO animal IDs
    
    # 2. Initialize BiRefNet for fine matting
    print("[Initialization] Staging local BiRefNet model on GPU accelerator...")
    provider_options = [{"device_id": "0"}]
    sod_session = new_session(
        "birefnet-general", 
        providers=["CUDAExecutionProvider"], 
        provider_options=provider_options
    )
    
    saved_count = 0
    skipped_empty_count = 0
    
    print(f"\n[Pipeline] Scanning for {target_count} non-empty animal profiles...")
    
    # Progress bar tracks verified saves, not raw rows
    pbar = tqdm(total=target_count, desc="Extracting Foregrounds")
    
    for idx, row in df.iterrows():
        file_name = row['file_name']
        source = row['dataset_source']
        
        # Meta-exclusion: Skip explicitly logged empty background tracks
        if str(source).lower() == 'empty':
            continue
            
        animal_class_label = "wolf" if source == "idaho_wolf" else "wildlife_subject"
        base_dir = IWILDCAM_RAW_DIR if source == 'iwildcam' else IDAHO_RAW_DIR
        img_path = base_dir / file_name
        
        if not img_path.exists():
            continue
            
        try:
            with Image.open(img_path).convert('RGB') as img:
                w, h = img.size
                
                # --- STAGE 1: METADATA / OBJECT VERIFICATION ---
                yolo_results = yolo_model(img, verbose=False)
                best_box = None
                highest_conf = -1.0
                
                for result in yolo_results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = box.conf[0].item()
                            if cls_id in animal_classes and conf > highest_conf:
                                highest_conf = conf
                                best_box = box.xyxy[0].cpu().numpy()
                
                # CRITICAL GUARDRAIL: If no animal passes our confidence score,
                # it's treated as an empty frame/noise. Skip it entirely!
                if best_box is None or highest_conf < 0.40:
                    skipped_empty_count += 1
                    continue
                
                # --- STAGE 2: TIGHT LOCALIZATION CROP ---
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
                
                alpha_array = np.array(a)
                # Binarize mask to strip away blurry/feathered edge gradients entirely
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
                    pbar.update(1)
                    
        except Exception as e:
            continue

        # Halt automatically once your target batch of real animals is processed
        if saved_count >= target_count:
            break
            
    pbar.close()
    print("\n=====================================================================")
    print("                     PRODUCTION FILTER COMPLETE                      ")
    print("=====================================================================")
    print(f"Verified Non-Empty Saves:      {saved_count}")
    print(f"Skipped Empty/Noise Frames:    {skipped_empty_count}")
    print(f"Output Directory Location:     {HYBRID_OUTPUT_DIR}")
    print("=====================================================================")

if __name__ == "__main__":
    execute_non_empty_pipeline(target_count=50)