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

def execute_cascaded_pipeline():
    print("=====================================================================")
    print("         CASCADED PIPELINE: YOLOv8 FILTER + BiRefNet MATTING         ")
    print("=====================================================================")
    
    if not MANIFEST_PATH.exists():
        print(f"[CRITICAL] Manifest file does not exist at: {MANIFEST_PATH}")
        return
        
    df = pd.read_csv(MANIFEST_PATH)
    
    # 1. Initialize YOLOv8 for coarse localization
    print("[Initialization] Loading YOLOv8 localization weights...")
    yolo_model = YOLO("yolov8n.pt")
    # COCO animal class indices
    animal_classes = [15, 16, 17, 18, 19, 20, 21, 22, 23] 
    
    # 2. Initialize BiRefNet for fine-grained boundary extraction
    print("[Initialization] Staging local BiRefNet model on GPU accelerator...")
    provider_options = [{"device_id": "0"}]
    sod_session = new_session(
        "birefnet-general", 
        providers=["CUDAExecutionProvider"], 
        provider_options=provider_options
    )
    
    saved_count = 0
    fallback_count = 0
    
    print("\n[Pipeline Loop] Initiating cascaded extraction...")
    for idx, row in df.iterrows():
        file_name = row['file_name']
        source = row['dataset_source']
        animal_class_label = "wolf" if source == "idaho_wolf" else "wildlife_subject"
        
        base_dir = IWILDCAM_RAW_DIR if source == 'iwildcam' else IDAHO_RAW_DIR
        img_path = base_dir / file_name
        
        if not img_path.exists():
            continue
            
        try:
            with Image.open(img_path).convert('RGB') as img:
                w, h = img.size
                
                # --- STAGE 1: YOLO REGION FILTERING ---
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
                
                # Coarse crop selection
                if best_box is not None and highest_conf > 0.25:
                    xmin, ymin, xmax, ymax = best_box
                    
                    # Add a 10% spatial context padding buffer so BiRefNet sees the full outline edge
                    pad_w = (xmax - xmin) * 0.10
                    pad_h = (ymax - ymin) * 0.10
                    
                    crop_box = (
                        max(0, int(xmin - pad_w)),
                        max(0, int(ymin - pad_h)),
                        min(w, int(xmax + pad_w)),
                        min(h, int(ymax + pad_h))
                    )
                    coarse_crop = img.crop(crop_box)
                    used_fallback = False
                else:
                    # Fallback context window if YOLO misses entirely
                    coarse_crop = img.crop((int(w*0.05), int(h*0.05), int(w*0.95), int(h*0.95)))
                    used_fallback = True
                    fallback_count += 1
                
                rgba_output = remove(coarse_crop, session=sod_session)
                
                # Split channels to isolate the Alpha (transparency) layer
                r, g, b, a = rgba_output.split()
                
                # --- MASK HARDENING ENGINE ---
                # Convert Alpha channel to a numpy array to manipulate pixels directly
                alpha_array = np.array(a)
                
                # Force a hard threshold: if alpha > 128, make it completely solid (255)
                # Otherwise, make it completely transparent (0). This kills the feathering/blur.
                hard_alpha_array = np.where(alpha_array > 128, 255, 0).astype(np.uint8)
                
                # Reconstruct the hardened alpha channel
                hard_alpha_channel = Image.fromarray(hard_alpha_array)
                
                # Merge the crisp alpha channel back with original RGB channels
                crisp_rgba = Image.merge("RGBA", (r, g, b, hard_alpha_channel))
                
                # Tighten transparent borders to isolate the subject flawlessly
                final_bbox = crisp_rgba.getbbox()
                if final_bbox:
                    tight_crop = crisp_rgba.crop(final_bbox)
                    save_name = f"hybrid_{animal_class_label}_{Path(file_name).name}"
                    save_path = HYBRID_OUTPUT_DIR / Path(save_name).with_suffix('.png')
                    tight_crop.save(save_path)
                    saved_count += 1
                    
        except Exception as e:
            continue

        # Diagnostic constraint to inspect the first 50 images
        if saved_count >= 50:
            print(f"\n[Test Break] Staged 50 cascaded images for analysis.")
            break

    print("\n=====================================================================")
    print("                     CASCADED EXECUTION COMPLETE                     ")
    print("=====================================================================")
    print(f"Successfully Extracted Images: {saved_count}")
    print(f"Processed via YOLO Fallback:   {fallback_count}")
    print(f"Output Location:               {HYBRID_OUTPUT_DIR}")
    print("=====================================================================")

if __name__ == "__main__":
    execute_cascaded_pipeline()