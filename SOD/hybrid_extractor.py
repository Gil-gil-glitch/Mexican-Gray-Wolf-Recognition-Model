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

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from rembg import remove, new_session
from tqdm import tqdm


MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_final_train_manifest.csv")
IWILDCAM_RAW_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/train_images")
IDAHO_RAW_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/wolf_images")

# Output directory 
HYBRID_OUTPUT_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/hybrid_crops")
HYBRID_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def execute_hybrid_pipeline():
    print("=====================================================================")
    print("          STARTING HYBRID METADATA + SOD EXTRACTION ENGINE           ")
    print("=====================================================================")
    
    df = pd.read_csv(MANIFEST_PATH)
    
    # This automatically leverages your RTX 2080 Ti via CUDA
    sod_session = new_session("u2net")
    
    # Filter out empty frames immediately using your manifest metadata
    print("[Data Filter] Filtering out known empty frames using manifest labels...")

    valid_records = df[df['dataset_source'] != 'empty'] # Adjust based on your 'empty' column logic
    
    # Process a sample of records to extract foregrounds using SOD, while retaining the original class labels for each image
    print(f"[Pipeline] Processing {len(valid_records)} verified animal profiles...")
    
    for idx, row in tqdm(valid_records.iterrows(), total=len(valid_records), desc="Extracting Foregrounds"):
        file_name = row['file_name']
        source = row['dataset_source']
        
        # 1. READ COGNITIVE METADATA (What the animal actually is)
        # We hold onto this label programmatically
        animal_class_label = "wolf" if source == "idaho_wolf" else "wildlife_subject"
        
        base_dir = IWILDCAM_RAW_DIR if source == 'iwildcam' else IDAHO_RAW_DIR
        img_path = base_dir / file_name
        
        if not img_path.exists():
            continue
            
        try:
            with Image.open(img_path).convert('RGB') as img:
                # 2. STRUCTURAL EXTRACTION (Let SOD handle the pixel boundaries)
                # rembg separates foreground from background, returning an RGBA image
                rgba_output = remove(img, session=sod_session)
                
                # Convert alpha channel into a bounding box crop to isolate the subject tightly
                bbox = rgba_output.getbbox()
                if bbox:
                    tight_crop = rgba_output.crop(bbox)
                    
                    # Save the high-fidelity crop
                    save_name = f"hybrid_{animal_class_label}_{Path(file_name).name}"
                    tight_crop.save(HYBRID_OUTPUT_DIR / save_name)
                    
        except Exception as e:
            # Skip corrupted or unreadable camera-trap files safely
            continue

    print(f"\n[Success] High-fidelity hybrid crops exported safely to: {HYBRID_OUTPUT_DIR}")

if __name__ == "__main__":
    execute_hybrid_pipeline()