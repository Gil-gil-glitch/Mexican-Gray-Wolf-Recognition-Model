import os
import sys
import torch
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO


NUM_WORKERS = 8  # cores

IWILDCAM_RAW_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/train_images")
IDAHO_RAW_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/wolf_images")

TRAIN_MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/final_train_manifest.csv")
VAL_MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/final_val_manifest.csv")

OUTPUT_TRAIN_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/segmented_train_images")
OUTPUT_VAL_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/segmented_val_images")

OUTPUT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_VAL_DIR.mkdir(parents=True, exist_ok=True)


# Square Padding Engine
def pad_to_square(image, fill_color=(0, 0, 0)):
    width, height = image.size
    if width == height:
        return image
    
    if width > height:
        total_padding = width - height
        padding_top = total_padding // 2
        padding_bottom = total_padding - padding_top
        padding = (0, padding_top, 0, padding_bottom)
    else:
        total_padding = height - width
        padding_left = total_padding // 2
        padding_right = total_padding - padding_left
        padding = (padding_left, 0, padding_right, 0)
        
    return ImageOps.expand(image, padding, fill=fill_color)


# O-SegNet Background Filtering Engine
class BackgroundSegmenter:
    def __init__(self, use_cuda=True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        print(f"[Engine] Initializing YOLOv8 Segmentation Backend on: {self.device}")
        
        self.model = YOLO("yolov8n.pt")
        self.model.to(self.device)
        self.animal_classes = [15, 16, 17, 18, 19, 20, 21, 22, 23]
        
    def get_animal_crop(self, pil_image):
        width, height = pil_image.size
        
        # Run inference (Using a lock internally or keeping it local to worker thread)
        results = self.model(pil_image, verbose=False, device=self.device)
        
        best_box = None
        highest_conf = -1.0
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    
                    if cls_id in self.animal_classes and conf > highest_conf:
                        highest_conf = conf
                        best_box = box.xyxy[0].cpu().numpy()
        
        if best_box is not None and highest_conf > 0.25:
            xmin, ymin, xmax, ymax = best_box
            left = max(0, int(xmin))
            top = max(0, int(ymin))
            right = min(width, int(xmax))
            bottom = min(height, int(ymax))
            return pil_image.crop((left, top, right, bottom))
            
        crop_pct = 0.85
        left = int(width * (1 - crop_pct) / 2)
        top = int(height * (1 - crop_pct) / 2)
        right = int(width * (1 + crop_pct) / 2)
        bottom = int(height * (1 + crop_pct) / 2)
        return pil_image.crop((left, top, right, bottom))


# Worker function executed by individual CPU cores
def process_single_image(row_tuple, target_output_dir, segmenter):
    idx, row = row_tuple
    file_name = row['file_name']
    source = row['dataset_source']
    
    base_dir = IWILDCAM_RAW_DIR if source == 'iwildcam' else IDAHO_RAW_DIR
    full_path = base_dir / file_name
    
    try:
        with Image.open(full_path) as raw_img:
            rgb_img = raw_img.convert('RGB')
            animal_crop = segmenter.get_animal_crop(rgb_img)
            squared_subject = pad_to_square(animal_crop, fill_color=(0, 0, 0))
            
            new_file_name = f"seg_{source}_{Path(file_name).name}"
            save_path = target_output_dir / new_file_name
            
            squared_subject.save(save_path, "JPEG", quality=95)
            return idx, new_file_name
    except Exception as e:
        return idx, None


# Automated Background Filtering Pipeline with Multi-Threaded Core Scaling
def run_segmentation_pipeline(manifest_path, target_output_dir, segmenter):
    if not manifest_path.exists():
        print(f"[Error] Source manifest not found at {manifest_path}. Aborting.")
        return
        
    df = pd.read_csv(manifest_path)
    
    # Initialize placeholder column
    df['segmented_file_name'] = None
    
    print(f"\n[Processing] Distributing operations across {NUM_WORKERS} threads for: {manifest_path.name}")
    
    # Convert dataframe rows into a list of tuples for the thread pool
    rows = list(df.iterrows())
    results_map = {}
    
    # Execute image decoding, cropping, and encoding concurrently
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_image, row, target_output_dir, segmenter): row for row in rows}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Filtering Backgrounds"):
            idx, new_filename = future.result()
            results_map[idx] = new_filename

    # Map the unordered multi-threaded results back to their correct structural index
    df['file_name'] = df.index.map(results_map)
    
    # Clean out any files that completely failed to decode
    df = df.dropna(subset=['file_name'])
    
    output_manifest_name = f"segmented_{manifest_path.name}"
    output_manifest_path = manifest_path.parent / output_manifest_name
    df.to_csv(output_manifest_path, index=False)
    
    print(f"[Success] Generated updated file manifest at: {output_manifest_path}")
    print(f"[Success] Saved {len(df)} clean animal vectors inside: {target_output_dir}")


if __name__ == "__main__":
    print("=====================================================================")
    print("        O-SEGNET ISOLATION & MULTI-CORE STANDARD PIPELINE            ")
    print("=====================================================================")
    
    # Force PyTorch to utilize all reserved CPU cores for sub-operation threading
    torch.set_num_threads(NUM_WORKERS)
    
    segmenter_engine = BackgroundSegmenter(use_cuda=True)
    
    print("\n--- STAGE 1: TRAINING PARTITION PROCESSING ---")
    run_segmentation_pipeline(TRAIN_MANIFEST_PATH, OUTPUT_TRAIN_DIR, segmenter_engine)
    
    print("\n--- STAGE 2: VALIDATION PARTITION PROCESSING ---")
    run_segmentation_pipeline(VAL_MANIFEST_PATH, OUTPUT_VAL_DIR, segmenter_engine)
    
    print("\n=====================================================================")
    print("Pipeline Execution Complete. Multi-core performance metrics closed.")
    print("=====================================================================")