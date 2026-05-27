import os
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from ultralytics import YOLO

# Import your compiled architecture from models.py
from models import OSegNet


TRAIN_MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_final_train_manifest.csv")
IWILDCAM_RAW_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/train_images")
IDAHO_RAW_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/wolf_images")
WEIGHTS_PATH = "/home/greatgilbertsoco/WolfDetect/code/osegnet_weights.pth"

# Outputs for comparison
COMPARISON_OUT_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/segmentation_comparison")
YOLO_OUT_DIR = COMPARISON_OUT_DIR / "yolov8_output"
OSEG_OUT_DIR = COMPARISON_OUT_DIR / "osegnet_output"

YOLO_OUT_DIR.mkdir(parents=True, exist_ok=True)
OSEG_OUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# SHARED GEOMETRY ENGINE
# =====================================================================
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


# =====================================================================
# O-SEGNET INFERENCE MANAGER
# =====================================================================
class OSegNetInferenceWrapper:
    def __init__(self, weights_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = OSegNet(num_classes=2).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_crop(self, pil_image):
        width, height = pil_image.size
        # Prepare image tensor matching training dimensions
        tensor_in = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor_in)
            # Fetch predicted pixel classes (0 or 1)
            mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
        # Resize mask back to match raw input image dimensions
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_resized = np.array(mask_img.resize((width, height), resample=Image.NEAREST))
        
        # Locate positive coordinates where the animal mask was predicted
        where_animal = np.argwhere(mask_resized == 1)
        if len(where_animal) > 0:
            ymin, xmin = where_animal.min(axis=0)
            ymax, xmax = where_animal.max(axis=0)
            
            # Add a safe 5% pixel padding buffer around the mask boundaries
            pad_w = int((xmax - xmin) * 0.05)
            pad_h = int((ymax - ymin) * 0.05)
            
            left = max(0, xmin - pad_w)
            top = max(0, ymin - pad_h)
            right = min(width, xmax + pad_w)
            bottom = min(height, ymax + pad_h)
            
            return pil_image.crop((left, top, right, bottom)), "Mask-Detected"
            
        # Fallback if no clean mask clusters were found
        return pil_image.crop((int(width*0.1), int(height*0.1), int(width*0.9), int(height*0.9))), "Fallback"


# =====================================================================
# EVALUATION MATRIX PIPELINE
# =====================================================================
def run_evaluation_comparison(sample_count=20):
    if not TRAIN_MANIFEST_PATH.exists():
        print(f"[Error] Blueprint manifest not found at {TRAIN_MANIFEST_PATH}")
        return

    print("[Initialization] Loading YOLOv8 baseline...")
    yolo_model = YOLO("yolov8n.pt")
    animal_classes = [15, 16, 17, 18, 19, 20, 21, 22, 23]
    
    print("[Initialization] Loading Custom O-SegNet layers...")
    oseg_model = OSegNetInferenceWrapper(WEIGHTS_PATH)
    
    # =====================================================================
    # SMART FILTERING ENGINE (Replaces the random sampler)
    # =====================================================================
    print("[Data Filter] Scanning manifest to eliminate empty camera triggers...")
    df = pd.read_csv(TRAIN_MANIFEST_PATH)
    active_mask_directory = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_masks")
    
    positive_samples = []
    # Iterate through manifest to find rows with non-empty pseudo-masks
    for idx, row in df.iterrows():
        mask_path = active_mask_directory / row['pseudo_mask_name']
        if mask_path.exists():
            mask_data = np.load(mask_path)
            # Threshold: ensure at least 200 pixels belong to an active animal subject
            if np.sum(mask_data) > 200:
                positive_samples.append(row)
                if len(positive_samples) >= sample_count:
                    break
                    
    if len(positive_samples) == 0:
        print("[Error] No positive animal profiles found in pseudo_masks directory.")
        return
        
    samples = pd.DataFrame(positive_samples)
    print(f"[Data Filter] Successfully isolated {len(samples)} valid animal profiles for evaluation.")
    
    # =====================================================================
    # THE INFRASTRUCTURE EXECUTION LOOP (Keep this exactly as it was)
    # =====================================================================
    metrics = []
    print(f"\n[Running Execution] Testing {len(samples)} targeted samples across both architectures...\n")
    
    for idx, row in samples.iterrows():
        file_name = row['file_name']
        source = row['dataset_source']
        base_dir = IWILDCAM_RAW_DIR if source == 'iwildcam' else IDAHO_RAW_DIR
        full_path = base_dir / file_name
        
        if not full_path.exists():
            continue
            
        with Image.open(full_path).convert('RGB') as img:
            w, h = img.size
            
            # 1. METRIC TRACKING: YOLOv8 APPROACH
            start_time = time.time()
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
            
            if best_box is not None and highest_conf > 0.25:
                xmin, ymin, xmax, ymax = best_box
                yolo_crop = img.crop((max(0, int(xmin)), max(0, int(ymin)), min(w, int(xmax)), min(h, int(ymax))))
                yolo_status = "Box-Detected"
            else:
                yolo_crop = img.crop((int(w*0.07), int(h*0.07), int(w*0.93), int(h*0.93)))
                yolo_status = "Fallback"
                
            yolo_squared = pad_to_square(yolo_crop)
            yolo_time = time.time() - start_time
            yolo_squared.save(YOLO_OUT_DIR / f"yolo_{source}_{Path(file_name).name}")
            
            # 2. METRIC TRACKING: CUSTOM O-SEGNET
            start_time = time.time()
            oseg_crop, oseg_status = oseg_model.get_crop(img)
            oseg_squared = pad_to_square(oseg_crop)
            oseg_time = time.time() - start_time
            oseg_squared.save(OSEG_OUT_DIR / f"oseg_{source}_{Path(file_name).name}")
            
            # Append record
            metrics.append({
                "File": file_name,
                "Source": source,
                "YOLO_Time(s)": round(yolo_time, 4),
                "YOLO_Status": yolo_status,
                "OSeg_Time(s)": round(oseg_time, 4),
                "OSeg_Status": oseg_status
            })
            print(f"Processed: {Path(file_name).name} | YOLO: {yolo_time:.3f}s ({yolo_status}) | O-Seg: {oseg_time:.3f}s ({oseg_status})")

    # Display final metric summary dataframe
    print("\n=====================================================================")
    print("                 BENCHMARK PERFORMANCE MATRIX                        ")
    print("=====================================================================")
    summary_df = pd.DataFrame(metrics)
    print(summary_df.to_string(index=False))
    
    print(f"\n[Complete] Check cropped visual assets for quality assertions inside:")
    print(f" -> {YOLO_OUT_DIR}")
    print(f" -> {OSEG_OUT_DIR}")


if __name__ == "__main__":
    run_evaluation_comparison(sample_count=30)