#
## cascaded_dynamic_inference.py
#
#  This script implements a cascaded dynamic inference pipeline for wildlife recognition. It processes 
#  input images through multiple stages, applying increasingly complex models to filter out non-target 
#  species and focus on identifying wolves. The pipeline is designed to optimize computational 
#  resources while maintaining high accuracy in species identification.
#

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from rembg import remove, new_session
from torchvision import transforms
from tqdm import tqdm

# Import your custom network from Step 2
from dual_attention_model import DualAttentionClassifier

# Statistics for final output mapping
from sklearn.metrics import classification_report
import torch.nn.functional as F

# Resources
MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_final_train_manifest.csv")
RAW_IWILDCAM_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/train_images")
RAW_IDAHO_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/wolf_images")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output mapping for final display
INVERSE_CLASS_MAP = {0: "Empty Landscape", 1: "Mexican Gray Wolf", 2: "Coyote", 3: "Domestic Dog"}

# Custom Image Transform matching Step 2 Validation Configurations
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================================
# DYNAMIC SPECIES-SPECIFIC THRESHOLD DICTIONARY
# =====================================================================
# COCO Animal Classes mapping to lower risk-aware gating floors
DYNAMIC_THRESHOLDS = {
    16: 0.12,  # COCO Dog: Drop significantly to catch heavily camouflaged wild wolves/coyotes
    15: 0.18,  # COCO Cat: Low threshold for ambiguous or low-profile feline/canine structures
    17: 0.30,  # COCO Horse: Standard strict gate for large ungulate distractor shapes
    18: 0.30,  # COCO Sheep: Keep strict to prevent rocky textures from passing
}
DEFAULT_THRESHOLD = 0.25

def initialize_cascaded_pipeline():
    print("=====================================================================")
    print("      INITIALIZING DYNAMIC SPECIES-WEIGHTED INFERENCE SYSTEM        ")
    print("=====================================================================")
    
    # Stage 1 Gating Weights
    print("[Stage 1] Loading YOLOv8 localization gatekeeper...")
    yolo_model = YOLO("yolov8n.pt")
    
    # Stage 2 Edge Matting Session
    print("[Stage 2] Loading BiRefNet matting background removal session...")
    provider_options = [{"device_id": "0"}] if torch.cuda.is_available() else []
    providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    sod_session = new_session("birefnet-general", providers=providers, provider_options=provider_options)
    
    # Stage 3 Custom Dual-Attention Recognition Weights
    print("[Stage 3] Loading custom Dual-Attention Fine-Grain Classifier...")
    classifier = DualAttentionClassifier(num_classes=4)
    classifier_weights_path = Path("best_dual_attention_model.pth")
    
    if not classifier_weights_path.exists():
        print(f"[CRITICAL] Classifier weights not found at {classifier_weights_path}!")
        return None, None, None
        
    classifier.load_state_dict(torch.load(classifier_weights_path, map_location=DEVICE))
    classifier = classifier.to(DEVICE)
    classifier.eval()
    
    print("[Status] Adaptive multi-stage architecture loaded successfully.\n")
    return yolo_model, sod_session, classifier

def run_dynamic_pipeline():
    yolo_model, sod_session, classifier = initialize_cascaded_pipeline()
    if yolo_model is None:
        return

    df = pd.read_csv(MANIFEST_PATH)
    
    # Run on the exact same 10000-sample validation slice to directly compare with your static baseline
    TEST_SIZE = 10000
    print(f"[Sample Isolation] Drawing standard {TEST_SIZE} sample target matrix...")
    test_candidates = df[df['category_id'].isin([11, 15, 18])].sample(n=TEST_SIZE, random_state=101)
    
    all_true = []
    all_pred = []
    dropped_samples = 0
    processed_samples = 0
    dropped_files = []

    print("=====================================================================")
    print("             LAUNCHING COST-SENSITIVE PIPELINE EXECUTION             ")
    print("=====================================================================")
    
    COCO_ANIMAL_CLASSES = [15, 16, 17, 18, 19, 20, 21, 22, 23]
    
    for idx, row in tqdm(test_candidates.iterrows(), total=TEST_SIZE, desc="Processing Adaptive Pipeline"):
        file_name = row['file_name']
        source = row['dataset_source']
        true_id = int(row['category_id'])
        
        true_label = "Mexican Gray Wolf" if true_id == 15 else ("Coyote" if true_id == 11 else "Domestic Dog")
        base_dir = RAW_IWILDCAM_DIR if source == 'iwildcam' else RAW_IDAHO_DIR
        img_path = base_dir / file_name

        mapping = {15: 1, 11: 2, 18: 3, 0: 0}
        true_mapped = mapping.get(true_id, 0)

        if not img_path.exists():
            continue
        
        try:
            with Image.open(img_path).convert('RGB') as img:
                w, h = img.size
                
                # --- STAGE 1: ADAPTIVE SPATIAL GATING ---
                yolo_results = yolo_model(img, verbose=False)
                best_box = None
                highest_conf = -1.0
                
                for result in yolo_results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = box.conf[0].item()
                            
                            if cls_id in COCO_ANIMAL_CLASSES:
                                # Apply the dynamic species-weighted threshold value
                                dynamic_gate = DYNAMIC_THRESHOLDS.get(cls_id, DEFAULT_THRESHOLD)
                                
                                if conf > highest_conf and conf >= dynamic_gate:
                                    highest_conf = conf
                                    best_box = box.xyxy[0].cpu().numpy()
                
                # Risk-Aware Decision Check
                if best_box is None:
                    dropped_samples += 1
                    dropped_files.append({
                        'dropped_file_names': file_name,
                        'true_label': true_label,
                        'true_id': true_id
                    })
                    continue
                
                # --- STAGE 2: BACKGROUND STRIPPING AND MASK HARDENING ---
                xmin, ymin, xmax, ymax = best_box
                pad_w, pad_h = (xmax - xmin) * 0.10, (ymax - ymin) * 0.10
                crop_box = (max(0, int(xmin - pad_w)), max(0, int(ymin - pad_h)), min(w, int(xmax + pad_w)), min(h, int(ymax + pad_h)))
                coarse_crop = img.crop(crop_box)
                
                rgba_output = remove(coarse_crop, session=sod_session)
                r, g, b, a = rgba_output.split()
                
                alpha_array = np.array(a)
                hard_alpha_array = np.where(alpha_array > 128, 255, 0).astype(np.uint8)
                hard_alpha_channel = Image.fromarray(hard_alpha_array)
                
                crisp_rgba = Image.merge("RGBA", (r, g, b, hard_alpha_channel))
                final_bbox = crisp_rgba.getbbox()
                
                if not final_bbox:
                    dropped_samples += 1
                    dropped_files.append({
                        'dropped_file_names': file_name,
                        'true_label': true_label,
                        'true_id': true_id
                    })
                    continue
                    
                final_silhouette = crisp_rgba.crop(final_bbox).convert('RGB')
                
                # --- STAGE 3: FINE-GRAIN DUAL ATTENTION EVALUATION ---
                input_tensor = inference_transforms(final_silhouette).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = classifier(input_tensor)
                    probabilities = F.softmax(logits, dim=1)
                    _, predicted_idx = torch.max(probabilities, 1)
                
                processed_samples += 1
                all_true.append(true_mapped)
                all_pred.append(predicted_idx.item())

        except Exception as e:
            continue

    # Final Summary Results
    print("\n" + "="*50)
    print("ADAPTIVE PERFORMANCE METRICS RESULT")
    print("="*50)
    print(f"Pipeline Complete: {processed_samples} Successes | {dropped_samples} Dropped Frame Rejections.")

    if dropped_files:
        df_dropped = pd.DataFrame(dropped_files)
        df_dropped.to_csv("dynamic_dropped_samples_log.csv", index=False)
        print(f"[Saved] Dynamic drop records saved to 'dynamic_dropped_samples_log.csv'.")

    report = classification_report(
        all_true, 
        all_pred, 
        labels=[0, 1, 2, 3], 
        target_names=list(INVERSE_CLASS_MAP.values()),
        zero_division=0        
    )
    print(report)

if __name__ == "__main__":
    run_dynamic_pipeline()