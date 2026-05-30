#
## cascaded_inference.py
#
#  This script implements the full cascaded multi-stage inference pipeline for our wildlife classification system. 
#  It integrates the YOLOv8 localization gatekeeper, BiRefNet matting for background removal, and the custom 
# Dual-Attention Fine-Grain Classifier into a seamless end-to-end process. The script simulates the inference 
# flow on a sample of test images from the master manifest, demonstrating how each stage contributes to the 
# final classification output while providing detailed logging for each step of the pipeline.
#
#
#
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

# Import  custom network from Step 2
from dual_attention_model import DualAttentionClassifier

# Stastics for final output mapping
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import torch.nn.functional as F

# Respources
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

def initialize_cascaded_pipeline():
    print("=====================================================================")
    print("         INITIALIZING CASCADED MULTI-STAGE INFERENCE SYSTEM          ")
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
        print(f"[CRITICAL] Classifier weights not found at {classifier_weights_path}! Run step 2 first.")
        return None, None, None
        
    classifier.load_state_dict(torch.load(classifier_weights_path, map_location=DEVICE))
    classifier = classifier.to(DEVICE)
    classifier.eval()
    
    print("[Status] All cascaded stages loaded successfully onto target accelerator device.\n")
    return yolo_model, sod_session, classifier

def run_pipeline_simulation():
    yolo_model, sod_session, classifier = initialize_cascaded_pipeline()
    if yolo_model is None:
        return

    df = pd.read_csv(MANIFEST_PATH)
    
    # Filter for rows that contain animals we want to test
    # (Excluding empty rows 0, prioritizing target canids 15, 11, 18)
    test_candidates = df[df['category_id'].isin([11, 15, 18])].sample(n=30, random_state=101)
    
    # Metrics tracking
    all_true = []
    all_pred = []

    print("=====================================================================")
    print("                 LAUNCHING CASCADED PIPELINE SIMULATION              ")
    print("=====================================================================")
    
    COCO_ANIMAL_CLASSES = [15, 16, 17, 18, 19, 20, 21, 22, 23]
    
    for idx, row in test_candidates.iterrows():
        file_name = row['file_name']
        source = row['dataset_source']
        true_id = int(row['category_id'])
        
        # Resolve true label string
        true_label = "Mexican Gray Wolf" if true_id == 15 else ("Coyote" if true_id == 11 else "Domestic Dog")
        
        base_dir = RAW_IWILDCAM_DIR if source == 'iwildcam' else RAW_IDAHO_DIR
        img_path = base_dir / file_name

        # Mapping for final output display
        mapping = {15: 1, 11: 2, 18: 3, 0: 0}
        true_mapped = mapping.get(true_id, 0) # Default to 0 (Empty) if not found

        
        if not img_path.exists():
            continue
            
        print(f"\n[Processing Raw Input Image]: {file_name}")
        print(f" -> True Ground Truth Class: {true_label}")
        
        try:
            with Image.open(img_path).convert('RGB') as img:
                w, h = img.size
                
                # --- STAGE 1: NATIVE SPATIAL GATING ---
                yolo_results = yolo_model(img, verbose=False)
                best_box = None
                highest_conf = -1.0
                
                for result in yolo_results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = box.conf[0].item()
                            
                            if cls_id in COCO_ANIMAL_CLASSES and conf > highest_conf:
                                highest_conf = conf
                                best_box = box.xyxy[0].cpu().numpy()
                
                # Dynamic Gating Decision Check
                if best_box is None or highest_conf < 0.40:
                    print(f" -> [Stage 1 GATING]: No animal localized above 0.40 threshold (Max Conf: {highest_conf:.2f})")
                    print(f" >> FINAL PIPELINE PREDICTION: {INVERSE_CLASS_MAP[0]}")
                    continue
                    
                print(f" -> [Stage 1 PASSED]: Localized biological shape. Confidence: {highest_conf:.2f}")
                
                # --- STAGE 2: BACKGROUND STRIPPING AND MASK HARDENING ---
                xmin, ymin, xmax, ymax = best_box
                pad_w, pad_h = (xmax - xmin) * 0.10, (ymax - ymin) * 0.10
                crop_box = (max(0, int(xmin - pad_w)), max(0, int(ymin - pad_h)), min(w, int(xmax + pad_w)), min(h, int(ymax + pad_h)))
                coarse_crop = img.crop(crop_box)
                
                # Apply BiRefNet matting via session
                rgba_output = remove(coarse_crop, session=sod_session)
                r, g, b, a = rgba_output.split()
                
                # Vectorized edge mask hardening
                alpha_array = np.array(a)
                hard_alpha_array = np.where(alpha_array > 128, 255, 0).astype(np.uint8)
                hard_alpha_channel = Image.fromarray(hard_alpha_array)
                
                crisp_rgba = Image.merge("RGBA", (r, g, b, hard_alpha_channel))
                final_bbox = crisp_rgba.getbbox()
                
                if not final_bbox:
                    print(" -> [Stage 2 MATTING]: Matting operations collapsed mask completely.")
                    print(f" >> FINAL PIPELINE PREDICTION: {INVERSE_CLASS_MAP[0]}")
                    continue
                    
                # Extract clean background removed tensor crop
                final_silhouette = crisp_rgba.crop(final_bbox).convert('RGB')
                print(" -> [Stage 2 PASSED]: Background extracted. Edge masks hardened to eliminate blur.")
                
                # --- STAGE 3: FINE-GRAIN DUAL ATTENTION EVALUATION ---
                input_tensor = inference_transforms(final_silhouette).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = classifier(input_tensor)
                    probabilities = F.softmax(logits, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    
                final_pred_string = INVERSE_CLASS_MAP[predicted_idx.item()]
                pred_conf_val = confidence.item() * 100
                
                print(f" -> [Stage 3 PASSED]: Spatial & Spectral texture attention vectors resolved.")
                print(f" >> FINAL PIPELINE PREDICTION: {final_pred_string} ({pred_conf_val:.2f}% Confidence)")
                
                # Track metrics
                final_pred_idx = predicted_idx.item()
                all_true.append(true_mapped)
                all_pred.append(final_pred_idx)


                # Print success tag
                if final_pred_string == true_label:
                    print(" [MATCH STATUS]: CORRECT!")
                else:
                    print(" [MATCH STATUS]: MISCLASSIFIED!")
                    print("\n" + "="*50)
                
                    
        except Exception as e:
            print(f" [CRITICAL PIPELINE ERROR]: Could not process sample image due to: {e}")
            continue

     # Final Metrics Display after all samples processed
    print("\n" + "="*50)
    print("FINAL PERFORMANCE METRICS")
    print("="*50)

    report = classification_report(
            all_true, 
            all_pred, 
            labels=[0, 1, 2, 3], 
            target_names=list(INVERSE_CLASS_MAP.values()),
            zero_division=0        
    )
    print(report)
if __name__ == "__main__":
    import torch.nn.functional as F
    run_pipeline_simulation()