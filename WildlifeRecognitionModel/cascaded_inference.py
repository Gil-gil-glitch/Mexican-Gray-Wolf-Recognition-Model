## cascaded_inference.py
#
# Optimized cascaded multi-stage inference pipeline with Soft-Gating Fallback.
# Prevents false-negative drops on night/camouflaged frames by feeding center crops
# directly to Stage 3 when Stage 1 localization falls below confidence bounds.
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
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F

# Import custom network from Step 2
from dual_attention_model import DualAttentionClassifier

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

def initialize_cascaded_pipeline():
    print("=====================================================================")
    print("      INITIALIZING OPTIMIZED CASCADED INFERENCE SYSTEM (SOFT-GATE)   ")
    print("=====================================================================")
    
    print("[Stage 1] Loading YOLOv8 localization gatekeeper...")
    yolo_model = YOLO("yolov8n.pt")
    
    print("[Stage 2] Loading BiRefNet matting background removal session...")
    provider_options = [{"device_id": "0"}] if torch.cuda.is_available() else []
    providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    sod_session = new_session("birefnet-general", providers=providers, provider_options=provider_options)
    
    print("[Stage 3] Loading custom Dual-Attention Fine-Grain Classifier...")
    classifier = DualAttentionClassifier(num_classes=4)
    classifier_weights_path = Path("best_dual_attention_model.pth")
    
    if not classifier_weights_path.exists():
        print(f"[CRITICAL] Classifier weights not found at {classifier_weights_path}!")
        return None, None, None
        
    classifier.load_state_dict(torch.load(classifier_weights_path, map_location=DEVICE))
    classifier = classifier.to(DEVICE)
    classifier.eval()
    
    print("[Status] All cascaded stages loaded successfully onto target device.\n")
    return yolo_model, sod_session, classifier

def run_pipeline_simulation():
    yolo_model, sod_session, classifier = initialize_cascaded_pipeline()
    if yolo_model is None:
        return

    df = pd.read_csv(MANIFEST_PATH)
    
    TEST_SIZE = 10000
    print(f"[Sampling] Selecting {TEST_SIZE} random canid validation samples from master database...")
    test_candidates = df[df['category_id'].isin([11, 15, 18])].sample(n=TEST_SIZE, random_state=101)
    
    # Metrics tracking
    all_true = []
    all_pred = []

    # Architectural Optimization Telemetry Counters
    total_processed = 0
    standard_pipeline_successes = 0
    soft_gate_fallback_activations = 0
    catastrophic_drops = 0

    dropped_files = []

    print("=====================================================================")
    print("             LAUNCHING SOFT-GATED PIPELINE SIMULATION                ")
    print("=====================================================================")
    
    COCO_ANIMAL_CLASSES = [15, 16, 17, 18, 19, 20, 21, 22, 23]
    GATING_THRESHOLD = 0.25 
    
    for idx, row in tqdm(test_candidates.iterrows(), total=TEST_SIZE, desc="Processing Pipeline"):
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
                use_fallback = False
                
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
                
                # Soft Gating Evaluation Point
                if best_box is None or highest_conf < GATING_THRESHOLD:
                    use_fallback = True
                
                if not use_fallback:
                    # --- OPTION A: STANDARD CASCADED ROUTE (YOLO BBox -> BiRefNet -> Stage 3) ---
                    try:
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
                            use_fallback = True  # Fall back if matting erases the frame contents completely
                        else:
                            final_silhouette = crisp_rgba.crop(final_bbox).convert('RGB')
                            standard_pipeline_successes += 1
                    except Exception:
                        use_fallback = True

                if use_fallback:
                    # --- OPTION B: ADVANCED MULTI-SCALE RESOLUTION ENSEMBLE FALLBACK ---
                    try:
                        # First stream: Extract a generous 85% area crop to capture background context and off-center animals
                        crop_fraction_a = 0.85
                        left_a = int((1 - crop_fraction_a) * w / 2)
                        top_a = int((1 - crop_fraction_a) * h / 2)
                        right_a = int((1 + crop_fraction_a) * w / 2)
                        bottom_a = int((1 + crop_fraction_a) * h / 2)
                        generous_crop = img.crop((left_a, top_a, right_a, bottom_a))
                        
                        # Second stream: Run single-pass BiRefNet background stripping on the broad region
                        rgba_output = remove(generous_crop, session=sod_session)
                        r, g, b, a = rgba_output.split()
                        
                        alpha_array = np.array(a)
                        hard_alpha_array = np.where(alpha_array > 128, 255, 0).astype(np.uint8)
                        hard_alpha_channel = Image.fromarray(hard_alpha_array)
                        crisp_rgba = Image.merge("RGBA", (r, g, b, hard_alpha_channel))
                        
                        final_bbox = crisp_rgba.getbbox()
                        
                        if final_bbox:
                            # Stream A: Full Silhouette bounding box (Captures macro skeletal proportions)
                            silhouette_a = crisp_rgba.crop(final_bbox).convert('RGB')
                            
                            full_black_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 255))
                            
                            # Paste the isolated silhouette back into its precise original pixel coordinates
                            full_black_canvas.paste(crisp_rgba, (left, top), hard_alpha_channel)
                            silhouette_b = full_black_canvas.convert('RGB')
                        else:
                            # Emergency duplicate fallback if matting turns up empty
                            silhouette_a = generous_crop.convert('RGB')
                            silhouette_b = generous_crop.convert('RGB')
                        
                        # 3. Transform both streams into model tensors
                        tensor_a = inference_transforms(silhouette_a).unsqueeze(0).to(DEVICE)
                        tensor_b = inference_transforms(silhouette_b).unsqueeze(0).to(DEVICE)
                        
                        # 4. Evaluate both streams sequentially through Stage 3 Classifier
                        with torch.no_grad():
                            logits_a = classifier(tensor_a)
                            probs_a = F.softmax(logits_a, dim=1)
                            
                            logits_b = classifier(tensor_b)
                            probs_b = F.softmax(logits_b, dim=1)
                            
                            # Blend predictions. We weight Stream A slightly higher (0.60) for anatomy,
                            # and Stream B (0.40) to provide the structural scale verification.
                            gamma = 0.60
                            final_probs = (gamma * probs_a) + ((1.0 - gamma) * probs_b)
                            _, predicted_idx = torch.max(final_probs, 1)
                            
                        soft_gate_fallback_activations += 1
                        
                    except Exception:
                        emergency_silhouette = img.resize((224, 224))
                        tensor_emergency = inference_transforms(emergency_silhouette).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            logits = classifier(tensor_emergency)
                            _, predicted_idx = torch.max(F.softmax(logits, dim=1), 1)
                        soft_gate_fallback_activations += 1
                    
                    # Update telemetry pipelines
                    total_processed += 1
                    all_true.append(true_mapped)
                    all_pred.append(predicted_idx.item())
                    continue
                        
                # --- STAGE 3: FINE-GRAIN DUAL ATTENTION EVALUATION ---
                input_tensor = inference_transforms(final_silhouette).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = classifier(input_tensor)
                    probabilities = F.softmax(logits, dim=1)
                    _, predicted_idx = torch.max(probabilities, 1)
                
                total_processed += 1
                all_true.append(true_mapped)
                all_pred.append(predicted_idx.item())

        except Exception as e:
            catastrophic_drops += 1
            dropped_files.append({
                'dropped_file_names': file_name,
                'true_label': true_label,
                'true_id': true_id
            })
            continue

    # Final Summary Reports
    print("\n" + "="*60)
    print("      SOFT-GATED PIPELINE OPTIMIZATION ANALYSIS SUMMARY")
    print("=========================================================")
    print(f"Total Source Images Processed:       {total_processed}")
    print(f"  ↳ Standard Matting Route:          {standard_pipeline_successes}")
    print(f"  ↳ Soft-Gate Fallback Crops:         {soft_gate_fallback_activations} (Saved from Dropping!)")
    print(f"  ↳ Uncaught Structural Exceptions:    {catastrophic_drops}")
    print("="*60)

    if dropped_files:
        df_dropped = pd.DataFrame(dropped_files)
        df_dropped.to_csv("dropped_samples_log.csv", index=False)

    print("\nSTANDARD CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_true, all_pred, labels=[1, 2, 3], 
                                target_names=list(INVERSE_CLASS_MAP.values())[1:], zero_division=0))

    cm = confusion_matrix(all_true, all_pred, labels=[1, 2, 3])
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(INVERSE_CLASS_MAP.values())[1:])
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Optimized Soft-Gated Pipeline Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("[Success] Pipeline update completed. Metrics visual exported to 'confusion_matrix.png'")

if __name__ == "__main__":
    run_pipeline_simulation()