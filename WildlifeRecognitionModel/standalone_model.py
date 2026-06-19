## standalone_classifier_evaluation.py
#
# This script evaluates the Stage 3 DualAttentionClassifier standalone directly 
# on raw, unsegmented images. It bypasses YOLOv8 and BiRefNet to assess 
# baseline scene-bias, cross-species confusion matrices, and False Positive Rates (FPR).
#

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Import your custom network from Step 2
from dual_attention_model import DualAttentionClassifier

# Statistics for final metric tracking
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F

# Resources
MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_final_train_manifest.csv")
RAW_IWILDCAM_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/train_images")
RAW_IDAHO_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/wolf_images")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output mapping matching your network's 4 output logits
INVERSE_CLASS_MAP = {0: "Empty Landscape", 1: "Mexican Gray Wolf", 2: "Coyote", 3: "Domestic Dog"}
CLASS_NAMES = list(INVERSE_CLASS_MAP.values())

# Custom Image Transform matching your exact training validation configuration
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def initialize_standalone_classifier():
    print("=====================================================================")
    print("       INITIALIZING STANDALONE CLASSIFIER EVALUATION SUITE           ")
    print("=====================================================================")
    
    print("[Target Initialization] Loading custom Dual-Attention Fine-Grain Classifier...")
    classifier = DualAttentionClassifier(num_classes=4)
    classifier_weights_path = Path("best_dual_attention_model.pth")
    
    if not classifier_weights_path.exists():
        print(f"[CRITICAL] Classifier weights not found at {classifier_weights_path}!")
        return None
        
    classifier.load_state_dict(torch.load(classifier_weights_path, map_location=DEVICE))
    classifier = classifier.to(DEVICE)
    classifier.eval()
    
    print("[Status] Standalone model network weights loaded successfully.\n")
    return classifier

def run_standalone_evaluation():
    classifier = initialize_standalone_classifier()
    if classifier is None:
        return

    df = pd.read_csv(MANIFEST_PATH)
    
    # Target the exact same 10,000-sample slice for alignment
    TEST_SIZE = 10000
    print(f"[Sample Isolation] Drawing standard {TEST_SIZE} sample target matrix...")
    test_candidates = df[df['category_id'].isin([0, 11, 15, 18])].sample(n=TEST_SIZE, random_state=101)
    
    all_true = []
    all_pred = []
    processed_samples = 0

    print("=====================================================================")
    print("            LAUNCHING STANDALONE DIRECT INFERENCE LOOP                ")
    print("=====================================================================")
    
    for idx, row in tqdm(test_candidates.iterrows(), total=TEST_SIZE, desc="Evaluating Raw Imagery"):
        file_name = row['file_name']
        source = row['dataset_source']
        true_id = int(row['category_id'])
        
        base_dir = RAW_IWILDCAM_DIR if source == 'iwildcam' else RAW_IDAHO_DIR
        img_path = base_dir / file_name

        # Enforce structural mapping dictionary 
        mapping = {15: 1, 11: 2, 18: 3, 0: 0}
        true_mapped = mapping.get(true_id, 0)

        if not img_path.exists():
            continue
        
        try:
            # --- BYPASS STAGES 1 & 2: Process Raw Image Directly ---
            with Image.open(img_path).convert('RGB') as raw_img:
                
                # Transform original uncropped, unsegmented picture
                input_tensor = inference_transforms(raw_img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = classifier(input_tensor)
                    probabilities = F.softmax(logits, dim=1)
                    _, predicted_idx = torch.max(probabilities, 1)
                
                processed_samples += 1
                all_true.append(true_mapped)
                all_pred.append(predicted_idx.item())

        except Exception as e:
            continue

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # =====================================================================
    # POST-PROCESSING EXPLICIT METRIC MAPS
    # =====================================================================
    print("\n" + "="*60)
    print("STANDALONE PERFORMANCE SCENERY EVALUATION RESULTS")
    print("="*60)
    print(f"Evaluation Complete. Total Unsegmented Images Processed: {processed_samples}\n")

    # 1. Generate Raw Confusion Matrix
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2, 3])
    
    print("--- Detailed Confusion Matrix ---")
    header = f"Actual \\ Pred" + " " * 7 + "".join([f"{name:<22}" for name in CLASS_NAMES])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{CLASS_NAMES[i]:<20}" + "".join([f"{val:<22}" for val in row])
        print(row_str)
        
    print("\n" + "-"*60)
    print("--- Per-Class False Positive Rate (FPR) Analysis ---")
    print("-"*60)
    
    # 2. Extract Per-Class Parameters to derive False Positive Rates
    for idx, class_name in enumerate(CLASS_NAMES):
        tp = cm[idx, idx]
        fn = np.sum(cm[idx, :]) - tp
        fp = np.sum(cm[:, idx]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        # FPR Calculation formula: FP / (FP + TN)
        fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
        sensitivity = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        
        print(f"Class [{class_name}]:")
        print(f"  - Standalone Accuracy/Recall: {sensitivity:.2f}%")
        print(f"  - False Positive Rate (FPR):   {fpr:.2f}% (How often other classes/scenes tripped this label)")

    print("\n" + "="*60)
    print("STANDARD CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(
        all_true, 
        all_pred, 
        labels=[0, 1, 2, 3], 
        target_names=CLASS_NAMES,
        zero_division=0        
    )
    print(report)

if __name__ == "__main__":
    run_standalone_evaluation()