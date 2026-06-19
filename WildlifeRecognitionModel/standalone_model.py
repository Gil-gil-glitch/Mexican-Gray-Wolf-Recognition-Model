## standalone_classifier_evaluation.py
#
# This script evaluates the Stage 3 DualAttentionClassifier standalone directly 
# on raw, unsegmented images. It instantiates the required 4 classes to match 
# model weights, but excludes the empty class from final performance metrics.
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

# Keep the full dictionary mapping to match training weights
INVERSE_CLASS_MAP = {0: "Empty Landscape", 1: "Mexican Gray Wolf", 2: "Coyote", 3: "Domestic Dog"}
FULL_CLASS_NAMES = list(INVERSE_CLASS_MAP.values())

# Define the evaluation target labels (excluding Index 0)
ANIMAL_LABELS = [1, 2, 3]
ANIMAL_CLASS_NAMES = ["Mexican Gray Wolf", "Coyote", "Domestic Dog"]

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
    # FIX: Must be 4 to match your physical "best_dual_attention_model.pth" matrix shape
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
    
    # 1. Isolation Filter: Pull target canid records exclusively (excluding category 0)
    TEST_SIZE = 10000
    print(f"[Sample Isolation] Drawing standard {TEST_SIZE} sample target matrix...")
    test_candidates = df[df['category_id'].isin([11, 15, 18])].sample(n=TEST_SIZE, random_state=101)
    
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

        # 2. Map raw IDs to your exact 4-class contiguous training scheme
        mapping = {0: 0, 15: 1, 11: 2, 18: 3}
        true_mapped = mapping.get(true_id)

        if not img_path.exists() or true_mapped is None:
            continue
        
        try:
            with Image.open(img_path).convert('RGB') as raw_img:
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
    print("STANDALONE PERFORMANCE CANID EVALUATION RESULTS")
    print("="*60)
    print(f"Evaluation Complete. Total Unsegmented Images Processed: {processed_samples}\n")

    # FIX: Compute the matrix and classification reports ONLY for labels [1, 2, 3]
    cm = confusion_matrix(all_true, all_pred, labels=ANIMAL_LABELS)
    
    print("--- Detailed Confusion Matrix ---")
    header = f"Actual \\ Pred" + " " * 7 + "".join([f"{name:<22}" for name in ANIMAL_CLASS_NAMES])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{ANIMAL_CLASS_NAMES[i]:<20}" + "".join([f"{val:<22}" for val in row])
        print(row_str)
        
    print("\n" + "-"*60)
    print("--- Per-Class False Positive Rate (FPR) Analysis ---")
    print("-"*60)
    
    for idx, class_name in enumerate(ANIMAL_CLASS_NAMES):
        # Index matches our sliced confusion matrix positions
        tp = cm[idx, idx]
        fn = np.sum(cm[idx, :]) - tp
        fp = np.sum(cm[:, idx]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
        sensitivity = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
        
        print(f"Class [{class_name}]:")
        print(f"  - Standalone Accuracy/Recall: {sensitivity:.2f}%")
        print(f"  - False Positive Rate (FPR):   {fpr:.2f}%")

    print("\n" + "="*60)
    print("STANDARD CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(
        all_true, 
        all_pred, 
        labels=ANIMAL_LABELS, 
        target_names=ANIMAL_CLASS_NAMES,
        zero_division=0        
    )
    print(report)

if __name__ == "__main__":
    run_standalone_evaluation()