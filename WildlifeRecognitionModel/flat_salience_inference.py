#
## flat_salience_inference.py
#
#  The purpose of this program is to test the flat saliency pipeline for wildlife detection and classification without the use of YOLO. It performs the following stages:
#  1. Load the BiRefNet matting background removal session
#  2. Load the custom Dual-Attention Fine-Grain Classifier
#  3. Process each image through the flat saliency pipeline
#  4. Evaluate the classification performance
#

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from pathlib import Path
from PIL import Image
from rembg import remove, new_session
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F

# Import custom network
from dual_attention_model import DualAttentionClassifier

# Resources
MANIFEST_PATH = Path("/home/greatgilbertsoco/WolfDetect/data/pseudo_final_train_manifest.csv")
RAW_IWILDCAM_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/train_images")
RAW_IDAHO_DIR = Path("/home/greatgilbertsoco/WolfDetect/data/wolf_images")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INVERSE_CLASS_MAP = {0: "Empty Landscape", 1: "Mexican Gray Wolf", 2: "Coyote", 3: "Domestic Dog"}

inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def run_flat_saliency_simulation():
    print("=====================================================================")
    print("     LAUNCHING EXPERIMENT: FLAT DENSE SALIENCY PIPELINE (NO YOLO)    ")
    print("=====================================================================")
    
    print("[Stage 1] BYPASSED.")
    print("[Stage 2] Loading BiRefNet matting background removal session...")
    provider_options = [{"device_id": "0"}] if torch.cuda.is_available() else []
    providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    sod_session = new_session("birefnet-general", providers=providers, provider_options=provider_options)
    
    print("[Stage 3] Loading custom Dual-Attention Fine-Grain Classifier...")
    classifier = DualAttentionClassifier(num_classes=4)
    classifier.load_state_dict(torch.load("best_dual_attention_model.pth", map_location=DEVICE))
    classifier = classifier.to(DEVICE).eval()

    df = pd.read_csv(MANIFEST_PATH)
    TEST_SIZE = 10000
    test_candidates = df[df['category_id'].isin([11, 15, 18])].sample(n=TEST_SIZE, random_state=101)
    
    all_true, all_pred = [], []
    catastrophic_drops = 0

    # Start Wall-Clock Performance Timer
    start_time = time.perf_counter()

    for idx, row in tqdm(test_candidates.iterrows(), total=TEST_SIZE, desc="Processing Flat Saliency"):
        file_name = row['file_name']
        source = row['dataset_source']
        true_id = int(row['category_id'])
        
        base_dir = RAW_IWILDCAM_DIR if source == 'iwildcam' else RAW_IDAHO_DIR
        img_path = base_dir / file_name

        mapping = {15: 1, 11: 2, 18: 3, 0: 0}
        true_mapped = mapping.get(true_id, 0)

        if not img_path.exists():
            continue
        
        try:
            with Image.open(img_path).convert('RGB') as img:
                w, h = img.size
                
                # Direct Saliency Routing: Bypassing YOLO completely
                crop_fraction = 0.85
                left = int((1 - crop_fraction) * w / 2)
                top = int((1 - crop_fraction) * h / 2)
                right = int((1 + crop_fraction) * w / 2)
                bottom = int((1 + crop_fraction) * h / 2)
                generous_crop = img.crop((left, top, right, bottom))
                
                # Force heavy transformer background stripping on every single sample
                rgba_output = remove(generous_crop, session=sod_session)
                r, g, b, a = rgba_output.split()
                
                alpha_array = np.array(a)
                hard_alpha_array = np.where(alpha_array > 128, 255, 0).astype(np.uint8)
                hard_alpha_channel = Image.fromarray(hard_alpha_array)
                
                crisp_rgba = Image.merge("RGBA", (r, g, b, hard_alpha_channel))
                final_bbox = crisp_rgba.getbbox()
                
                if final_bbox:
                    final_silhouette = crisp_rgba.crop(final_bbox).convert('RGB')
                else:
                    final_silhouette = generous_crop.convert('RGB')
                        
                # Forward Pass Through Stage 3 Fine-Grain Classifier
                input_tensor = inference_transforms(final_silhouette).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = classifier(input_tensor)
                    probabilities = F.softmax(logits, dim=1)
                    _, predicted_idx = torch.max(probabilities, 1)
                
                all_true.append(true_mapped)
                all_pred.append(predicted_idx.item())

        except Exception:
            catastrophic_drops += 1
            continue

    end_time = time.perf_counter()
    total_execution_time = end_time - start_time

    print("\n" + "="*60)
    print("      FLAT SALIENCY PIPELINE BENCHMARK SUMMARY")
    print("=========================================================")
    print(f"Total Images Processed:             {len(all_true)}")
    print(f"Total Benchmark Compute Time:       {total_execution_time:.2f} seconds")
    print(f"Average Latency Per Frame:          {(total_execution_time / len(all_true)) * 1000:.2f} ms")
    print("="*60)

    print("\nSTANDARD CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_true, all_pred, labels=[1, 2, 3], 
                                target_names=list(INVERSE_CLASS_MAP.values())[1:], zero_division=0))

if __name__ == "__main__":
    run_flat_saliency_simulation()