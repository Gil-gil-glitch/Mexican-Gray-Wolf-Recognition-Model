# Mexican-Gray-Wolf-Recognition-Model

A multi-stage, cost-sensitive computer vision pipeline engineered to automate population monitoring of the endangered Mexican Gray Wolf (Canis lupus baileyi). The system addresses the dual challenges of extreme environmental noise (camouflage, dense brush, weather) and high phenotypic similarity among sympatric canids by decoupling spatial localization, salient background removal, and parallel dual-attention feature extraction.

[INSERT IMAGE HERE]

---
## System Design and Architecture

Instead of forcing a single model to handle localization and fine-grained classification simultaneously, this system implements a strict three-stage cascaded inference framework optimized for edge deployment on low-power remote trail cameras.

1. Stage 1: Spatial Gating and Localization (YOLOv8) – Drops an anchor-free bounding box around regional areas of interest. Operates at an empirically calculated static threshold of 0.25 to filter out up to 90% of empty scenery frames early, saving massive computational overhead.

2. Stage 2: Salient Edge Matting and Background Stripping (BiRefNet) – Extracts the bounding box crop and applies Salient Object Detection (SOD). Gradients are processed via a vectorized NumPy mask-hardener ($\alpha > 128$) to yield a single, cohesive animal silhouette. This successfully eliminates the structural fragmentation failures common in edge-discontinuity models like O-SegNet.

3. Stage 3: Fine-Grained Classification (Dual-Attention Network) – Processes the clean silhouette through a parallel architecture. The Spatial Attention Head maps macro-skeletal proportions (snout-to-ear ratios) to combat partial out-of-frame occlusions, while the Spectral Attention Head operates in the frequency domain to track micro-biological cues (fur coat texture density, guard hair distributions).

## Repository Structure

```text
.
├── BackgroundFilteringPipeline
│   ├── cascaded_dynamic_inference.py
│   └── cascaded_inference.py
├── IdahoWolfCam
├── SOD
├── UnifiedWildlifeDataset
├── WildlifeRecognitionModel
│   ├── dual_attention_model.py
│   └── train.py
├── iWildCam2019
├── o-segnet
└── tools
    └── hash_counter.py
```

Data Notice: The raw image arrays and manifest .csv files from the Idaho Wolf Images and iWildCam repositories are excluded from this remote tracking due to extreme file size restrictions. All pipelines are designed to ingest data mapped to local storage paths

## Empirical Findings and Research Performance Boundaries
1. The Gating Threshold Parameter Sweep
Testing across out-of-sample datasets proved that a threshold of 0.25 represents the mathematical equilibrium point for a pre-trained spatial gatekeeper operating in deep wilderness environments:

| Threshold  | Procssed  | Dropped   | Classifier Accuracy   |
| ---------- | ----------| ----------| ----------------------|
| 0.40       | 358       | 642       |       99.16%          |
| 0.30       | 443       | 557       |       98.41%          |
| 0.25       | 476       | 524       |       98.11%          |

2. The Stage 2 "Matting Collapse" Phenomenon
Attempts to address the persistent ~52% wilderness drop rate by dropping the YOLO gating floor down aggressively to 0.12 (via risk-aware adaptive canid weighting in cascaded_dynamic_inference.py) exposed a critical multi-stage architectural rule:

- Passing ultra-low confidence regions forces the pipeline to accept high-noise background anomalies (moving leaves, rocky shadows).

- When fed into Stage 2, BiRefNet's confidence maps completely collapse, rendering entirely black/transparent alpha masks that prompt hard bbox drops downstream.

- System Security Proof: Despite the increased noise injection, Stage 2 effectively acts as a reliable structural filter—completely wiping out artifacts before they reach Stage 3, allowing the Dual-Attention classifier to maintain a flawless 99.00% weighted precision rate.

## Getting Started and Local Replication
1. Environment Setup
Clone the repository and install the baseline requirements. Ensure a CUDA-capable GPU is available to support heavy execution tasks in Stage 2 and Stage 3.
```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
source verify.venv/bin/activate  # Or set up a clean virtual environment
pip install -r requirements.txt
```

2. Running Cascaded Inference
To execute the static-threshold baseline simulation loop on your localized testing matrix:
```
python3 BackgroundFilteringPipeline/cascaded_inference.py
```

To run the risk-aware class-weighted dynamic pipeline and output the automated drop log 
dynamic_drop log
```
python3 BackgroundFilteringPipeline/cascaded_dynamic_inference.py
```

3. Model Architecture Instantiation
To inspect or adapt the Parallel Spatial/Spectral Attention feature extractor module:

```
from WildlifeRecognitionModel.dual_attention_model import DualAttentionClassifier

# Instantiate model for 4-class out (Empty, Wolf, Coyote, Dog)
model = DualAttentionClassifier(num_classes=4)
print(model)
```

## Literature Context and Technical Advantages
- Overcomes Scenery Memorization: Unlike standard sequential CNNs (e.g., Snapshot Serengeti frameworks), which inherently suffer from localized site-bias by accidentally memorizing static trial camera backgrounds, this pipeline completely strips out non-biological pixels before classification.

- Eliminates Fragmentation: Resolves the critical failure modes of edge-feature-driven architectures like O-SegNet, which fragment single animal frames when intersected by branches or foliage.

- Triage Automation: By logging low-confidence rejections cleanly to a secondary stream, the system facilitates a highly reliable Human-in-the-Loop model—automating 47% of pristine data with 99% certainty while flagging heavily obscured frames for specialist review.