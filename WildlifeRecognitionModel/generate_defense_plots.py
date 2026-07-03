import numpy as np
import matplotlib.pyplot as plt

# Defining Target Categories: Wolves, Coyotes, and Domestic Dogs
classes = ['Mexican Gray Wolf', 'Coyote', 'Domestic Dog']
x = np.arange(len(classes))  # Label locations
width = 0.15                 # Reduced width to comfortably fit 5 bars side-by-side

# --- COMPLETE EXPERIMENTAL METRICS MATRIX ---

# 1. Standalone Classifier Baseline (No Pipeline Preprocessing / Stages 1 & 2 completely omitted)
standalone_p = [0.94, 0.94, 0.94]
standalone_r = [0.62, 0.75, 0.73]
standalone_f1 = [0.75, 0.83, 0.82]

# 2. Baseline Pipeline (YOLO Gatekeeper only - Hard Dropping configuration)
baseline_p = [0.99, 0.99, 0.98]
baseline_r = [0.40, 0.48, 0.59]
baseline_f1 = [0.57, 0.65, 0.74]

# 3. Optimization 1 (Soft-Gating with Raw Multi-Region Anchor Window Slicing)
opt1_p = [0.93, 0.94, 0.92]
opt1_r = [0.40, 0.40, 0.40]
opt1_f1 = [0.56, 0.56, 0.55]

# 4. Optimization 2 (Final Locked Main: Soft-Gating + Dense BiRefNet Saliency Fallback)
opt2_p = [0.93, 0.94, 0.92]
opt2_r = [0.63, 0.75, 0.73]
opt2_f1 = [0.74, 0.84, 0.81]

# 5. Flat Saliency Pipeline (Stage 1 YOLO Bypassed - All frames direct to 85% Saliency Crop)
flat_sod_p = [0.92, 0.91, 0.87]
flat_sod_r = [0.58, 0.75, 0.69]
flat_sod_f1 = [0.71, 0.82, 0.77]

# Set up the plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(1, 3, figsize=(22, 6), sharey=True)

# Colors tailored for high distinction on presentation slides
colors = ['#95a5a6', '#e74c3c', '#f39c12', '#2ecc71', '#3498db']
labels = [
    'Ctrl A: Standalone Classifier (No Pipeline)',
    'Phase 1: Baseline (YOLO Hard Drop)',
    'Phase 2: Opt 1 (Raw Window Slicing)',
    'Phase 3: Opt 2 (Dense SOD Fallback - Main)',
    'Ctrl B: Flat Saliency (No YOLO Gate)'
]

# ---------------------------------------------------------
# GRAPH 1: PRECISION COMPARISON
# ---------------------------------------------------------
axs[0].bar(x - 2*width, standalone_p, width, label=labels[0], color=colors[0], edgecolor='black')
axs[0].bar(x - width, baseline_p, width, label=labels[1], color=colors[1], edgecolor='black')
axs[0].bar(x, opt1_p, width, label=labels[2], color=colors[2], edgecolor='black')
axs[0].bar(x + width, opt2_p, width, label=labels[3], color=colors[3], edgecolor='black')
axs[0].bar(x + 2*width, flat_sod_p, width, label=labels[4], color=colors[4], edgecolor='black')

axs[0].set_title('Precision Evolution Across Layouts', fontsize=13, fontweight='bold')
axs[0].set_xticks(x)
axs[0].set_xticklabels(classes, rotation=10, fontsize=11)
axs[0].set_ylabel('Score (0.0 - 1.0)', fontsize=12)
axs[0].set_ylim(0, 1.15)

# ---------------------------------------------------------
# GRAPH 2: RECALL COMPARISON (The Core Spatial-Gating Validation)
# ---------------------------------------------------------
axs[1].bar(x - 2*width, standalone_r, width, label=labels[0], color=colors[0], edgecolor='black')
axs[1].bar(x - width, baseline_r, width, label=labels[1], color=colors[1], edgecolor='black')
axs[1].bar(x, opt1_r, width, label=labels[2], color=colors[2], edgecolor='black')
axs[1].bar(x + width, opt2_r, width, label=labels[3], color=colors[3], edgecolor='black')
axs[1].bar(x + 2*width, flat_sod_r, width, label=labels[4], color=colors[4], edgecolor='black')

axs[1].set_title('Recall Comparison & Fallback Impact', fontsize=13, fontweight='bold')
axs[1].set_xticks(x)
axs[1].set_xticklabels(classes, rotation=10, fontsize=11)

# ---------------------------------------------------------
# GRAPH 3: F1-SCORE COMPARISON (Overall Balanced Architecture)
# ---------------------------------------------------------
axs[2].bar(x - 2*width, standalone_f1, width, label=labels[0], color=colors[0], edgecolor='black')
axs[2].bar(x - width, baseline_f1, width, label=labels[1], color=colors[1], edgecolor='black')
axs[2].bar(x, opt1_f1, width, label=labels[2], color=colors[2], edgecolor='black')
axs[2].bar(x + width, opt2_f1, width, label=labels[3], color=colors[3], edgecolor='black')
axs[2].bar(x + 2*width, flat_sod_f1, width, label=labels[4], color=colors[4], edgecolor='black')

axs[2].set_title('F1-Score Metrics Synthesis', fontsize=13, fontweight='bold')
axs[2].set_xticks(x)
axs[2].set_xticklabels(classes, rotation=10, fontsize=11)

# Centralized Legend Placement beneath the subplots
axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=11, frameon=True)

plt.tight_layout()
output_chart_path = "pipeline_optimization_metrics.png"
plt.savefig(output_chart_path, dpi=300, bbox_inches='tight')
print(f"[Success] Multi-stage optimization comparison chart exported as '{output_chart_path}'")