#
## generate_defense_plots.py
#
#
#  This script generates a multi-panel bar chart comparing the performance metrics 
#  (Precision, Recall, F1-Score) of the YOLO Gatekeeper pipeline across three stages:
#  1. Baseline (YOLO Gatekeeper only)
#  2. Optimization 1 (Soft-Gating with Raw Multi-Region Anchor S
#  3. Optimization 2 (Soft-Gating + Dense BiRefNet Saliency Fallback)
#

import numpy as np
import matplotlib.pyplot as plt

# Defining Target Categories: Wolves, Coyotes, and Domestic Dogs

classes = ['Mexican Gray Wolf', 'Coyote', 'Domestic Dog']
x = np.arange(len(classes))  # Label locations
width = 0.25                 # Width of the bars

# --- METRICS DATASET ---
# Baseline (YOLO Gatekeeper only - values based on initial run before fallback)
# Note: Since baseline dropped 51.69% of frames, recall was capped significantly.
baseline_p = [0.99, 0.99, 0.98]
baseline_r = [0.40, 0.48, 0.59]
baseline_f1 = [0.57, 0.65, 0.74]

# Optimization 1 (Soft-Gating with Raw Multi-Region Anchor Slicing)
opt1_p = [0.99, 0.99, 0.98]
opt1_r = [0.40, 0.49, 0.59]
opt1_f1 = [0.57, 0.65, 0.74]

# Optimization 2 (Final: Soft-Gating + Dense BiRefNet Saliency Fallback)
opt2_p = [0.93, 0.93, 0.90]
opt2_r = [0.63, 0.77, 0.73]
opt2_f1 = [0.75, 0.84, 0.81]

# Set up the plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available() else 'default')
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# ---------------------------------------------------------
# GRAPH 1: PRECISION COMPARISON
# ---------------------------------------------------------
axs[0].bar(x - width, baseline_p, width, label='Baseline (YOLO Gating)', color='#e74c3c', edgecolor='black')
axs[0].bar(x, opt1_p, width, label='Opt 1 (Raw Window)', color='#f39c12', edgecolor='black')
axs[0].bar(x + width, opt2_p, width, label='Opt 2 (Dense SOD Fallback)', color='#2ecc71', edgecolor='black')
axs[0].set_title('Precision Evolution Across Pipeline Phases', fontsize=12, fontweight='bold')
axs[0].set_xticks(x)
axs[0].set_xticklabels(classes, rotation=15)
axs[0].set_ylabel('Score (0.0 - 1.0)', fontsize=11)
axs[0].set_ylim(0, 1.1)

# ---------------------------------------------------------
# GRAPH 2: RECALL COMPARISON (The Core Victory Slide)
# ---------------------------------------------------------
axs[1].bar(x - width, baseline_r, width, label='Baseline (YOLO Gating)', color='#e74c3c', edgecolor='black')
axs[1].bar(x, opt1_r, width, label='Opt 1 (Raw Window)', color='#f39c12', edgecolor='black')
axs[1].bar(x + width, opt2_r, width, label='Opt 2 (Dense SOD Fallback)', color='#2ecc71', edgecolor='black')
axs[1].set_title('Recall Evolution Across Pipeline Phases', fontsize=12, fontweight='bold')
axs[1].set_xticks(x)
axs[1].set_xticklabels(classes, rotation=15)

# ---------------------------------------------------------
# GRAPH 3: F1-SCORE COMPARISON (Overall Balanced Metric)
# ---------------------------------------------------------
axs[3 - 1].bar(x - width, baseline_f1, width, label='Baseline (YOLO Gating)', color='#e74c3c', edgecolor='black')
axs[3 - 1].bar(x, opt1_f1, width, label='Opt 1 (Raw Window)', color='#f39c12', edgecolor='black')
axs[3 - 1].bar(x + width, opt2_f1, width, label='Opt 2 (Dense SOD Fallback)', color='#2ecc71', edgecolor='black')
axs[3 - 1].set_title('F1-Score Evolution Across Pipeline Phases', fontsize=12, fontweight='bold')
axs[3 - 1].set_xticks(x)
axs[3 - 1].set_xticklabels(classes, rotation=15)

# Add Legend to the center plot layout to avoid redundancy
axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=11, frameon=True)

plt.tight_layout()
output_chart_path = "pipeline_optimization_metrics.png"
plt.savefig(output_chart_path, dpi=300, bbox_inches='tight')
print(f"[Success] Multi-stage optimization comparison chart exported as '{output_chart_path}'")