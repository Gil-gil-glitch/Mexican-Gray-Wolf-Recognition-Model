import numpy as np
import matplotlib.pyplot as plt

# Defining Target Categories: Wolves, Coyotes, and Domestic Dogs
classes = ['Mexican Gray Wolf', 'Coyote', 'Domestic Dog']
x = np.arange(len(classes))  # Label locations
width = 0.18                 # Adjusted bar width for 4 side-by-side variables

# --- COMPLETE EXPERIMENTAL METRICS MATRIX ---

# 1. Ctrl A: Standalone Classifier Baseline
standalone_p = [1.00, 0.93, 0.89]
standalone_r = [0.00, 0.03, 0.00]
standalone_f1 = [0.00, 0.06, 0.01]

# 2. Ctrl B: Flat Saliency (No YOLO Gate - BiRefNet directly on full raw frame)
ctrl_b_p = [0.91, 0.92, 0.90]
ctrl_b_r = [0.45, 0.50, 0.52]
ctrl_b_f1 = [0.60, 0.65, 0.66]

# 3. Phase 1: Baseline Pipeline (YOLO Gatekeeper only - Hard Dropping configuration)
baseline_p = [0.99, 0.99, 0.98]
baseline_r = [0.40, 0.48, 0.59]
baseline_f1 = [0.57, 0.65, 0.74]

# 4. Phase 2: Final Soft-Gated Pipeline (Soft-Gating + Dense BiRefNet Saliency Fallback)
opt2_p = [0.93, 0.94, 0.92]
opt2_r = [0.62, 0.75, 0.73]  
opt2_f1 = [0.74, 0.84, 0.81]

# Plot configs
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
colors = ['#95a5a6', '#e74c3c', '#f39c12', '#3498db']
labels = [
    'Ctrl A: Standalone Classifier (Raw Images)',
    'Ctrl B: Flat Saliency (No YOLO Gate)',
    'Phase 1: Baseline (YOLO Hard Drop)',
    'Phase 2: Final Soft-Gated Pipeline'
]

# Package data structures for clean iteration loops
metrics_bundles = [
    {"name": "precision", "title": "Precision Evolution Across Layouts", "data": [standalone_p, ctrl_b_p, baseline_p, opt2_p]},
    {"name": "recall", "title": "Recall Comparison & Fallback Impact (Core Project Goal)", "data": [standalone_r, ctrl_b_r, baseline_r, opt2_r]},
    {"name": "f1_score", "title": "F1-Score Metrics Synthesis (Overall System Balance)", "data": [standalone_f1, ctrl_b_f1, baseline_f1, opt2_f1]}
]

# =========================================================================
# STEP 1: GENERATE THE INTEGRATED MULTI-PANEL CHART (For Thesis/Report Log)
# =========================================================================
fig, axs = plt.subplots(1, 3, figsize=(22, 6), sharey=True)

for idx, bundle in enumerate(metrics_bundles):
    d = bundle["data"]
    axs[idx].bar(x - 1.5*width, d[0], width, label=labels[0], color=colors[0], edgecolor='black')
    axs[idx].bar(x - 0.5*width, d[1], width, label=labels[1], color=colors[1], edgecolor='black')
    axs[idx].bar(x + 0.5*width, d[2], width, label=labels[2], color=colors[2], edgecolor='black')
    axs[idx].bar(x + 1.5*width, d[3], width, label=labels[3], color=colors[3], edgecolor='black')
    
    axs[idx].set_title(bundle["title"], fontsize=13, fontweight='bold')
    axs[idx].set_xticks(x)
    axs[idx].set_xticklabels(classes, rotation=10, fontsize=11)
    if idx == 0:
        axs[idx].set_ylabel('Score (0.0 - 1.0)', fontsize=12)
    axs[idx].set_ylim(0, 1.15)

axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=11, frameon=True)
plt.tight_layout()
plt.savefig("pipeline_optimization_metrics.png", dpi=300, bbox_inches='tight')
plt.close()
print("[Success] Combined reference graphic exported as 'pipeline_optimization_metrics.png'")


# =========================================================================
# STEP 2: GENERATE THE INDIVIDUAL SLIDE GRAPHICS (For Slide Deck Presentation)
# =========================================================================
for i, bundle in enumerate(metrics_bundles):
    fig, ax = plt.subplots(figsize=(9, 5))
    d = bundle["data"]
    
    ax.bar(x - 1.5*width, d[0], width, label=labels[0], color=colors[0], edgecolor='black')
    ax.bar(x - 0.5*width, d[1], width, label=labels[1], color=colors[1], edgecolor='black')
    ax.bar(x + 0.5*width, d[2], width, label=labels[2], color=colors[2], edgecolor='black')
    ax.bar(x + 1.5*width, d[3], width, label=labels[3], color=colors[3], edgecolor='black')
    
    ax.set_title(bundle["title"], fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel('Score (0.0 - 1.0)', fontsize=12)
    ax.set_ylim(0, 1.3)  
    
    ax.legend(loc='upper right', fontsize=8.5, frameon=True, shadow=False)
    
    plt.tight_layout()
    slide_filename = f"evolution_{i+1}_{bundle['name']}.png"
    plt.savefig(slide_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Success] Presentation slide asset exported as '{slide_filename}'")