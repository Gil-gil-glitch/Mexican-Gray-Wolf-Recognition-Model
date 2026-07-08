import numpy as np
import matplotlib.pyplot as plt

# Defining Target Categories: Wolves, Coyotes, and Domestic Dogs
classes = ['Mexican Gray Wolf', 'Coyote', 'Domestic Dog']
x = np.arange(len(classes))  # Label locations
width = 0.15                 # Optimal bar width for 5 side-by-side variables

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


# Plot configs
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
colors = ['#95a5a6', '#e74c3c', '#f39c12', '#2ecc71', '#3498db']
labels = [
    'Ctrl A: Standalone Classifier (No Pipeline)',
    'Ctrl B: Flat Saliency (No YOLO Gate)',
    'Phase 1: Baseline (YOLO Hard Drop)',
    'Phase 2: Opt 1 (Raw Window Slicing)'
]

# Package data structures for clean iteration loops
metrics_bundles = [
    {"name": "precision", "title": "Precision Evolution Across Layouts", "data": [standalone_p, baseline_p, opt1_p, opt2_p]},
    {"name": "recall", "title": "Recall Comparison & Fallback Impact (Core Project Goal)", "data": [standalone_r, baseline_r, opt1_r, opt2_r]},
    {"name": "f1_score", "title": "F1-Score Metrics Synthesis (Overall System Balance)", "data": [standalone_f1, baseline_f1, opt1_f1, opt2_f1]}
]

# =========================================================================
# STEP 1: GENERATE THE INTEGRATED MULTI-PANEL CHART (For Thesis/Report Log)
# =========================================================================
fig, axs = plt.subplots(1, 3, figsize=(22, 6), sharey=True)

for idx, bundle in enumerate(metrics_bundles):
    d = bundle["data"]
    axs[idx].bar(x - 2*width, d[0], width, label=labels[0], color=colors[0], edgecolor='black')
    axs[idx].bar(x - width, d[1], width, label=labels[1], color=colors[1], edgecolor='black')
    axs[idx].bar(x, d[2], width, label=labels[2], color=colors[2], edgecolor='black')
    axs[idx].bar(x + width, d[3], width, label=labels[3], color=colors[3], edgecolor='black')
    
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
    # Higher aspect ratio (8x5) fits widescreen 16:9 slides beautifully
    fig, ax = plt.subplots(figsize=(8, 5))
    d = bundle["data"]
    
    ax.bar(x - 2*width, d[0], width, label=labels[0], color=colors[0], edgecolor='black')
    ax.bar(x - width, d[1], width, label=labels[1], color=colors[1], edgecolor='black')
    ax.bar(x, d[2], width, label=labels[2], color=colors[2], edgecolor='black')
    ax.bar(x + width, d[3], width, label=labels[3], color=colors[3], edgecolor='black')
    
    ax.set_title(bundle["title"], fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel('Score (0.0 - 1.0)', fontsize=12)
    ax.set_ylim(0, 1.4)  
    
    # Pack the legend cleanly onto the individual chart itself
    ax.legend(loc='upper right', fontsize=8.5, frameon=True, shadow=False)
    
    plt.tight_layout()
    slide_filename = f"evolution_{i+1}_{bundle['name']}.png"
    plt.savefig(slide_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Success] Presentation slide asset exported as '{slide_filename}'")