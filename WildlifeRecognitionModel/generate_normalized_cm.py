import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. Define the raw confusion matrix values from your image
# Rows = True label, Columns = Predicted label
cm_raw = np.array([[1957, 179, 68], [132, 3694, 93], [6, 110, 1515]], dtype=float)

# 2. Normalize by true labels (divide each row by its sum)
cm_normalized = cm_raw / cm_raw.sum(axis=1)[:, np.newaxis]

# Define labels
labels = ["Mexican Gray Wolf", "Coyote", "Domestic Dog"]

# 3. Plot the normalized confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    cbar=True,
    square=True,
)

# Formatting
plt.title("Optimized Soft-Gated Pipeline Normalized Confusion Matrix", fontsize=14, pad=15)
plt.xlabel("Predicted label", fontsize=12)
plt.ylabel("True label", fontsize=12)
plt.tight_layout()

# Save and show
plt.savefig("normalized_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()