import cv2
import matplotlib.pyplot as plt


def plot_segmentation_comparison(
    raw_img_path, 
    mask_img_path, 
    title, 
    caption, 
    save_name
):
    raw_img = cv2.cvtColor(cv2.imread(raw_img_path), cv2.COLOR_BGR2RGB)
    mask_img = cv2.cvtColor(cv2.imread(mask_img_path), cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Before Image
    axs[0].imshow(raw_img)
    axs[0].set_title("Before: Raw Input Frame", fontsize=12, fontweight="bold")
    axs[0].axis("off")

    # After Image
    axs[1].imshow(mask_img)
    axs[1].set_title(
        "After: BiRefNet Segmentation", fontsize=12, fontweight="bold"
    )
    axs[1].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.02,
        caption,
        ha="center",
        fontsize=10,
        style="italic",
        wrap=True,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.show()



# plot_segmentation_comparison("wolf_raw.jpg", "wolf_masked.jpg", "Case Study: Low-Contrast Night Isolation", "Figure 1: Strength - Accurate boundary precision under low-contrast conditions.", "strength_1.png")