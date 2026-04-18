import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUTPUT_DIR = "results/safety_maps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_MAP = {
    "Low":      (0,   200, 0),
    "Medium":   (255, 200, 0),
    "High":     (255, 120, 0),
    "Critical": (220, 0,   0),
}

def draw_safety_overlay(img_path, patch_label_grid, grid=8, save_path=None):
    img     = cv2.imread(img_path)
    img     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w    = img.shape[:2]
    overlay = img.copy()
    ph, pw  = h // grid, w // grid

    for i in range(grid):
        for j in range(grid):
            label     = patch_label_grid[i, j]
            r, g, b   = COLOR_MAP[label]
            y1, y2    = i*ph, (i+1)*ph
            x1, x2    = j*pw, (j+1)*pw
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (r,g,b), -1)

    blended = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)

    # Check for critical zones
    critical_count = np.sum(patch_label_grid == "Critical")
    alert = critical_count > 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(blended)
    title = f"Safety map {'— CRITICAL ALERT!' if alert else '— Safe'}"
    axes[1].set_title(title, color="red" if alert else "green")
    axes[1].axis("off")

    legend = [mpatches.Patch(color=(r/255,g/255,b/255), label=lbl)
              for lbl,(r,g,b) in COLOR_MAP.items()]
    axes[1].legend(handles=legend, loc="lower right", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()



IMAGE_DIR    = "data/part_A_final/test_data/images"
DENSITY_DIR  = "data/density_maps"

img_files    = sorted(os.listdir(IMAGE_DIR))[:5]   # run on first 5 images

for img_file in img_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    dm_path  = os.path.join(DENSITY_DIR, img_file.replace(".jpg", ".npy"))

    if not os.path.exists(dm_path):
        continue

    dm = np.load(dm_path)

    # Build 8x8 patch label grid
    grid = 8
    h, w = dm.shape
    ph, pw = h//grid, w//grid
    patch_grid = np.empty((grid, grid), dtype=object)

    for i in range(grid):
        for j in range(grid):
            patch      = dm[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            mean_val   = patch.mean()
            if   mean_val < 0.01:  patch_grid[i,j] = "Low"
            elif mean_val < 0.05:  patch_grid[i,j] = "Medium"
            elif mean_val < 0.15:  patch_grid[i,j] = "High"
            else:                  patch_grid[i,j] = "Critical"

    save_path = f"results/safety_maps/{img_file.replace('.jpg','_safety.png')}"
    draw_safety_overlay(img_path, patch_grid, save_path=save_path)
