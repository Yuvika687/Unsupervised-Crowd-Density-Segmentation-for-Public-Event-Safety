"""
preprocess_dataset.py — Generate density maps from ShanghaiTech ground truth
=============================================================================
Generates Gaussian-smoothed density .npy files for both train AND test sets.
"""

import os
import cv2
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def generate_density_map(img, points, sigma=15):
    """Create a Gaussian-smoothed density map from head annotations."""
    density = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    for p in points:
        x = min(int(p[0]), img.shape[1] - 1)
        y = min(int(p[1]), img.shape[0] - 1)
        density[y, x] += 1
    density = gaussian_filter(density, sigma=sigma)
    return density


def process_split(split_name, data_path, output_path):
    """Generate density maps for one split (train or test)."""
    img_path = os.path.join(data_path, "images")
    gt_path  = os.path.join(data_path, "ground_truth")
    os.makedirs(output_path, exist_ok=True)

    image_files = sorted([f for f in os.listdir(img_path) if f.endswith(".jpg")])
    print(f"\nProcessing {split_name}: {len(image_files)} images → {output_path}")

    for img_file in tqdm(image_files, desc=split_name):
        img = cv2.imread(os.path.join(img_path, img_file))
        mat_path = os.path.join(gt_path, "GT_" + img_file.replace(".jpg", ".mat"))
        mat = sio.loadmat(mat_path)
        points = mat["image_info"][0][0][0][0][0]

        density = generate_density_map(img, points)

        save_path = os.path.join(output_path, img_file.replace(".jpg", ".npy"))
        np.save(save_path, density)

    print(f"  ✓ {split_name} done — {len(image_files)} density maps saved.")


if __name__ == "__main__":
    # Train set
    process_split(
        "train",
        "data/part_A_final/train_data",
        "data/density_maps",
    )
    # Test set
    process_split(
        "test",
        "data/part_A_final/test_data",
        "data/density_maps_test",
    )
    print("\nAll density maps generated successfully!")
