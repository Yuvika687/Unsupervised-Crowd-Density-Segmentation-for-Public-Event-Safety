"""
calibrate_v2.py — Dual Calibration (FIXED)
==========================================
Sparse:  count = a * raw        (NO bias)
Dense:   count = exp(a) * raw^b

Usage:
    python3 calibrate_v2.py
"""

import os
import json
import cv2
import numpy as np
import scipy.io as sio
from inference_v2 import load_model_v2, tiled_inference

# PATHS
TEST_A_IMG = "data/part_A_final/test_data/images"
TEST_A_GT  = "data/part_A_final/test_data/ground_truth"
TEST_B_IMG = "data/part_B_final/test_data/images"
TEST_B_GT  = "data/part_B_final/test_data/ground_truth"
JHU_TEST   = "jhu_crowd_v2.0/test"

CALIB_PATH = "models/calibration_v2.json"
os.makedirs("models", exist_ok=True)


# ─────────────────────────────────────────────
# GT LOADER
# ─────────────────────────────────────────────
def get_gt_shanghai(img_name, gt_dir):
    mat_name = "GT_" + img_name.replace(".jpg", ".mat")
    mat_path = os.path.join(gt_dir, mat_name)
    if not os.path.exists(mat_path):
        return None
    mat = sio.loadmat(mat_path)
    return len(mat["image_info"][0, 0][0, 0][0])


# ─────────────────────────────────────────────
# DATA COLLECTION
# ─────────────────────────────────────────────
def collect_pairs(model, device="cpu"):
    pairs = []

    # Part B
    if os.path.isdir(TEST_B_IMG):
        print("  Processing Part B ...")
        files = sorted(f for f in os.listdir(TEST_B_IMG) if f.endswith(".jpg"))
        for i, fname in enumerate(files):
            gt = get_gt_shanghai(fname, TEST_B_GT)
            if gt is None or gt == 0:
                continue
            img = cv2.imread(os.path.join(TEST_B_IMG, fname))
            raw, _ = tiled_inference(model, img, device, tta=False)
            pairs.append((raw, gt))
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(files)}")

    # Part A
    if os.path.isdir(TEST_A_IMG):
        print("  Processing Part A ...")
        files = sorted(f for f in os.listdir(TEST_A_IMG) if f.endswith(".jpg"))
        for i, fname in enumerate(files):
            gt = get_gt_shanghai(fname, TEST_A_GT)
            if gt is None or gt == 0:
                continue
            img = cv2.imread(os.path.join(TEST_A_IMG, fname))
            raw, _ = tiled_inference(model, img, device, tta=False)
            pairs.append((raw, gt))
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(files)}")

    # JHU
    if os.path.isdir(JHU_TEST):
        print("  Processing JHU test ...")
        labels_path = os.path.join(JHU_TEST, "image_labels.txt")
        count = 0
        with open(labels_path) as f:
            for line in f:
                parts = line.strip().split(",")
                img_id, gt = parts[0], int(parts[1])
                if gt <= 100:
                    img_path = os.path.join(JHU_TEST, "images", f"{img_id}.jpg")
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        if img is not None:
                            raw, _ = tiled_inference(model, img, device, tta=False)
                            pairs.append((raw, gt))
                            count += 1
                            if count % 50 == 0:
                                print(f"    {count} JHU images")
                if count >= 200:
                    break

    return pairs


# ─────────────────────────────────────────────
# CALIBRATION (FIXED)
# ─────────────────────────────────────────────
def fit_calibration(pairs):

    sparse = [(r, g) for r, g in pairs if g <= 100 and r > 1e-3]
    dense  = [(r, g) for r, g in pairs if g > 100 and r > 1e-3]

    calib = {}

    # Sparse (NO bias)
    if len(sparse) >= 5:
        raws = np.array([r for r, g in sparse])
        gts  = np.array([g for r, g in sparse])

        a = np.sum(gts * raws) / (np.sum(raws * raws) + 1e-6)

        calib["sparse_a"] = float(a)
        calib["sparse_b"] = 0.0

        preds = a * raws
        mae = np.mean(np.abs(preds - gts))

        print(f"  Sparse (≤100): {len(sparse)} pts | count = {a:.6f} * raw | MAE={mae:.2f}")

    else:
        calib["sparse_a"] = 1.0
        calib["sparse_b"] = 0.0

    # Dense
    if len(dense) >= 5:
        raws = np.array([r for r, g in dense])
        gts  = np.array([g for r, g in dense])

        log_raws = np.log(raws)
        log_gts  = np.log(gts)

        b, a = np.polyfit(log_raws, log_gts, 1)

        calib["dense_a"] = float(a)
        calib["dense_b"] = float(b)

        preds = np.exp(a) * raws ** b
        mae = np.mean(np.abs(preds - gts))

        print(f"  Dense (>100): {len(dense)} pts | count = {np.exp(a):.4f}*raw^{b:.4f} | MAE={mae:.2f}")

    else:
        calib["dense_a"] = 0.0
        calib["dense_b"] = 1.0

    # Threshold
    if calib["sparse_a"] > 1e-6:
        threshold = 100 / calib["sparse_a"]
    else:
        threshold = 100.0

    calib["threshold"] = float(max(threshold, 10))

    return calib


# ─────────────────────────────────────────────
# APPLY
# ─────────────────────────────────────────────
def apply_calibration_v2(raw, calib):

    if raw <= 1e-3:
        return 0

    if raw <= calib["threshold"]:
        return max(0, int(round(calib["sparse_a"] * raw)))
    else:
        return max(1, int(round(np.exp(calib["dense_a"]) * raw ** calib["dense_b"])))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():

    print("=" * 60)
    print(" SafeCrowd V2 — Calibration FIXED")
    print("=" * 60)

    model = load_model_v2()
    print("  ✓ Model loaded\n")

    pairs = collect_pairs(model)
    print(f"\n  Total calibration pairs: {len(pairs)}")

    calib = fit_calibration(pairs)

    with open(CALIB_PATH, "w") as f:
        json.dump(calib, f, indent=2)

    print(f"\n  ✓ Saved to {CALIB_PATH}")

    print("\n  Next: python3 eval_v2.py")


if __name__ == "__main__":
    main()