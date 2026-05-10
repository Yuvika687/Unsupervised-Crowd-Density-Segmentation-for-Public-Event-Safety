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
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.inference_v2 import load_model_v2, tiled_inference

# PATHS
TEST_A_IMG = "data/part_A_final/test_data/images"
TEST_A_GT  = "data/part_A_final/test_data/ground_truth"
TEST_B_IMG = "data/part_B_final/test_data/images"
TEST_B_GT  = "data/part_B_final/test_data/ground_truth"
JHU_TEST   = "jhu_crowd_v2.0/test"

CALIB_PATH = "models/calibration_v2.json"
os.makedirs("models", exist_ok=True)

INFERENCE_VERSION = 2
RAW_SIGNAL = "overlap_weighted_regression"
CALIBRATE_WITH_TTA = True


def _log(msg=""):
    print(msg, flush=True)


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
        _log("  Processing Part B ...")
        files = sorted(f for f in os.listdir(TEST_B_IMG) if f.endswith(".jpg"))
        for i, fname in enumerate(files):
            gt = get_gt_shanghai(fname, TEST_B_GT)
            if gt is None or gt == 0:
                continue
            img = cv2.imread(os.path.join(TEST_B_IMG, fname))
            raw, _ = tiled_inference(
                model, img, device, tta=CALIBRATE_WITH_TTA)
            pairs.append((raw, gt))
            if (i + 1) % 50 == 0:
                _log(f"    {i+1}/{len(files)}")

    # Part A
    if os.path.isdir(TEST_A_IMG):
        _log("  Processing Part A ...")
        files = sorted(f for f in os.listdir(TEST_A_IMG) if f.endswith(".jpg"))
        for i, fname in enumerate(files):
            gt = get_gt_shanghai(fname, TEST_A_GT)
            if gt is None or gt == 0:
                continue
            img = cv2.imread(os.path.join(TEST_A_IMG, fname))
            raw, _ = tiled_inference(
                model, img, device, tta=CALIBRATE_WITH_TTA)
            pairs.append((raw, gt))
            if (i + 1) % 50 == 0:
                _log(f"    {i+1}/{len(files)}")

    # JHU
    if os.path.isdir(JHU_TEST):
        _log("  Processing JHU test ...")
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
                            raw, _ = tiled_inference(
                                model, img, device, tta=CALIBRATE_WITH_TTA)
                            pairs.append((raw, gt))
                            count += 1
                            if count % 50 == 0:
                                _log(f"    {count} JHU images")
                if count >= 200:
                    break

    return pairs


# ─────────────────────────────────────────────
# CALIBRATION (FIXED)
# ─────────────────────────────────────────────
def fit_calibration(pairs):

    sparse = [(r, g) for r, g in pairs if g <= 100 and r > 1e-3]
    dense  = [(r, g) for r, g in pairs if g > 100 and r > 1e-3]

    calib = {
        "inference_version": INFERENCE_VERSION,
        "raw_signal": RAW_SIGNAL,
        "tta": bool(CALIBRATE_WITH_TTA),
    }

    # Sparse (NO bias)
    if len(sparse) >= 5:
        raws = np.array([r for r, g in sparse])
        gts  = np.array([g for r, g in sparse])

        a = np.sum(gts * raws) / (np.sum(raws * raws) + 1e-6)

        calib["sparse_a"] = float(a)
        calib["sparse_b"] = 0.0

        preds = a * raws
        mae = np.mean(np.abs(preds - gts))

        _log(
            f"  Sparse (≤100): {len(sparse)} pts | "
            f"count = {a:.6f} * raw | MAE={mae:.2f}"
        )

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

        _log(
            f"  Dense (>100): {len(dense)} pts | "
            f"count = {np.exp(a):.4f}*raw^{b:.4f} | MAE={mae:.2f}"
        )

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


def is_calibration_compatible(calib):
    return (
        isinstance(calib, dict) and
        calib.get("inference_version") == INFERENCE_VERSION and
        calib.get("raw_signal") == RAW_SIGNAL
    )


# ─────────────────────────────────────────────
# APPLY
# ─────────────────────────────────────────────
def apply_calibration_v2(raw, calib):
    if not is_calibration_compatible(calib):
        return max(0, int(round(raw)))

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

    _log("=" * 60)
    _log(" SafeCrowd V2 — Calibration FIXED")
    _log("=" * 60)
    _log(f"  TTA during calibration: {CALIBRATE_WITH_TTA}")

    model = load_model_v2()
    _log("  ✓ Model loaded\n")

    pairs = collect_pairs(model)
    _log(f"\n  Total calibration pairs: {len(pairs)}")

    calib = fit_calibration(pairs)

    with open(CALIB_PATH, "w") as f:
        json.dump(calib, f, indent=2)

    _log(f"\n  ✓ Saved to {CALIB_PATH}")
    _log("\n  Next: python3 eval_v2.py")


if __name__ == "__main__":
    main()
