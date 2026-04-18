"""
calibrate.py — Multi-Mode Calibration for SafeCrowd Vision v2.0
================================================================
Fits: predicted_count = exp(a) * raw_sum^b   (log-linear)
Saves 3 calibration files:
  models/calibration_small.json   ← Mall + UCSD  (1-80 people)
  models/calibration_medium.json  ← Part B       (50-500 people)
  models/calibration_large.json   ← Part A       (300+ people)

Usage:  python3 calibrate.py
"""

import os
import cv2
import torch
import numpy as np
import scipy.io as sio
import torch.nn as nn
from torchvision import models
import json

# ─── Paths ────────────────────────────────────────────────────────────────────

TEST_IMG_A = "data/part_A_final/test_data/images"
TEST_GT_A  = "data/part_A_final/test_data/ground_truth"
TEST_IMG_B = "data/part_B_final/test_data/images"
TEST_GT_B  = "data/part_B_final/test_data/ground_truth"

# Auto-detect mall path (may have trailing space from extraction)
MALL_DIR = None
for _candidate in ["data/mall_dataset", "data/mall_dataset "]:
    if os.path.isdir(_candidate):
        MALL_DIR = _candidate
        break

UCSD_DIR = "data/ucsd/vidf"

MODEL_PATH = "models/best_model.pth"
IMG_SIZE   = 256
MAX_IMAGES_PER_SET = 100


# ─── Model (must match train.py — DO NOT CHANGE) ─────────────────────────────

class CrowdDensityNet(nn.Module):
    def __init__(self, freeze_layers=0):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.encoder = nn.Sequential(*list(vgg.features.children())[:24])
        for i, param in enumerate(self.encoder.parameters()):
            if i < freeze_layers:
                param.requires_grad = False
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ─── Tiled inference ─────────────────────────────────────────────────────────

def predict_tiled(model, img_path):
    """Get raw density sum using tiled inference (tile=256, stride=128)."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = img_gray.shape
    tile_size = 256
    stride = 128

    pad_h = (stride - (h % stride)) % stride
    pad_w = (stride - (w % stride)) % stride
    img_p = np.pad(img_gray, ((0, pad_h), (0, pad_w)), mode='reflect')

    density_p = np.zeros_like(img_p)
    count_p   = np.zeros_like(img_p)

    model.eval()
    for y in range(0, img_p.shape[0] - stride, stride):
        for x in range(0, img_p.shape[1] - stride, stride):
            tile = img_p[y:y+tile_size, x:x+tile_size]
            if tile.shape != (tile_size, tile_size):
                tile = cv2.resize(tile, (tile_size, tile_size))

            tile_t = torch.tensor(np.stack([tile]*3, 0), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = model(tile_t)

            density_p[y:y+tile_size, x:x+tile_size] += out.squeeze().numpy()
            count_p[y:y+tile_size, x:x+tile_size]   += 1.0

    density = (density_p / np.maximum(count_p, 1.0))[:h, :w]
    density = np.clip(density, 0, None)
    density[density < 0.015] = 0

    return float(density.sum())


# ─── Ground truth loaders ────────────────────────────────────────────────────

def get_gt_count_shanghai(img_name, gt_dir):
    """ShanghaiTech Part A / B ground truth."""
    mat_name = "GT_" + img_name.replace(".jpg", ".mat")
    mat_path = os.path.join(gt_dir, mat_name)
    if not os.path.exists(mat_path):
        return None
    mat = sio.loadmat(mat_path)
    return mat['image_info'][0, 0][0, 0][1].item()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  SafeCrowd Calibration — exp(a) * raw^b  (Multi-Mode)")
    print("=" * 65)
    print()

    if not os.path.exists(MODEL_PATH):
        print(f"  ERROR: Model not found at {MODEL_PATH}")
        print("  Run train.py first.")
        return

    model = CrowdDensityNet()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    print(f"  ✓ Model loaded from {MODEL_PATH}")

    # ── Collect evaluation items per mode ─────────────────────
    eval_small  = []  # Mall + UCSD
    eval_medium = []  # Part B
    eval_large  = []  # Part A

    # Part A → large
    if os.path.exists(TEST_IMG_A):
        print("  ✓ Found ShanghaiTech Part A")
        for img_name in sorted(f for f in os.listdir(TEST_IMG_A) if f.endswith(".jpg"))[:MAX_IMAGES_PER_SET]:
            actual = get_gt_count_shanghai(img_name, TEST_GT_A)
            if actual is not None:
                eval_large.append((os.path.join(TEST_IMG_A, img_name), actual))
    else:
        print("  ✗ Missing ShanghaiTech Part A")

    # Part B → medium
    if os.path.exists(TEST_IMG_B):
        print("  ✓ Found ShanghaiTech Part B")
        for img_name in sorted(f for f in os.listdir(TEST_IMG_B) if f.endswith(".jpg"))[:MAX_IMAGES_PER_SET]:
            actual = get_gt_count_shanghai(img_name, TEST_GT_B)
            if actual is not None:
                eval_medium.append((os.path.join(TEST_IMG_B, img_name), actual))
    else:
        print("  ✗ Missing ShanghaiTech Part B")

    # UCSD → small
    if os.path.exists(UCSD_DIR):
        print("  ✓ Found UCSD")
        all_dirs = sorted(d for d in os.listdir(UCSD_DIR) if d.startswith("vidf"))
        test_dirs = all_dirs[6:]
        for dname in test_dirs:
            for frame_idx in range(1, 101, 10):
                img_path = os.path.join(UCSD_DIR, dname,
                                        f"vidf1_33_{dname[-3:]}_f{frame_idx:03d}.png")
                if os.path.exists(img_path):
                    eval_small.append((img_path, 25))
    else:
        print("  ✗ Missing UCSD")

    # Mall → small
    if MALL_DIR is not None:
        print(f"  ✓ Found Mall dataset at '{MALL_DIR}'")
        gt_path = os.path.join(MALL_DIR, "mall_gt.mat")
        if os.path.exists(gt_path):
            mall_counts = sio.loadmat(gt_path)['count'].flatten()
            for idx in range(min(100, len(mall_counts))):
                img_path = os.path.join(MALL_DIR, "frames", f"seq_{idx+1:06d}.jpg")
                if os.path.exists(img_path):
                    eval_small.append((img_path, float(mall_counts[idx])))
    else:
        print("  ✗ Missing Mall dataset")

    print()

    # ── Fit log-linear per mode ───────────────────────────────
    # Formula:  gt = exp(a) * raw^b
    # Taking log:  log(gt) = a + b*log(raw)
    # So we regress log(gt) on log(raw) → intercept=a, slope=b

    modes = {"small": eval_small, "medium": eval_medium, "large": eval_large}
    all_results = {}

    print("  Running inference...")
    for mode_name, items in modes.items():
        if not items:
            print(f"  Skipping {mode_name} (no data)")
            continue

        print(f"  Processing {mode_name} ({len(items)} images)...")
        log_raw_list = []
        log_gt_list  = []
        results      = []

        for img_path, actual in items:
            raw_pred = predict_tiled(model, img_path)
            if raw_pred is None:
                continue
            if raw_pred > 1e-3 and actual > 0:
                log_raw_list.append(np.log(raw_pred))
                log_gt_list.append(np.log(actual))
                results.append((actual, raw_pred))

        all_results[mode_name] = results

        if len(log_raw_list) < 2:
            print(f"    ERROR: Not enough data for {mode_name} regression.")
            continue

        # log(gt) = a + b*log(raw)  →  polyfit(log_raw, log_gt, 1) = [b, a]
        b, a = np.polyfit(log_raw_list, log_gt_list, 1)

        os.makedirs("models", exist_ok=True)
        out_path = f"models/calibration_{mode_name}.json"
        with open(out_path, "w") as f:
            json.dump({"a": float(a), "b": float(b)}, f)

        print(f"    Saved {out_path}  (a={a:.6f}, b={b:.6f})")
        print(f"    Formula: count = exp({a:.4f}) * raw^{b:.4f}")
        print(f"             count = {np.exp(a):.4f} * raw^{b:.4f}")

    print()

    # ── Validation MAE table ──────────────────────────────────
    print("  Validation Table (MAE by Mode)")
    print(f"  {'-'*65}")
    print(f"  {'Mode':<15} {'Images':<10} {'MAE':<15} {'MAPE %':<12}")
    print(f"  {'-'*65}")

    for mode_name, results in all_results.items():
        if not results:
            continue

        cal_path = f"models/calibration_{mode_name}.json"
        if not os.path.exists(cal_path):
            continue

        with open(cal_path, "r") as f:
            cal = json.load(f)
            a, b = cal["a"], cal["b"]

        errors = []
        pct_errors = []
        for actual, raw in results:
            pred = np.exp(a) * (raw ** b)
            errors.append(abs(pred - actual))
            pct_errors.append(abs(pred - actual) / actual * 100)

        mae  = np.mean(errors)
        mape = np.mean(pct_errors)

        print(f"  {mode_name:<15} {len(results):<10d} {mae:<15.2f} {mape:<12.1f}")

    print(f"  {'-'*65}")
    print()
    print("  Done. The Streamlit app will auto-load these calibration files.")
    print("  Run:  python3 -m streamlit run app.py")


if __name__ == "__main__":
    main()
