"""
eval_hybrid.py — Bucket-wise Hybrid vs YOLO vs VGG Evaluation
===============================================================
Splits the ShanghaiTech Part A + Part B test sets into 3 density
buckets (Sparse / Medium / Dense) and computes MAE, RMSE, and MAPE
for each of:

  (a) YOLO-only  — ultralytics YOLOv8 person detector (class 0)
  (b) VGG-only   — CrowdDensityNet density-map regression + calibration
  (c) Hybrid      — YOLO for sparse, VGG for dense, smooth blend between

Produces:
  • Console table      (tabulate)
  • results/eval_hybrid_table.csv
  • results/eval_hybrid_bar.png         — grouped MAE bar chart
  • results/eval_hybrid_scatter.png     — pred vs GT scatter per method
  • results/eval_hybrid_bucket_box.png  — box-plot of errors by bucket

Usage
-----
    pip install ultralytics tabulate matplotlib   # one-time
    python3 eval_hybrid.py

If YOLO weights are missing it auto-downloads yolov8n.pt (~6 MB).
"""

import os
import json
import cv2
import torch
import numpy as np
import scipy.io as sio
import torch.nn as nn
from torchvision import models
from tabulate import tabulate
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

IMG_SIZE = 256

# ShanghaiTech paths (Parts A + B test)
DATASETS = [
    ("Part A", "data/part_A_final/test_data/images",
               "data/part_A_final/test_data/ground_truth"),
    ("Part B", "data/part_B_final/test_data/images",
               "data/part_B_final/test_data/ground_truth"),
]

MODEL_PATH = "models/best_model.pth"
os.makedirs("results", exist_ok=True)

# Bucket edges (GT count)
SPARSE_MAX  = 50
MEDIUM_MAX  = 200
BUCKET_NAMES = ["Sparse (1–50)", "Medium (50–200)", "Dense (200+)"]

# Hybrid transition thresholds (tune as needed)
YOLO_UPPER  = 60   # below this: 100 % YOLO
VGG_LOWER   = 120  # above this: 100 % VGG
# Between YOLO_UPPER and VGG_LOWER we linearly blend.


# ═══════════════════════════════════════════════════════════════
# MODEL DEFINITION (copy from train.py to be self-contained)
# ═══════════════════════════════════════════════════════════════

class CrowdDensityNet(nn.Module):
    def __init__(self, freeze_layers=10):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.encoder = nn.Sequential(*list(vgg.features.children())[:24])
        for i, param in enumerate(self.encoder.parameters()):
            if i < freeze_layers:
                param.requires_grad = False
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def load_vgg_model():
    model = CrowdDensityNet()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


def vgg_tiled_predict(model, img_bgr):
    """Tiled inference → raw density sum (un-calibrated)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    tile, stride = IMG_SIZE, IMG_SIZE // 2

    pad_h = (stride - (h % stride)) % stride
    pad_w = (stride - (w % stride)) % stride
    img_p = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="reflect")

    den_p = np.zeros_like(img_p)
    cnt_p = np.zeros_like(img_p)

    for y in range(0, img_p.shape[0] - stride, stride):
        for x in range(0, img_p.shape[1] - stride, stride):
            t = img_p[y:y+tile, x:x+tile]
            if t.shape != (tile, tile):
                t = cv2.resize(t, (tile, tile))
            t_t = torch.tensor(np.stack([t]*3, 0), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = model(t_t)
            den_p[y:y+tile, x:x+tile] += out.squeeze().numpy()
            cnt_p[y:y+tile, x:x+tile] += 1.0

    density = (den_p / np.maximum(cnt_p, 1.0))[:h, :w]
    density = np.clip(density, 0, None)
    density[density < 0.015] = 0
    kernel = np.ones((3, 3), np.uint8)
    mask = (density > 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    density = density * mask.astype(np.float32)
    return float(density.sum())


def load_calibration(mode: str):
    path = f"models/calibration_{mode}.json"
    fallback = "models/calibration.json"
    target = path if os.path.exists(path) else fallback
    try:
        with open(target) as f:
            c = json.load(f)
        return float(c["a"]), float(c["b"])
    except Exception:
        return 0.0, 1.0


def apply_calibration(raw_sum: float, mode: str) -> int:
    if raw_sum <= 1e-3:
        return 0
    a, b = load_calibration(mode)
    return max(1, int(round(np.exp(a) * (raw_sum ** b))))


def auto_mode(gt_count):
    """Pick the correct calibration mode based on GT bucket."""
    if gt_count <= SPARSE_MAX:
        return "small"
    elif gt_count <= MEDIUM_MAX:
        return "medium"
    else:
        return "large"


def load_yolo():
    """Load YOLOv8-nano. Auto-downloads ~6 MB if missing."""
    try:
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")          # caches locally
    except ImportError:
        print("  ⚠  ultralytics not installed — YOLO column will be NaN.")
        print("     Fix:  pip install ultralytics")
        return None


def yolo_count(yolo_model, img_bgr):
    """Count persons (class 0) detected by YOLO."""
    if yolo_model is None:
        return float("nan")
    results = yolo_model(img_bgr, verbose=False)
    boxes = results[0].boxes
    # class 0 = person in COCO
    return int((boxes.cls == 0).sum())


def get_gt_count(img_name, gt_dir):
    mat_name = "GT_" + img_name.replace(".jpg", ".mat")
    mat_path = os.path.join(gt_dir, mat_name)
    if not os.path.exists(mat_path):
        return None
    mat = sio.loadmat(mat_path)
    pts = mat["image_info"][0, 0][0, 0][0]
    return len(pts)


def bucket_index(gt):
    if gt <= SPARSE_MAX:
        return 0
    elif gt <= MEDIUM_MAX:
        return 1
    else:
        return 2


def hybrid_count(yolo_cnt, vgg_cnt):
    """
    Blended hybrid prediction.
    - Below YOLO_UPPER → pure YOLO
    - Above VGG_LOWER  → pure VGG
    - Between → linearly interpolate  (alpha goes 0→1)
    Uses the YOLO count as the "rough estimate" to decide the regime.
    """
    rough = yolo_cnt  # initial estimate from YOLO for regime decision
    if rough <= YOLO_UPPER:
        return yolo_cnt
    elif rough >= VGG_LOWER:
        return vgg_cnt
    else:
        alpha = (rough - YOLO_UPPER) / (VGG_LOWER - YOLO_UPPER)
        return int(round((1 - alpha) * yolo_cnt + alpha * vgg_cnt))


# ═══════════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════

def run_evaluation():
    print("=" * 70)
    print("  Hybrid Crowd Counting — Bucket-wise Evaluation")
    print("=" * 70)

    vgg_model = load_vgg_model()
    print("  ✓ VGG CrowdDensityNet loaded")

    yolo_model = load_yolo()
    if yolo_model is not None:
        print("  ✓ YOLOv8-nano loaded")

    # ── Collect per-image results ─────────────────────────────
    records = []  # (gt, yolo_pred, vgg_pred, hybrid_pred, bucket_idx, dataset)

    for ds_name, img_dir, gt_dir in DATASETS:
        if not os.path.isdir(img_dir):
            print(f"  ✗ Skipping {ds_name}: {img_dir} not found")
            continue

        files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
        print(f"\n  Processing {ds_name} ({len(files)} images) ...")

        for i, fname in enumerate(files):
            gt = get_gt_count(fname, gt_dir)
            if gt is None or gt == 0:
                continue

            img_path = os.path.join(img_dir, fname)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            # --- YOLO ---
            y_cnt = yolo_count(yolo_model, img_bgr)

            # --- VGG (with auto-mode calibration) ---
            raw_sum = vgg_tiled_predict(vgg_model, img_bgr)
            mode = auto_mode(gt)  # oracle mode for fair comparison
            v_cnt = apply_calibration(raw_sum, mode)

            # --- Hybrid ---
            h_cnt = hybrid_count(y_cnt, v_cnt)

            records.append((gt, y_cnt, v_cnt, h_cnt, bucket_index(gt), ds_name))

            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(files)} done")

    if not records:
        print("  ERROR: no images evaluated.")
        return

    # ── Convert to arrays ─────────────────────────────────────
    gt_arr    = np.array([r[0] for r in records], dtype=float)
    yolo_arr  = np.array([r[1] for r in records], dtype=float)
    vgg_arr   = np.array([r[2] for r in records], dtype=float)
    hybrid_arr= np.array([r[3] for r in records], dtype=float)
    bucket_arr= np.array([r[4] for r in records], dtype=int)

    # ── Metrics per bucket ────────────────────────────────────
    def mae(pred, gt):
        return float(np.mean(np.abs(pred - gt)))

    def rmse(pred, gt):
        return float(np.sqrt(np.mean((pred - gt) ** 2)))

    def mape(pred, gt):
        valid = gt > 0
        return float(np.mean(np.abs(pred[valid] - gt[valid]) / gt[valid] * 100))

    print("\n")
    rows = []
    for b in range(3):
        mask = bucket_arr == b
        n = int(mask.sum())
        if n == 0:
            rows.append([BUCKET_NAMES[b], 0, "—", "—", "—", "—", "—", "—", "—", "—"])
            continue
        g = gt_arr[mask]
        y = yolo_arr[mask]
        v = vgg_arr[mask]
        h = hybrid_arr[mask]
        rows.append([
            BUCKET_NAMES[b], n,
            f"{mae(y, g):.1f}", f"{mae(v, g):.1f}", f"{mae(h, g):.1f}",
            f"{rmse(y, g):.1f}", f"{rmse(v, g):.1f}", f"{rmse(h, g):.1f}",
            f"{mape(y, g):.1f}%", f"{mape(v, g):.1f}%", f"{mape(h, g):.1f}%",
        ])

    # Overall
    rows.append([
        "OVERALL", len(gt_arr),
        f"{mae(yolo_arr, gt_arr):.1f}",
        f"{mae(vgg_arr, gt_arr):.1f}",
        f"{mae(hybrid_arr, gt_arr):.1f}",
        f"{rmse(yolo_arr, gt_arr):.1f}",
        f"{rmse(vgg_arr, gt_arr):.1f}",
        f"{rmse(hybrid_arr, gt_arr):.1f}",
        f"{mape(yolo_arr, gt_arr):.1f}%",
        f"{mape(vgg_arr, gt_arr):.1f}%",
        f"{mape(hybrid_arr, gt_arr):.1f}%",
    ])

    headers = ["Bucket", "N",
               "MAE\nYOLO", "MAE\nVGG", "MAE\nHybrid",
               "RMSE\nYOLO", "RMSE\nVGG", "RMSE\nHybrid",
               "MAPE\nYOLO", "MAPE\nVGG", "MAPE\nHybrid"]

    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()

    # ── Save CSV ──────────────────────────────────────────────
    csv_path = "results/eval_hybrid_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "n",
                     "mae_yolo", "mae_vgg", "mae_hybrid",
                     "rmse_yolo", "rmse_vgg", "rmse_hybrid",
                     "mape_yolo", "mape_vgg", "mape_hybrid"])
        for r in rows:
            w.writerow(r)
    print(f"  ✓ Table saved → {csv_path}")

    # ── Save per-image CSV ────────────────────────────────────
    detail_path = "results/eval_hybrid_detail.csv"
    with open(detail_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt", "yolo", "vgg", "hybrid", "bucket", "dataset"])
        for r in records:
            w.writerow(r)
    print(f"  ✓ Per-image detail → {detail_path}")

    # ═══════════════════════════════════════════════════════════
    # VISUALISATIONS
    # ═══════════════════════════════════════════════════════════

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.facecolor": "#0D1117",
        "figure.facecolor": "#0D1117",
        "axes.edgecolor": "#30363D",
        "axes.labelcolor": "#C9D1D9",
        "xtick.color": "#8B949E",
        "ytick.color": "#8B949E",
        "text.color": "#C9D1D9",
        "grid.color": "#21262D",
    })

    COLORS = {"YOLO": "#58A6FF", "VGG": "#F5A623", "Hybrid": "#3FB950"}

    # ── 1. Grouped MAE bar chart ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(3)
    width = 0.22
    for offset, (method, arr) in enumerate([
            ("YOLO", yolo_arr), ("VGG", vgg_arr), ("Hybrid", hybrid_arr)]):
        maes = []
        for b in range(3):
            m = bucket_arr == b
            maes.append(mae(arr[m], gt_arr[m]) if m.sum() > 0 else 0)
        bars = ax.bar(x + offset * width, maes, width,
                      label=method, color=COLORS[method],
                      edgecolor="#30363D", linewidth=0.5)
        for bar, val in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=9, color="#C9D1D9", fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(BUCKET_NAMES)
    ax.set_ylabel("MAE ↓")
    ax.set_title("MAE by Density Bucket — YOLO vs VGG vs Hybrid",
                 fontweight="bold", fontsize=14)
    ax.legend(framealpha=0.3, edgecolor="#30363D")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/eval_hybrid_bar.png", dpi=200)
    print("  ✓ Chart  → results/eval_hybrid_bar.png")
    plt.close(fig)

    # ── 2. Pred vs GT scatter ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, (method, arr, color) in zip(axes, [
            ("YOLO", yolo_arr, COLORS["YOLO"]),
            ("VGG",  vgg_arr,  COLORS["VGG"]),
            ("Hybrid", hybrid_arr, COLORS["Hybrid"])]):
        ax.scatter(gt_arr, arr, c=color, alpha=0.5, s=18, edgecolors="none")
        lim = max(gt_arr.max(), arr.max()) * 1.05
        ax.plot([0, lim], [0, lim], "--", color="#DA3633", linewidth=1,
                label="Perfect")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted")
        ax.set_title(method, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.3, edgecolor="#30363D")
        ax.grid(alpha=0.2)
    fig.suptitle("Predicted vs Ground Truth", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("results/eval_hybrid_scatter.png", dpi=200)
    print("  ✓ Chart  → results/eval_hybrid_scatter.png")
    plt.close(fig)

    # ── 3. Box-plot of absolute errors by bucket ──────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = []
    labels_arr = []
    data = []
    for b in range(3):
        m = bucket_arr == b
        if m.sum() == 0:
            continue
        for method, arr, color in [
                ("YOLO", yolo_arr, COLORS["YOLO"]),
                ("VGG",  vgg_arr,  COLORS["VGG"]),
                ("Hybrid", hybrid_arr, COLORS["Hybrid"])]:
            data.append(np.abs(arr[m] - gt_arr[m]))
            positions.append(b * 4 + list(COLORS.keys()).index(
                [k for k, v in COLORS.items() if v == color][0]))
            labels_arr.append(f"{BUCKET_NAMES[b]}\n{method}")

    bp = ax.boxplot(data, positions=positions, widths=0.7,
                    patch_artist=True, showfliers=False)
    colour_cycle = list(COLORS.values())
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colour_cycle[i % 3] + "66")
        patch.set_edgecolor(colour_cycle[i % 3])
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("#8B949E")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_arr, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Error Distribution by Bucket",
                 fontweight="bold", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/eval_hybrid_bucket_box.png", dpi=200)
    print("  ✓ Chart  → results/eval_hybrid_bucket_box.png")
    plt.close(fig)

    # ── 4. Sparse-focus: prove hybrid ≈ YOLO for small crowds ─
    sparse_mask = bucket_arr == 0
    if sparse_mask.sum() > 0:
        print("\n" + "=" * 70)
        print("  SPARSE BUCKET DEEP-DIVE (proving Hybrid ≈ YOLO for small crowds)")
        print("=" * 70)
        g = gt_arr[sparse_mask]
        y = yolo_arr[sparse_mask]
        v = vgg_arr[sparse_mask]
        h = hybrid_arr[sparse_mask]
        print(f"  N = {len(g)} images  |  GT range: {int(g.min())}–{int(g.max())}")
        print(f"  MAE   YOLO: {mae(y,g):.2f}   VGG: {mae(v,g):.2f}   Hybrid: {mae(h,g):.2f}")
        print(f"  RMSE  YOLO: {rmse(y,g):.2f}   VGG: {rmse(v,g):.2f}   Hybrid: {rmse(h,g):.2f}")
        print(f"  MAPE  YOLO: {mape(y,g):.1f}%  VGG: {mape(v,g):.1f}%  Hybrid: {mape(h,g):.1f}%")

        # Paired improvement test
        yolo_err = np.abs(y - g)
        hybrid_err = np.abs(h - g)
        vgg_err = np.abs(v - g)
        improvement = float(np.mean(vgg_err - hybrid_err))
        print(f"\n  Mean(|VGG error| − |Hybrid error|) = {improvement:+.2f}")
        if improvement > 0:
            print("  → Hybrid IMPROVES over VGG in sparse scenes by "
                  f"{improvement:.2f} avg count ✓")
        else:
            print("  → Hybrid does NOT dominate VGG here; consider tuning thresholds.")

        # Wilcoxon signed-rank test (non-parametric paired test)
        try:
            from scipy.stats import wilcoxon
            stat, p_val = wilcoxon(vgg_err, hybrid_err, alternative="greater")
            print(f"  Wilcoxon signed-rank test: stat={stat:.1f}, p={p_val:.4f}")
            if p_val < 0.05:
                print("  → Statistically significant improvement (p < 0.05) ✓")
            else:
                print("  → Not statistically significant at α=0.05")
        except Exception as e:
            print(f"  Wilcoxon test skipped: {e}")

    print("\n  Done. All results saved to results/")


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_evaluation()
