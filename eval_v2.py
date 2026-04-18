"""
eval_v2.py — V2 Evaluation (Step 11 — Enhanced)
=================================================
Metrics: MAE, RMSE, Bias, Accuracy %
Buckets: 1–20, 21–50, 51–100, 100+
Includes: Raw vs Cal comparison, per-dataset breakdown, worst-case debug

Evaluates on ShanghaiTech Part A + B test sets + JHU test.

Usage:
    python3 eval_v2.py
"""

import os
import json
import cv2
import numpy as np
import scipy.io as sio
from tabulate import tabulate
from inference_v2 import load_model_v2, tiled_inference
from calibrate_v2 import apply_calibration_v2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════

TEST_A_IMG = "data/part_A_final/test_data/images"
TEST_A_GT  = "data/part_A_final/test_data/ground_truth"
TEST_B_IMG = "data/part_B_final/test_data/images"
TEST_B_GT  = "data/part_B_final/test_data/ground_truth"
JHU_TEST   = "jhu_crowd_v2.0/test"

CALIB_PATH = "models/calibration_v2.json"
os.makedirs("results", exist_ok=True)

# Buckets
BUCKETS = [
    ("1–20",   1,   20),
    ("21–50",  21,  50),
    ("51–100", 51,  100),
    ("100+",   101, 999999),
]


# ═══════════════════════════════════════════════════════════════
# GT LOADERS
# ═══════════════════════════════════════════════════════════════

def get_gt_shanghai(img_name, gt_dir):
    mat_name = "GT_" + img_name.replace(".jpg", ".mat")
    mat_path = os.path.join(gt_dir, mat_name)
    if not os.path.exists(mat_path):
        return None
    mat = sio.loadmat(mat_path)
    return len(mat["image_info"][0, 0][0, 0][0])


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate():
    print("=" * 70)
    print("  SafeCrowd V2 — Evaluation Report")
    print("=" * 70)

    model = load_model_v2()
    print("  ✓ Model loaded")

    # Load calibration
    calib = None
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH) as f:
            calib = json.load(f)
        print("  ✓ Calibration loaded")
    else:
        print("  ⚠ No calibration found — using raw counts")

    # ── Collect predictions ──
    records = []  # (gt, raw_pred, cal_pred, dataset)

    # Part B
    if os.path.isdir(TEST_B_IMG):
        files = sorted(f for f in os.listdir(TEST_B_IMG)
                       if f.endswith(".jpg"))
        print(f"\n  Part B: {len(files)} images ...")
        for i, fname in enumerate(files):
            gt = get_gt_shanghai(fname, TEST_B_GT)
            if gt is None or gt == 0:
                continue
            img = cv2.imread(os.path.join(TEST_B_IMG, fname))
            raw, _ = tiled_inference(model, img, tta=True)
            cal = apply_calibration_v2(raw, calib) if calib else int(round(raw))
            records.append((gt, raw, cal, "PartB"))
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(files)}")

    # Part A
    if os.path.isdir(TEST_A_IMG):
        files = sorted(f for f in os.listdir(TEST_A_IMG)
                       if f.endswith(".jpg"))
        print(f"\n  Part A: {len(files)} images ...")
        for i, fname in enumerate(files):
            gt = get_gt_shanghai(fname, TEST_A_GT)
            if gt is None or gt == 0:
                continue
            img = cv2.imread(os.path.join(TEST_A_IMG, fname))
            raw, _ = tiled_inference(model, img, tta=True)
            cal = apply_calibration_v2(raw, calib) if calib else int(round(raw))
            records.append((gt, raw, cal, "PartA"))
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(files)}")

    # JHU test (≤100 only for sparse focus)
    if os.path.isdir(JHU_TEST):
        labels_path = os.path.join(JHU_TEST, "image_labels.txt")
        print(f"\n  JHU test (≤100) ...")
        count = 0
        with open(labels_path) as f:
            for line in f:
                parts = line.strip().split(",")
                img_id, gt = parts[0], int(parts[1])
                if gt <= 100 and gt > 0:
                    img_path = os.path.join(JHU_TEST, "images",
                                            f"{img_id}.jpg")
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        if img is not None:
                            raw, _ = tiled_inference(model, img, tta=True)
                            cal = apply_calibration_v2(raw, calib) if calib else int(round(raw))
                            records.append((gt, raw, cal, "JHU"))
                            count += 1
                            if count % 50 == 0:
                                print(f"    {count} JHU images")
                if count >= 300:
                    break

    if not records:
        print("  ERROR: No images evaluated.")
        return

    print(f"\n  Total images evaluated: {len(records)}")

    # ── Compute metrics ──
    gt_all   = np.array([r[0] for r in records], dtype=float)
    raw_all  = np.array([r[1] for r in records], dtype=float)
    cal_all  = np.array([r[2] for r in records], dtype=float)

    def mae(p, g):
        return float(np.mean(np.abs(p - g)))

    def rmse(p, g):
        return float(np.sqrt(np.mean((p - g) ** 2)))

    def bias(p, g):
        return float(np.mean(p - g))

    def accuracy(p, g):
        mean_gt = np.mean(g)
        if mean_gt < 1e-6:
            return 0.0
        return max(0.0, 100.0 - (mae(p, g) / mean_gt) * 100.0)

    # ── Bucket-wise table ──
    print("\n")
    rows = []
    bucket_data = {}  # for plotting

    for bname, bmin, bmax in BUCKETS:
        mask = (gt_all >= bmin) & (gt_all <= bmax)
        n = int(mask.sum())
        if n == 0:
            rows.append([bname, 0, "—", "—", "—", "—"])
            continue
        g = gt_all[mask]
        r = raw_all[mask]
        c = cal_all[mask]
        rows.append([
            bname, n,
            f"{mae(r, g):.2f}", f"{rmse(r, g):.2f}",
            f"{mae(c, g):.2f}", f"{rmse(c, g):.2f}",
        ])
        bucket_data[bname] = {
            "n": n,
            "raw_mae": mae(r, g), "raw_rmse": rmse(r, g),
            "cal_mae": mae(c, g), "cal_rmse": rmse(c, g),
            "gt": g, "raw": r, "cal": c,
        }

    # Overall
    rows.append([
        "OVERALL", len(gt_all),
        f"{mae(raw_all, gt_all):.2f}", f"{rmse(raw_all, gt_all):.2f}",
        f"{mae(cal_all, gt_all):.2f}", f"{rmse(cal_all, gt_all):.2f}",
    ])

    headers = ["Bucket", "N",
               "Raw MAE", "Raw RMSE",
               "Cal MAE", "Cal RMSE"]

    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # ══════════════════════════════════════════════════════════
    # STEP 1: RAW vs CALIBRATED GLOBAL COMPARISON
    # ══════════════════════════════════════════════════════════

    raw_mae_global = mae(raw_all, gt_all)
    cal_mae_global = mae(cal_all, gt_all)
    diff = raw_mae_global - cal_mae_global
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  RAW vs CALIBRATED (Global)                 │")
    print(f"  ├─────────────────────────────────────────────┤")
    print(f"  │  Raw MAE:        {raw_mae_global:>10.2f}                │")
    print(f"  │  Calibrated MAE: {cal_mae_global:>10.2f}                │")
    print(f"  │  Improvement:    {diff:>+10.2f}  ", end="")
    if diff > 0:
        print(f"(calibration helps) ✓  │")
    else:
        print(f"(model is better raw)  │")
    print(f"  └─────────────────────────────────────────────┘")

    # ══════════════════════════════════════════════════════════
    # STEP 2: BIAS DETECTION
    # ══════════════════════════════════════════════════════════

    global_bias = bias(cal_all, gt_all)
    bias_dir = "OVERCOUNTING ↑" if global_bias > 0 else "UNDERCOUNTING ↓"
    print(f"\n  BIAS: {global_bias:+.2f}  →  {bias_dir}")

    # Per-bucket bias
    for bname, bmin, bmax in BUCKETS:
        mask = (gt_all >= bmin) & (gt_all <= bmax)
        if mask.sum() > 0:
            b_val = bias(cal_all[mask], gt_all[mask])
            arrow = "↑" if b_val > 0 else "↓"
            print(f"    {bname:>8s}: bias = {b_val:+.2f}  {arrow}")

    # ══════════════════════════════════════════════════════════
    # STEP 3: ACCURACY METRIC
    # ══════════════════════════════════════════════════════════

    acc_global = accuracy(cal_all, gt_all)
    print(f"\n  ACCURACY: {acc_global:.1f}%  "
          f"(100 − MAE/mean_GT × 100)")

    for bname, bmin, bmax in BUCKETS:
        mask = (gt_all >= bmin) & (gt_all <= bmax)
        if mask.sum() > 0:
            a = accuracy(cal_all[mask], gt_all[mask])
            print(f"    {bname:>8s}: {a:.1f}%")

    # ══════════════════════════════════════════════════════════
    # STEP 5: PER-DATASET RAW vs CAL COMPARISON
    # ══════════════════════════════════════════════════════════

    print("\n  Per-Dataset Breakdown (Raw vs Calibrated):")
    print(f"    {'Dataset':>6s}  {'N':>5s}  {'Raw MAE':>9s}  {'Cal MAE':>9s}  "
          f"{'Δ':>7s}  {'Bias':>7s}  {'Acc%':>6s}  {'GT Range':>12s}")
    print(f"    {'-'*72}")
    for ds in ["PartB", "PartA", "JHU"]:
        ds_records = [(g, r, c) for g, r, c, d in records if d == ds]
        if not ds_records:
            continue
        g = np.array([x[0] for x in ds_records])
        r = np.array([x[1] for x in ds_records])
        c = np.array([x[2] for x in ds_records])
        d_val = mae(r, g) - mae(c, g)
        print(f"    {ds:>6s}  {len(g):5d}  {mae(r,g):9.2f}  {mae(c,g):9.2f}  "
              f"{d_val:+7.2f}  {bias(c,g):+7.2f}  {accuracy(c,g):5.1f}%  "
              f"{int(g.min()):>5d}–{int(g.max()):<5d}")

    # ── Sparse focus (1-100) ──
    sparse_mask = gt_all <= 100
    if sparse_mask.sum() > 0:
        g_s = gt_all[sparse_mask]
        r_s = raw_all[sparse_mask]
        c_s = cal_all[sparse_mask]
        print(f"\n  ★ SPARSE FOCUS (1–100): N={len(g_s)}")
        print(f"    Raw MAE:  {mae(r_s, g_s):.2f}")
        print(f"    Cal MAE:  {mae(c_s, g_s):.2f}")
        print(f"    RMSE:     {rmse(c_s, g_s):.2f}")
        print(f"    Bias:     {bias(c_s, g_s):+.2f}")
        print(f"    Accuracy: {accuracy(c_s, g_s):.1f}%")

    # ══════════════════════════════════════════════════════════
    # STEP 6: TOP 10 WORST PREDICTIONS
    # ══════════════════════════════════════════════════════════

    abs_errors = np.abs(cal_all - gt_all)
    worst_idx = np.argsort(abs_errors)[::-1][:10]

    print("\n  TOP 10 WORST PREDICTIONS:")
    print(f"    {'#':>3s}  {'GT':>6s}  {'Raw':>10s}  {'Cal':>6s}  "
          f"{'|Error|':>8s}  {'Dataset':>7s}")
    print(f"    {'-'*52}")
    for rank, idx in enumerate(worst_idx, 1):
        gt_v = int(gt_all[idx])
        raw_v = raw_all[idx]
        cal_v = int(cal_all[idx])
        err_v = int(abs_errors[idx])
        ds_v = records[idx][3]
        print(f"    {rank:3d}  {gt_v:6d}  {raw_v:10.1f}  {cal_v:6d}  "
              f"{err_v:8d}  {ds_v:>7s}")

    # ═══════════════════════════════════════════════════════════
    # VISUALIZATIONS
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

    # ── 1. MAE by bucket bar chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    bnames = [b for b in bucket_data.keys()]
    raw_maes = [bucket_data[b]["raw_mae"] for b in bnames]
    cal_maes = [bucket_data[b]["cal_mae"] for b in bnames]
    x = np.arange(len(bnames))
    w = 0.3

    bars1 = ax.bar(x - w/2, raw_maes, w, label="Raw (no calib)",
                   color="#F5A623", edgecolor="#30363D")
    bars2 = ax.bar(x + w/2, cal_maes, w, label="Calibrated",
                   color="#3FB950", edgecolor="#30363D")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom",
                    fontsize=9, color="#C9D1D9", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(bnames)
    ax.set_ylabel("MAE ↓")
    ax.set_title("V2 MAE by Density Bucket", fontweight="bold", fontsize=14)
    ax.legend(framealpha=0.3, edgecolor="#30363D")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/eval_v2_mae_buckets.png", dpi=200)
    print("\n  ✓ results/eval_v2_mae_buckets.png")
    plt.close(fig)

    # ── 2. Pred vs GT scatter ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Full range
    ax1.scatter(gt_all, cal_all, c="#3FB950", alpha=0.4, s=12,
                edgecolors="none")
    lim = max(gt_all.max(), cal_all.max()) * 1.05
    ax1.plot([0, lim], [0, lim], "--", color="#DA3633", linewidth=1)
    ax1.set_xlabel("Ground Truth")
    ax1.set_ylabel("Predicted (calibrated)")
    ax1.set_title("All Images", fontweight="bold")
    ax1.grid(alpha=0.2)

    # Sparse zoom (0–100)
    s_mask = gt_all <= 100
    if s_mask.sum() > 0:
        ax2.scatter(gt_all[s_mask], cal_all[s_mask], c="#58A6FF",
                    alpha=0.5, s=18, edgecolors="none")
        ax2.plot([0, 110], [0, 110], "--", color="#DA3633", linewidth=1)
        ax2.set_xlabel("Ground Truth")
        ax2.set_ylabel("Predicted (calibrated)")
        ax2.set_title("Sparse Focus (1–100)", fontweight="bold")
        ax2.set_xlim(-5, 110)
        ax2.set_ylim(-5, 150)
        ax2.grid(alpha=0.2)

    fig.suptitle("V2 Predicted vs Ground Truth",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("results/eval_v2_scatter.png", dpi=200)
    print("  ✓ results/eval_v2_scatter.png")
    plt.close(fig)

    # ── 3. Error distribution (sparse) ──
    if sparse_mask.sum() > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        errors = cal_all[sparse_mask] - gt_all[sparse_mask]
        ax.hist(errors, bins=40, color="#58A6FF", alpha=0.7,
                edgecolor="#30363D")
        ax.axvline(0, color="#DA3633", linestyle="--", linewidth=1.5)
        ax.axvline(np.mean(errors), color="#3FB950", linestyle="-",
                   linewidth=1.5, label=f"Mean bias: {np.mean(errors):.1f}")
        ax.set_xlabel("Prediction Error (pred − GT)")
        ax.set_ylabel("Count")
        ax.set_title("Error Distribution (Sparse 1–100)",
                     fontweight="bold", fontsize=14)
        ax.legend(framealpha=0.3, edgecolor="#30363D")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig("results/eval_v2_error_hist.png", dpi=200)
        print("  ✓ results/eval_v2_error_hist.png")
        plt.close(fig)

    # ── Save CSV ──
    csv_path = "results/eval_v2_detail.csv"
    with open(csv_path, "w") as f:
        f.write("gt,raw_pred,cal_pred,dataset\n")
        for gt, raw, cal, ds in records:
            f.write(f"{gt},{raw:.4f},{cal},{ds}\n")
    print(f"  ✓ {csv_path}")

    print("\n" + "=" * 70)
    print("  Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
