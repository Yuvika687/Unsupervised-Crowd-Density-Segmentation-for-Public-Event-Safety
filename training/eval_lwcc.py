"""
eval_lwcc.py — Evaluate LWCC DM-Count Accuracy
================================================
Compares LWCC DM-Count (SHB/SHA scene-adaptive) against ground truth
on ShanghaiTech Part B test set (sparse focus).

Usage:
    python3 eval_lwcc.py
"""

import os
import sys
import numpy as np
import scipy.io as sio
import cv2
from tabulate import tabulate

# Monkey-patch LWCC's broken weights path (uses /.lwcc instead of ~/.lwcc)
def _patch_lwcc_weights_path():
    """Fix LWCC bug: weights_check uses root / instead of ~/."""
    try:
        import lwcc.util.functions as lwcc_funcs
        from pathlib import Path
        import gdown

        def _fixed_weights_check(model_name, model_weights):
            home = str(Path.home())
            weights_dir = os.path.join(home, ".lwcc", "weights")
            os.makedirs(weights_dir, exist_ok=True)

            file_name = f"{model_name}_{model_weights}.pth"
            url = lwcc_funcs.build_url(file_name)
            output = os.path.join(weights_dir, file_name)

            if not os.path.isfile(output):
                print(f"{file_name} will be downloaded to {output}")
                gdown.download(url, output, quiet=False)

            return output

        lwcc_funcs.weights_check = _fixed_weights_check
    except Exception:
        pass

_patch_lwcc_weights_path()

try:
    from lwcc import LWCC
except ImportError:
    print("ERROR: lwcc not installed. Run: pip3 install lwcc")
    sys.exit(1)

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

SCENE_THRESHOLD = 80  # SHB for <80, SHA for >=80

os.makedirs("results", exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# GT LOADER
# ═══════════════════════════════════════════════════════════════

def get_gt_shanghai(img_name, gt_dir):
    mat_name = "GT_" + img_name.replace(".jpg", ".mat")
    mat_path = os.path.join(gt_dir, mat_name)
    if not os.path.exists(mat_path):
        return None
    mat = sio.loadmat(mat_path)
    return len(mat["image_info"][0, 0][0, 0][0])


# ═══════════════════════════════════════════════════════════════
# SCENE-ADAPTIVE PREDICTION
# ═══════════════════════════════════════════════════════════════

def predict_adaptive(img_path, model_shb, model_sha):
    """
    Scene-adaptive counting:
    1. Run SHB first (sparse-optimized)
    2. If count >= threshold, re-run with SHA (dense-optimized)
    """
    count_shb, density_shb = LWCC.get_count(
        img_path, model_name="DM-Count", model_weights="SHB",
        model=model_shb, return_density=True, resize_img=True)

    if count_shb >= SCENE_THRESHOLD and model_sha is not None:
        count_sha, density_sha = LWCC.get_count(
            img_path, model_name="DM-Count", model_weights="SHA",
            model=model_sha, return_density=True, resize_img=False)
        return float(count_sha), density_sha, "SHA"
    else:
        return float(count_shb), density_shb, "SHB"


# NOTE: apply_calibration is NOT used for the LWCC path.
# DM-Count's raw output IS the system output — no calibration needed.
# The old calibration files (calibration_*.json) were fitted for the
# legacy VGG-16 model and would DEGRADE DM-Count accuracy.


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate():
    print("=" * 70)
    print("  LWCC DM-Count — Evaluation Report")
    print("=" * 70)

    # Load models
    print("  Loading DM-Count models...")
    model_shb = LWCC.load_model(model_name="DM-Count", model_weights="SHB")
    model_sha = LWCC.load_model(model_name="DM-Count", model_weights="SHA")
    print("  ✓ SHB and SHA models loaded\n")

    records = []  # (gt, pred, dataset, weights_used)

    BUCKETS = [
        ("1–20",   1,   20),
        ("21–50",  21,  50),
        ("51–100", 51,  100),
        ("100+",   101, 999999),
    ]

    # Part B (sparse)
    if os.path.isdir(TEST_B_IMG):
        files = sorted(f for f in os.listdir(TEST_B_IMG) if f.endswith(".jpg"))
        print(f"  Part B: {len(files)} images ...")
        for i, fname in enumerate(files):
            gt = get_gt_shanghai(fname, TEST_B_GT)
            if gt is None or gt == 0:
                continue
            img_path = os.path.join(TEST_B_IMG, fname)
            raw, density, weights = predict_adaptive(img_path, model_shb, model_sha)
            records.append((gt, raw, "PartB", weights))
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(files)}  "
                      f"(last: GT={gt}, Pred={raw:.1f}, {weights})")

    # Part A (dense)
    if os.path.isdir(TEST_A_IMG):
        files = sorted(f for f in os.listdir(TEST_A_IMG) if f.endswith(".jpg"))
        print(f"\n  Part A: {len(files)} images ...")
        for i, fname in enumerate(files):
            gt = get_gt_shanghai(fname, TEST_A_GT)
            if gt is None or gt == 0:
                continue
            img_path = os.path.join(TEST_A_IMG, fname)
            raw, density, weights = predict_adaptive(img_path, model_shb, model_sha)
            records.append((gt, raw, "PartA", weights))
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(files)}  "
                      f"(last: GT={gt}, Pred={raw:.1f}, {weights})")

    if not records:
        print("  ERROR: No images evaluated. Check data paths.")
        return

    print(f"\n  Total images evaluated: {len(records)}")

    # ── Compute metrics ──
    gt_all   = np.array([r[0] for r in records], dtype=float)
    pred_all = np.array([r[1] for r in records], dtype=float)

    def mae(p, g):
        return float(np.mean(np.abs(p - g)))

    def rmse(p, g):
        return float(np.sqrt(np.mean((p - g) ** 2)))

    def bias(p, g):
        return float(np.mean(p - g))

    def accuracy(p, g):
        mean_gt = np.mean(g)
        if mean_gt < 1e-6: return 0.0
        return max(0.0, 100.0 - (mae(p, g) / mean_gt) * 100.0)

    # ── Bucket-wise table ──
    print("\n")
    rows = []
    for bname, bmin, bmax in BUCKETS:
        mask = (gt_all >= bmin) & (gt_all <= bmax)
        n = int(mask.sum())
        if n == 0:
            rows.append([bname, 0, "—", "—", "—", "—"])
            continue
        g = gt_all[mask]
        p = pred_all[mask]
        rows.append([
            bname, n,
            f"{mae(p, g):.2f}", f"{rmse(p, g):.2f}",
            f"{bias(p, g):+.2f}", f"{accuracy(p, g):.1f}%",
        ])

    # Overall
    rows.append([
        "OVERALL", len(gt_all),
        f"{mae(pred_all, gt_all):.2f}",
        f"{rmse(pred_all, gt_all):.2f}",
        f"{bias(pred_all, gt_all):+.2f}",
        f"{accuracy(pred_all, gt_all):.1f}%",
    ])

    headers = ["Bucket", "N", "MAE", "RMSE", "Bias", "Accuracy"]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # ── Per-dataset breakdown ──
    print("\n  Per-Dataset Breakdown:")
    print(f"    {'Dataset':>6s}  {'N':>5s}  {'MAE':>9s}  "
          f"{'RMSE':>9s}  {'Bias':>7s}  {'Acc%':>6s}")
    print(f"    {'-'*50}")
    for ds in ["PartB", "PartA"]:
        ds_recs = [(g, p) for g, p, d, _ in records if d == ds]
        if not ds_recs:
            continue
        g = np.array([x[0] for x in ds_recs])
        p = np.array([x[1] for x in ds_recs])
        print(f"    {ds:>6s}  {len(g):5d}  {mae(p,g):9.2f}  "
              f"{rmse(p,g):9.2f}  {bias(p,g):+7.2f}  "
              f"{accuracy(p,g):5.1f}%")

    # ── Sparse focus (1-100) ──
    sparse_mask = gt_all <= 100
    if sparse_mask.sum() > 0:
        g_s = gt_all[sparse_mask]
        p_s = pred_all[sparse_mask]
        print(f"\n  ★ SPARSE FOCUS (1–100): N={len(g_s)}")
        print(f"    MAE:      {mae(p_s, g_s):.2f}")
        print(f"    RMSE:     {rmse(p_s, g_s):.2f}")
        print(f"    Bias:     {bias(p_s, g_s):+.2f}")
        print(f"    Accuracy: {accuracy(p_s, g_s):.1f}%")

    # ── Scene-adaptive stats ──
    shb_used = sum(1 for _, _, _, w in records if w == "SHB")
    sha_used = sum(1 for _, _, _, w in records if w == "SHA")
    print(f"\n  Scene-Adaptive Stats:")
    print(f"    SHB (sparse) used: {shb_used} images")
    print(f"    SHA (dense)  used: {sha_used} images")

    # ── Top 10 worst predictions ──
    abs_errors = np.abs(pred_all - gt_all)
    worst_idx = np.argsort(abs_errors)[::-1][:10]

    print("\n  TOP 10 WORST PREDICTIONS:")
    print(f"    {'#':>3s}  {'GT':>6s}  {'Pred':>8s}  "
          f"{'|Error|':>8s}  {'Dataset':>7s}  {'Weights':>7s}")
    print(f"    {'-'*48}")
    for rank, idx in enumerate(worst_idx, 1):
        gt_v = int(gt_all[idx])
        pred_v = pred_all[idx]
        err_v = int(abs_errors[idx])
        ds_v = records[idx][2]
        wt_v = records[idx][3]
        print(f"    {rank:3d}  {gt_v:6d}  {pred_v:8.1f}  "
              f"{err_v:8d}  {ds_v:>7s}  {wt_v:>7s}")

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

    # ── Pred vs GT scatter ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Full range
    ax1.scatter(gt_all, pred_all, c="#3FB950", alpha=0.4, s=12,
                edgecolors="none")
    lim = max(gt_all.max(), pred_all.max()) * 1.05
    ax1.plot([0, lim], [0, lim], "--", color="#DA3633", linewidth=1)
    ax1.set_xlabel("Ground Truth")
    ax1.set_ylabel("Predicted (DM-Count)")
    ax1.set_title("All Images", fontweight="bold")
    ax1.grid(alpha=0.2)

    # Sparse zoom (0–100)
    if sparse_mask.sum() > 0:
        ax2.scatter(gt_all[sparse_mask], pred_all[sparse_mask], c="#58A6FF",
                    alpha=0.5, s=18, edgecolors="none")
        ax2.plot([0, 110], [0, 110], "--", color="#DA3633", linewidth=1)
        ax2.set_xlabel("Ground Truth")
        ax2.set_ylabel("Predicted (DM-Count)")
        ax2.set_title("Sparse Focus (1–100)", fontweight="bold")
        ax2.set_xlim(-5, 110)
        ax2.set_ylim(-5, 150)
        ax2.grid(alpha=0.2)

    fig.suptitle("LWCC DM-Count: Predicted vs Ground Truth",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("results/eval_lwcc_scatter.png", dpi=200)
    print("\n  ✓ results/eval_lwcc_scatter.png")
    plt.close(fig)

    # ── Error distribution (sparse) ──
    if sparse_mask.sum() > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        errors = pred_all[sparse_mask] - gt_all[sparse_mask]
        ax.hist(errors, bins=40, color="#58A6FF", alpha=0.7,
                edgecolor="#30363D")
        ax.axvline(0, color="#DA3633", linestyle="--", linewidth=1.5)
        ax.axvline(np.mean(errors), color="#3FB950", linestyle="-",
                   linewidth=1.5, label=f"Mean bias: {np.mean(errors):.1f}")
        ax.set_xlabel("Prediction Error (pred − GT)")
        ax.set_ylabel("Count")
        ax.set_title("Error Distribution — Sparse (1–100)",
                     fontweight="bold", fontsize=14)
        ax.legend(framealpha=0.3, edgecolor="#30363D")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig("results/eval_lwcc_error_hist.png", dpi=200)
        print("  ✓ results/eval_lwcc_error_hist.png")
        plt.close(fig)

    # ── Save CSV ──
    csv_path = "results/eval_lwcc_detail.csv"
    with open(csv_path, "w") as f:
        f.write("gt,pred,dataset,weights\n")
        for gt, pred, ds, wt in records:
            f.write(f"{gt},{pred:.4f},{ds},{wt}\n")
    print(f"  ✓ {csv_path}")

    print("\n" + "=" * 70)
    print("  Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()



