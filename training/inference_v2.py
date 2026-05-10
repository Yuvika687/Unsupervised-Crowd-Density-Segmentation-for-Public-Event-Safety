"""
inference_v2.py — V2 Inference with TTA (Step 9)
==================================================
- Removed 0.015 threshold, uses 1e-4
- Test-Time Augmentation (horizontal flip)
- Tiled inference with 50% overlap
- Dual-head fused count
"""

import os
import sys

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.model_v2 import CrowdDensityNetV2

IMG_SIZE = 256


def _tile_start_positions(length, tile, stride):
    """Generate deterministic tile starts with full border coverage."""
    if length <= tile:
        return [0]
    starts = list(range(0, max(length - tile, 0) + 1, stride))
    last = max(length - tile, 0)
    if starts[-1] != last:
        starts.append(last)
    return starts


def tiled_inference(model, img_bgr, device="cpu", tta=True,
                    return_components=False):
    """
    Step 9: Tiled density inference + TTA.

    Returns:
        count: overlap-aware fused count
        density_map: full-resolution density (un-calibrated)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape

    def _run_tiles(img_2d):
        """Run tiled inference on a 2D grayscale image."""
        ih, iw = img_2d.shape
        tile, stride = IMG_SIZE, IMG_SIZE // 2

        pad_h = (stride - (ih % stride)) % stride
        pad_w = (stride - (iw % stride)) % stride
        padded = np.pad(img_2d, ((0, pad_h), (0, pad_w)), mode="reflect")

        den_acc = np.zeros_like(padded)
        cnt_acc = np.zeros_like(padded)
        tile_boxes = []
        tile_counts = []

        y_starts = _tile_start_positions(padded.shape[0], tile, stride)
        x_starts = _tile_start_positions(padded.shape[1], tile, stride)

        for y in y_starts:
            for x in x_starts:
                t = padded[y:y+tile, x:x+tile]
                if t.shape != (tile, tile):
                    t = cv2.resize(t, (tile, tile))

                t_tensor = torch.tensor(
                    np.stack([t]*3, 0), dtype=torch.float32
                ).unsqueeze(0).to(device)

                with torch.no_grad():
                    density, count = model(t_tensor)

                den_acc[y:y+tile, x:x+tile] += density.squeeze().cpu().numpy()
                cnt_acc[y:y+tile, x:x+tile] += 1.0
                tile_boxes.append((y, x))
                tile_counts.append(float(count.item()))

        density_map = (den_acc / np.maximum(cnt_acc, 1.0))[:ih, :iw]
        density_map = np.clip(density_map, 0, None)

        # Step 9: soft threshold at 1e-4 instead of 0.015
        density_map[density_map < 1e-4] = 0

        if not tile_counts:
            return density_map, 0.0, {
                "tile_count": 0,
                "reg_weighted": 0.0,
                "reg_avg": 0.0,
                "reg_max": 0.0,
            }

        # Convert per-tile crop counts into a full-image estimate by weighting
        # each tile by the average reciprocal overlap over its covered area.
        inv_cov = np.divide(
            1.0,
            cnt_acc,
            out=np.zeros_like(cnt_acc),
            where=cnt_acc > 0,
        )
        reg_weighted = 0.0
        for (y, x), tile_count in zip(tile_boxes, tile_counts):
            tile_weight = float(inv_cov[y:y+tile, x:x+tile].mean())
            reg_weighted += tile_count * tile_weight

        reg_meta = {
            "tile_count": len(tile_counts),
            "reg_weighted": float(reg_weighted),
            "reg_avg": float(np.mean(tile_counts)),
            "reg_max": float(np.max(tile_counts)),
        }
        return density_map, float(reg_weighted), reg_meta

    model.eval()

    # ── Forward pass ──
    den1, reg1, meta1 = _run_tiles(gray)

    if tta:
        # ── TTA: horizontal flip ──
        gray_flip = np.flip(gray, axis=1).copy()
        den2, reg2, meta2 = _run_tiles(gray_flip)
        # Flip density back
        den2 = np.flip(den2, axis=1).copy()
        # Average
        density_map = (den1 + den2) / 2.0
        reg_count = (reg1 + reg2) / 2.0
        tile_meta = {
            "tile_count": max(meta1["tile_count"], meta2["tile_count"]),
            "reg_weighted": float(reg_count),
            "reg_avg": float((meta1["reg_avg"] + meta2["reg_avg"]) / 2.0),
            "reg_max": float(max(meta1["reg_max"], meta2["reg_max"])),
        }
    else:
        density_map = den1
        reg_count = reg1
        tile_meta = meta1

    density_sum_raw = float(density_map.sum())
    if reg_count <= 1e-6:
        fused_count = density_sum_raw
    else:
        density_ratio = density_sum_raw / max(reg_count, 1e-6)
        if density_ratio < 0.15 or density_ratio > 3.5:
            # Density branch has collapsed or is badly scaled.
            fused_count = reg_count
        elif reg_count < 25:
            fused_count = 0.15 * density_sum_raw + 0.85 * reg_count
        else:
            fused_count = 0.30 * density_sum_raw + 0.70 * reg_count

    if density_sum_raw > 1e-6 and fused_count > 0:
        density_map = density_map * (fused_count / density_sum_raw)

    debug = {
        "density_sum_raw": float(density_sum_raw),
        "reg_weighted": float(reg_count),
        "density_ratio": (
            float(density_sum_raw / max(reg_count, 1e-6))
            if reg_count > 1e-6 else 0.0
        ),
        **tile_meta,
    }

    if return_components:
        return float(fused_count), density_map, debug
    return float(fused_count), density_map


def load_model_v2(path="models/best_model_v2.pth", device="cpu"):
    """Load V2 model."""
    model = CrowdDensityNetV2(pretrained=False)
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model.to(device)


# ═══════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    model = load_model_v2()
    test_img = "data/part_B_final/test_data/images/IMG_1.jpg"
    if os.path.exists(test_img):
        img = cv2.imread(test_img)
        count, density = tiled_inference(model, img, tta=True)
        print(f"Predicted count (fused, TTA): {count:.1f}")
        print(f"Density sum: {density.sum():.1f}")
    else:
        print("Test image not found")
