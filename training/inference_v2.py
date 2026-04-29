"""
inference_v2.py — V2 Inference with TTA (Step 9)
==================================================
- Removed 0.015 threshold, uses 1e-4
- Test-Time Augmentation (horizontal flip)
- Tiled inference with 50% overlap
- Dual-head fused count
"""

import cv2
import torch
import numpy as np
from model_v2 import CrowdDensityNetV2

IMG_SIZE = 256


def tiled_inference(model, img_bgr, device="cpu", tta=True):
    """
    Step 9: Tiled density inference + TTA.

    Returns:
        count: fused count (0.7*density + 0.3*regressor)
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
        reg_sum = 0.0
        n_tiles = 0

        for y in range(0, padded.shape[0] - stride, stride):
            for x in range(0, padded.shape[1] - stride, stride):
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
                reg_sum += count.item()
                n_tiles += 1

        density_map = (den_acc / np.maximum(cnt_acc, 1.0))[:ih, :iw]
        density_map = np.clip(density_map, 0, None)

        # Step 9: soft threshold at 1e-4 instead of 0.015
        density_map[density_map < 1e-4] = 0

        avg_reg = reg_sum / max(n_tiles, 1)
        return density_map, avg_reg

    model.eval()

    # ── Forward pass ──
    den1, reg1 = _run_tiles(gray)

    if tta:
        # ── TTA: horizontal flip ──
        gray_flip = np.flip(gray, axis=1).copy()
        den2, reg2 = _run_tiles(gray_flip)
        # Flip density back
        den2 = np.flip(den2, axis=1).copy()
        # Average
        density_map = (den1 + den2) / 2.0
        reg_count = (reg1 + reg2) / 2.0
    else:
        density_map = den1
        reg_count = reg1

    # Fused count: 0.7 * density_sum + 0.3 * regressor
    density_sum = float(density_map.sum())
    fused_count = 0.7 * density_sum + 0.3 * reg_count

    return fused_count, density_map


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
