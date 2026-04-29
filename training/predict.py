"""
predict.py — Run crowd density estimation on a single image
============================================================
Loads the VGG-16 based CrowdDensityNet from models/best_model.pth,
predicts a 256×256 density map, and displays the result.
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import models


# ─── Model (must match train.py) ─────────────────────────────────────────────

class CrowdDensityNet(nn.Module):
    """VGG-16 encoder → transposed-conv decoder for density estimation."""

    def __init__(self, freeze_layers=10):
        super(CrowdDensityNet, self).__init__()

        vgg = models.vgg16(weights=None)  # weights loaded from checkpoint
        self.encoder = nn.Sequential(*list(vgg.features.children())[:24])

        for i, param in enumerate(self.encoder.parameters()):
            if i < freeze_layers:
                param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ─── Load model ──────────────────────────────────────────────────────────────

MODEL_PATH = "models/best_model.pth"

model = CrowdDensityNet()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=True)
)
model.eval()
print(f"Model loaded from {MODEL_PATH}")


# ─── Load & preprocess image ─────────────────────────────────────────────────

IMG_PATH = "data/part_A_final/test_data/images/IMG_1.jpg"

img_bgr = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Grayscale → resize → normalize → replicate to 3ch
def predict_tiled(model, img_bgr):
    """Processes high-res image in 256x256 tiles with 50% overlap."""
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = img_gray.shape
    tile_size = 256
    stride = 128
    
    # Pad to fit tiles
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
            
            # Use linear blending for overlapping regions
            density_p[y:y+tile_size, x:x+tile_size] += out.squeeze().numpy()
            count_p[y:y+tile_size, x:x+tile_size]   += 1.0
            
    density = (density_p / np.maximum(count_p, 1.0))[:h, :w]
    return np.clip(density, 0, None)

# ─── Load & Process ───
IMG_PATH = "data/part_A_final/test_data/images/IMG_1.jpg"
img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    print(f"Error: Could not load {IMG_PATH}")
    exit()

density = predict_tiled(model, img_bgr)

# ─── Calibration ───
import json
CALIB_PATH = "models/calibration.json"
if os.path.exists(CALIB_PATH):
    with open(CALIB_PATH, "r") as f:
        calib = json.load(f)
        a, b = calib.get("a", 0.0), calib.get("b", 0.0)
    
    raw_sum = float(density.sum())
    if raw_sum > 1e-3:
        scale = np.exp(a * np.log(raw_sum) + b)
        density = density * scale

# ─── Noise Suppression ───
# Threshold + Morphological Opening to remove background artifacts
density[density < 0.015] = 0
kernel = np.ones((3, 3), np.uint8)
mask = (density > 0).astype(np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
density = density * mask.astype(np.float32)

estimated_count = int(density.sum())

# ─── Visualize ───
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].set_title("Original Image (Resized for View)")
axes[0].imshow(cv2.cvtColor(cv2.resize(img_bgr, (512, 512)), cv2.COLOR_BGR2RGB))
axes[0].axis("off")

axes[1].set_title(f"Predicted Density (Count: {estimated_count})")
axes[1].imshow(density, cmap="jet")
axes[1].axis("off")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/prediction_tiled.png", dpi=200)
print(f"Estimated Crowd Count: {estimated_count}")
print(f"Result saved to results/prediction_tiled.png")
plt.show()