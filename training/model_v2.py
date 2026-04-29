"""
model_v2.py — CrowdDensityNetV2 (Dual-Head)
=============================================
Step 4: Encoder + density decoder + regression head.
final_count = 0.7 * density_sum + 0.3 * direct_count
"""

import torch
import torch.nn as nn
from torchvision import models


class CrowdDensityNetV2(nn.Module):
    """Dual-head crowd density model: density map + direct regression."""

    def __init__(self, pretrained=True):
        super().__init__()

        # ── Encoder: VGG-16 features[:24] ──
        vgg = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.encoder = nn.Sequential(*list(vgg.features.children())[:24])

        # ── Density decoder (same architecture as v1) ──
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(True),
        )

        # ── Regression head: encoder features → scalar count ──
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.ReLU(True),  # count ≥ 0
        )

    def forward(self, x):
        features = self.encoder(x)
        density = self.decoder(features)
        count = self.regressor(features)  # (B, 1)
        return density, count.squeeze(1)

    def predict_count(self, x, alpha=0.7):
        """Fused prediction: alpha*density_sum + (1-alpha)*regressor."""
        density, direct = self.forward(x)
        density_sum = density.sum(dim=(1, 2, 3))
        return alpha * density_sum + (1 - alpha) * direct, density


# ── Backward-compatible wrapper for loading v1 weights ──

def load_v2_from_v1(v1_path, device="cpu"):
    """Load v1 checkpoint into v2 model (regressor starts random)."""
    model = CrowdDensityNetV2(pretrained=False)
    v1_state = torch.load(v1_path, map_location=device, weights_only=True)

    # Load matching keys, skip regressor
    model_state = model.state_dict()
    loaded = 0
    for k, v in v1_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
    model.load_state_dict(model_state)
    print(f"  ✓ Loaded {loaded}/{len(v1_state)} v1 weights into V2 "
          f"(regressor head initialized randomly)")
    return model
