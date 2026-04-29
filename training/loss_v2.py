"""
loss_v2.py — Weighted MSE + Count Loss
========================================
Step 3 + Step 7: Combined loss with density normalization.

Total = 0.6 * WeightedMSE(pred_density, gt_density)
      + 0.4 * L1(pred_count, gt_count)

Density normalization (Step 7) applied inside the loss.
"""

import torch
import torch.nn as nn


class CrowdCountingLoss(nn.Module):
    """Combined loss for dual-head crowd counting."""

    def __init__(self, density_weight=0.6, count_weight=0.4,
                 local_weight=5.0):
        super().__init__()
        self.dw = density_weight
        self.cw = count_weight
        self.local_weight = local_weight

    def forward(self, pred_density, pred_count,
                gt_density, gt_count):
        """
        Args:
            pred_density: (B, 1, H, W) predicted density map
            pred_count:   (B,) predicted scalar count from regressor
            gt_density:   (B, 1, H, W) ground-truth density map
            gt_count:     (B,) ground-truth count (sum of gt_density)
        """
        # ── Step 7: Density normalization ──
        # Normalize GT density so it sums to gt_count exactly
        gt_sum = gt_density.sum(dim=(1, 2, 3), keepdim=True)
        gt_density_norm = torch.where(
            gt_sum > 1e-6,
            gt_density / (gt_sum + 1e-6) * gt_count.view(-1, 1, 1, 1),
            gt_density,
        )

        # ── Step 3: Weighted MSE ──
        # Higher weight where density is higher → penalizes missing heads more
        weight = 1.0 + self.local_weight * gt_density_norm.detach()
        sq_diff = (pred_density - gt_density_norm) ** 2
        weighted_mse = (weight * sq_diff).mean()

        # ── Count loss (L1) ──
        count_loss = torch.abs(pred_count - gt_count).mean()

        total = self.dw * weighted_mse + self.cw * count_loss

        return total, weighted_mse.item(), count_loss.item()
