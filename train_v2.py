"""
train_v2.py — V2 Training Loop (Improved)
==========================================
Changes from previous version (architecture UNCHANGED):

LOSS IMPROVEMENTS:
  - SSIM loss term → sharper density maps, penalizes blurry predictions
  - Focal-weighted MSE → upweights low-density regions (reduces false negatives)
  - Log-cosh count loss → smoother than L1, robust to outliers
  - Count-aware weighting → dynamically balances density vs count loss
    based on crowd size (sparse scenes get stronger count supervision)

TRAINING STRATEGY:
  - Linear warmup (5 epochs) → prevents early overfitting of pretrained encoder
  - Gradient accumulation (effective batch = 16) → stabler gradients on CPU
  - EMA model for validation → smoother, more reliable checkpointing
  - Per-bucket validation logging → tracks sparse vs dense accuracy separately
  - Early stopping (patience=12) → prevents overtraining

NORMALIZATION:
  - Input normalization with ImageNet mean/std → better alignment with VGG features
  - GT density log-scaling aware weighting → stabilizes loss across count ranges

Usage:
    python3 train_v2.py
    python3 calibrate_v2.py
"""

import os
import copy
import math
import time
import torch
import torch.nn.functional as F
import numpy as np
from model_v2 import CrowdDensityNetV2, load_v2_from_v1
from dataset_v2 import build_dataloaders

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

EPOCHS       = 25
BATCH_SIZE   = 4
ACCUM_STEPS  = 4          # effective batch = 4 × 4 = 16
BASE_SIZE    = 256
LR_ENCODER   = 3e-6       # lower: prevent forgetting ImageNet features
LR_DECODER   = 5e-5
LR_REGRESSOR = 1e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0

WARMUP_EPOCHS   = 5       # linear warmup before cosine decay
EMA_DECAY       = 0.999   # exponential moving average for val model
EARLY_STOP_PAT  = 12      # early stopping patience (epochs)

MODEL_V1_PATH = "models/best_model.pth"
MODEL_V2_PATH = "models/best_model_v2.pth"
os.makedirs("models", exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# IMPROVED LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════

def _ssim_loss(pred, gt, window_size=7):
    """
    1 - SSIM between predicted and GT density maps.
    Encourages structural similarity → sharper density maps.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    pad = window_size // 2
    # Use average pooling as the window function (lightweight)
    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_gt   = F.avg_pool2d(gt,   window_size, stride=1, padding=pad)

    mu_pred_sq = mu_pred ** 2
    mu_gt_sq   = mu_gt ** 2
    mu_cross   = mu_pred * mu_gt

    sigma_pred_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=pad) - mu_pred_sq
    sigma_gt_sq   = F.avg_pool2d(gt ** 2,   window_size, stride=1, padding=pad) - mu_gt_sq
    sigma_cross   = F.avg_pool2d(pred * gt, window_size, stride=1, padding=pad) - mu_cross

    ssim = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
           ((mu_pred_sq + mu_gt_sq + C1) * (sigma_pred_sq + sigma_gt_sq + C2))

    return 1.0 - ssim.mean()


def improved_loss(pred_density, pred_count, gt_density, gt_count):
    """
    Combined loss with three components:

    1. Focal-weighted MSE  — upweights where GT has sparse heads (reduces FN)
    2. SSIM loss           — structural similarity (sharper maps)
    3. Log-cosh count loss — smooth, robust count supervision

    Dynamic weighting: sparse scenes (count < 20) get heavier count
    supervision since density maps are unreliable for few-pixel targets.
    """
    # ── Normalize GT density to sum to gt_count ──
    gt_sum = gt_density.sum(dim=(1, 2, 3), keepdim=True)
    gt_density_norm = torch.where(
        gt_sum > 1e-6,
        gt_density / (gt_sum + 1e-6) * gt_count.view(-1, 1, 1, 1),
        gt_density,
    )

    # ── 1. Focal-weighted MSE ──
    # Instead of just upweighting where GT is high (misses sparse heads),
    # also upweight where GT is nonzero but small → catches isolated people
    gt_det = gt_density_norm.detach()
    # Binary presence mask: 1 where any density exists
    presence = (gt_det > 1e-6).float()
    # Focal weight: base=1, +8 for any head region, +4 scaled by density
    weight = 1.0 + 8.0 * presence + 4.0 * gt_det
    # Global sparse boost: 2× weight for scenes with < 20 people
    sparse_boost = (gt_count < 20).float().view(-1, 1, 1, 1)
    weight = weight * (1.0 + 1.0 * sparse_boost)
    weight = torch.clamp(weight, max=20.0)
    sq_diff = (pred_density - gt_density_norm) ** 2
    focal_mse = (weight * sq_diff).mean()

    # ── 2. SSIM loss for sharpness ──
    ssim = _ssim_loss(pred_density, gt_density_norm)

    # ── 3. Log-cosh count loss (smoother than L1) ──
    diff = pred_count - gt_count
    logcosh = torch.log(torch.cosh(torch.clamp(diff, -10, 10))).mean()

    # ── Dynamic weighting based on mean batch count ──
    mean_count = gt_count.mean().item()
    if mean_count < 10:
        # Very sparse: count head matters more, density is few pixels
        w_den, w_ssim, w_cnt = 0.35, 0.15, 0.50
    elif mean_count < 30:
        w_den, w_ssim, w_cnt = 0.45, 0.15, 0.40
    elif mean_count < 60:
        w_den, w_ssim, w_cnt = 0.50, 0.15, 0.35
    else:
        w_den, w_ssim, w_cnt = 0.55, 0.15, 0.30

    total = w_den * focal_mse + w_ssim * ssim + w_cnt * logcosh

    return total, focal_mse.item(), ssim.item(), logcosh.item()


# ═══════════════════════════════════════════════════════════════
# EMA MODEL (Exponential Moving Average)
# ═══════════════════════════════════════════════════════════════

class EMAModel:
    """
    Maintains an exponential moving average of model weights.
    Provides more stable validation metrics and better final checkpoint.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(),
                                     model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


# ═══════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULE (Warmup + Cosine)
# ═══════════════════════════════════════════════════════════════

def get_lr_lambda(warmup_epochs, total_epochs):
    """Linear warmup → cosine annealing."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear 0→1
        # Cosine decay after warmup
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def train():
    device = torch.device("cpu")
    print("=" * 70)
    print("  SafeCrowd V2 Training (Improved)")
    print("=" * 70)
    print(f"  Device:          {device}")
    print(f"  Epochs:          {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Accum: {ACCUM_STEPS}")
    print(f"  Effective Batch: {BATCH_SIZE * ACCUM_STEPS}")
    print(f"  Size:            {BASE_SIZE}")
    print(f"  Warmup:          {WARMUP_EPOCHS} epochs")
    print(f"  EMA Decay:       {EMA_DECAY}")
    print(f"  Early Stop:      {EARLY_STOP_PAT} epochs patience")
    print()

    # ── Data ──
    train_loader, val_loader = build_dataloaders(
        batch_size=BATCH_SIZE, base_size=BASE_SIZE)

    # ── Model ──
    if os.path.exists(MODEL_V2_PATH):
        model = CrowdDensityNetV2(pretrained=False)
        model.load_state_dict(
            torch.load(MODEL_V2_PATH, map_location=device, weights_only=True))
        print("  ✓ Resuming from best_model_v2.pth")
    elif os.path.exists(MODEL_V1_PATH):
        model = load_v2_from_v1(MODEL_V1_PATH, device)
    else:
        model = CrowdDensityNetV2(pretrained=True)
        print("  ✓ Starting from ImageNet pretrained VGG-16")

    model = model.to(device)

    # ── EMA ──
    ema = EMAModel(model, decay=EMA_DECAY)

    # ── Optimizer: AdamW with per-component LR ──
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),  "lr": LR_ENCODER},
        {"params": model.decoder.parameters(),  "lr": LR_DECODER},
        {"params": model.regressor.parameters(), "lr": LR_REGRESSOR},
    ], weight_decay=WEIGHT_DECAY)

    # ── Scheduler: Warmup + Cosine ──
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=get_lr_lambda(WARMUP_EPOCHS, EPOCHS))

    # ── Training loop ──
    best_mae = float("inf")
    patience_counter = 0
    log_path = "training_v2_log.txt"

    print(f"\n{'Ep':>3} {'Loss':>9} {'FocMSE':>9} {'SSIM':>7} {'CntL':>7} "
          f"{'vMAE':>7} {'v1-20':>7} {'v21-50':>7} {'Best':>7} "
          f"{'LR':>10} {'Time':>6}")
    print("-" * 100)

    with open(log_path, "w") as log:
        log.write("epoch,loss,focal_mse,ssim_loss,count_loss,"
                  "val_mae,val_mae_1_20,val_mae_21_50,best_mae,lr,time_s\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        epoch_loss   = 0.0
        epoch_fmse   = 0.0
        epoch_ssim   = 0.0
        epoch_cnt    = 0.0
        n_batches    = 0

        optimizer.zero_grad()

        for batch_idx, (imgs, dens, counts) in enumerate(train_loader):
            imgs   = imgs.to(device)
            dens   = dens.to(device)
            counts = counts.to(device)

            pred_density, pred_count = model(imgs)
            loss, fmse_v, ssim_v, cnt_v = improved_loss(
                pred_density, pred_count, dens, counts)

            # Scale loss for gradient accumulation
            scaled_loss = loss / ACCUM_STEPS
            scaled_loss.backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0 or \
               (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
                # Update EMA after each optimizer step
                ema.update(model)

            epoch_loss += loss.item()
            epoch_fmse += fmse_v
            epoch_ssim += ssim_v
            epoch_cnt  += cnt_v
            n_batches  += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_fmse = epoch_fmse / max(n_batches, 1)
        avg_ssim = epoch_ssim / max(n_batches, 1)
        avg_cnt  = epoch_cnt  / max(n_batches, 1)
        lr = optimizer.param_groups[1]["lr"]  # decoder LR

        # ── Validate with EMA model ──
        ema_model = ema.shadow
        ema_model.eval()
        val_ae = 0.0
        val_n  = 0
        # Per-bucket tracking
        bucket_ae  = {1: 0.0, 2: 0.0, 3: 0.0}   # 1-20, 21-50, 51+
        bucket_n   = {1: 0,   2: 0,   3: 0}

        with torch.no_grad():
            for imgs, dens, counts in val_loader:
                imgs = imgs.to(device)
                pred_density, pred_count = ema_model(imgs)

                # Fused prediction
                density_sum = pred_density.sum(dim=(1, 2, 3))
                fused = 0.7 * density_sum + 0.3 * pred_count

                errors = torch.abs(fused - counts)
                val_ae += errors.sum().item()
                val_n  += len(counts)

                # Per-bucket
                for i in range(len(counts)):
                    c = counts[i].item()
                    e = errors[i].item()
                    if c <= 20:
                        bucket_ae[1] += e; bucket_n[1] += 1
                    elif c <= 50:
                        bucket_ae[2] += e; bucket_n[2] += 1
                    else:
                        bucket_ae[3] += e; bucket_n[3] += 1

        val_mae = val_ae / max(val_n, 1)
        mae_1_20  = bucket_ae[1] / max(bucket_n[1], 1)
        mae_21_50 = bucket_ae[2] / max(bucket_n[2], 1)
        elapsed = time.time() - t0

        # ── Checkpoint on best val MAE ──
        improved = ""
        if val_mae < best_mae:
            best_mae = val_mae
            # Save EMA weights (more stable)
            torch.save(ema.state_dict(), MODEL_V2_PATH)
            improved = " ★"
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"{epoch:3d} {avg_loss:9.5f} {avg_fmse:9.5f} {avg_ssim:7.4f} "
              f"{avg_cnt:7.4f} {val_mae:7.2f} {mae_1_20:7.2f} "
              f"{mae_21_50:7.2f} {best_mae:7.2f} {lr:10.2e} "
              f"{elapsed:5.1f}s{improved}")

        with open(log_path, "a") as log:
            log.write(f"{epoch},{avg_loss:.6f},{avg_fmse:.6f},"
                      f"{avg_ssim:.4f},{avg_cnt:.4f},"
                      f"{val_mae:.2f},{mae_1_20:.2f},{mae_21_50:.2f},"
                      f"{best_mae:.2f},{lr:.2e},{elapsed:.1f}\n")

        # ── Early stopping ──
        if patience_counter >= EARLY_STOP_PAT:
            print(f"\n  ⏹  Early stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PAT} epochs)")
            break

    print("\n" + "=" * 70)
    print(f"  Training complete. Best val MAE: {best_mae:.2f}")
    print(f"  Model saved to {MODEL_V2_PATH} (EMA weights)")
    print(f"  Log saved to {log_path}")
    print("=" * 70)
    print("\n  Next steps:")
    print("    python3 calibrate_v2.py")
    print("    python3 eval_v2.py")


if __name__ == "__main__":
    train()
