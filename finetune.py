import os
import cv2
import torch
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import random

# ─── Paths ────────────────────────────────────────────────────────────────────

TRAIN_IMG_A = "data/part_A_final/train_data/images"
TRAIN_GT_A  = "data/part_A_final/train_data/ground_truth"
TRAIN_IMG_B = "data/part_B_final/train_data/images"
TRAIN_GT_B  = "data/part_B_final/train_data/ground_truth"
TEST_IMG_A  = "data/part_A_final/test_data/images"
TEST_GT_A   = "data/part_A_final/test_data/ground_truth"
TEST_IMG_B  = "data/part_B_final/test_data/images"
TEST_GT_B   = "data/part_B_final/test_data/ground_truth"

JHU_ROOT = "jhu_crowd_v2.0"
IMG_SIZE = 256
MAX_CROWD_COUNT_SPARSE = 200

# ─── Density Map Generation (SIGMA ONLY — no count normalization) ─────────────

def generate_density_map(img_shape, points):
    h, w = img_shape[0], img_shape[1]
    density = np.zeros((h, w), dtype=np.float32)

    count = len(points)
    if count == 0:
        return density

    if count < 20:
        sigma = 4
    elif count < 50:
        sigma = 6
    elif count < 200:
        sigma = 10
    else:
        sigma = 15

    for p in points:
        x = min(max(int(round(p[0])), 0), w - 1)
        y = min(max(int(round(p[1])), 0), h - 1)
        density[y, x] += 1

    density = gaussian_filter(density, sigma=sigma)
    return density


# ─── ShanghaiTech Dataset ────────────────────────────────────────────────────

class ShanghaiTechDataset(Dataset):
    def __init__(self, img_dir, gt_dir, augment=False):
        self.img_dir = img_dir
        self.gt_dir  = gt_dir
        self.augment = augment
        self.images  = sorted(
            [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = cv2.imread(
            os.path.join(self.img_dir, img_name), cv2.IMREAD_GRAYSCALE
        )
        orig_h, orig_w = img.shape[:2]

        mat_path = os.path.join(
            self.gt_dir, "GT_" + img_name.replace(".jpg", ".mat")
        )
        mat = sio.loadmat(mat_path)
        points = mat["image_info"][0][0][0][0][0]

        density = generate_density_map((orig_h, orig_w), points)

        if self.augment:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
                density = np.flip(density, axis=1).copy()

            factor = 1.0 + np.random.uniform(-0.2, 0.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

        gt_count = density.sum()
        density = cv2.resize(density, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
        resized_sum = density.sum()
        if resized_sum > 1e-6:
            density = density / resized_sum * gt_count

        img_tensor = torch.tensor(np.stack([img, img, img], axis=0), dtype=torch.float32)
        den_tensor = torch.tensor(density, dtype=torch.float32).unsqueeze(0)

        return img_tensor, den_tensor

# ─── JHU-CROWD++ Dataset ────────────────────────────────────────────────────

class JHUSparseDataset(Dataset):
    def __init__(self, split="train", augment=False):
        self.split_dir = os.path.join(JHU_ROOT, split)
        self.img_dir = os.path.join(self.split_dir, "images")
        self.gt_dir = os.path.join(self.split_dir, "gt")
        self.augment = augment
        
        # Parse image labels to filter sparse images
        labels_file = os.path.join(self.split_dir, "image_labels.txt")
        self.images = []
        with open(labels_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    img_id = parts[0]
                    count = int(parts[1])
                    if count <= MAX_CROWD_COUNT_SPARSE:
                        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
                        gt_path = os.path.join(self.gt_dir, f"{img_id}.txt")
                        if os.path.exists(img_path) and os.path.exists(gt_path):
                            self.images.append((img_id, count))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id, count = self.images[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        gt_path = os.path.join(self.gt_dir, f"{img_id}.txt")
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = img.shape[:2]
        
        # Parse GT points
        points = []
        with open(gt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    points.append([float(parts[0]), float(parts[1])])
        points = np.array(points) if len(points) > 0 else np.zeros((0, 2))
        
        density = generate_density_map((orig_h, orig_w), points)

        if self.augment:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
                density = np.flip(density, axis=1).copy()

            factor = 1.0 + np.random.uniform(-0.2, 0.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

        gt_count = density.sum()
        density = cv2.resize(density, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
        resized_sum = density.sum()
        if resized_sum > 1e-6:
            density = density / resized_sum * gt_count

        img_tensor = torch.tensor(np.stack([img, img, img], axis=0), dtype=torch.float32)
        den_tensor = torch.tensor(density, dtype=torch.float32).unsqueeze(0)

        return img_tensor, den_tensor


# ─── Model (ARCHITECTURE UNCHANGED) ──────────────────────────────────────────

class CrowdDensityNet(nn.Module):
    def __init__(self, freeze_layers=0):
        super(CrowdDensityNet, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(vgg.features.children())[:24])

        for i, param in enumerate(self.encoder.parameters()):
            if i < freeze_layers:
                param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_all(model, loaders_dict, device):
    results = {}
    total_mae = 0.0
    total_count = 0

    for name, loader in loaders_dict.items():
        if loader is None or len(loader.dataset) == 0:
            continue
        model.eval()
        dataset_mae = 0.0
        dataset_count = 0
        with torch.no_grad():
            for imgs, densities in loader:
                imgs, densities = imgs.to(device), densities.to(device)
                preds = model(imgs)
                pred_counts = preds.sum(dim=(1, 2, 3))
                actual_counts = densities.sum(dim=(1, 2, 3))
                dataset_mae += torch.abs(pred_counts - actual_counts).sum().item()
                dataset_count += imgs.size(0)

        avg_mae = dataset_mae / max(dataset_count, 1)
        results[name] = avg_mae
        total_mae += dataset_mae
        total_count += dataset_count

    results["Combined"] = total_mae / max(total_count, 1)
    return results


# ─── Training ─────────────────────────────────────────────────────────────────

def finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    BEST_MODEL_PATH = "models/best_model.pth"
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: Could not find pre-trained model at {BEST_MODEL_PATH}. Please run train.py first.")
        return

    print("Loading datasets...")

    train_A = ShanghaiTechDataset(TRAIN_IMG_A, TRAIN_GT_A, augment=True)
    print(f"  Part A train: {len(train_A)} images")

    train_B = ShanghaiTechDataset(TRAIN_IMG_B, TRAIN_GT_B, augment=True)
    print(f"  Part B train: {len(train_B)} images")
    
    train_JHU = JHUSparseDataset(split="train", augment=True)
    print(f"  JHU Sparse train: {len(train_JHU)} images")

    train_set = ConcatDataset([train_A, train_B, train_JHU])

    val_A = ShanghaiTechDataset(TEST_IMG_A, TEST_GT_A, augment=False)
    val_B = ShanghaiTechDataset(TEST_IMG_B, TEST_GT_B, augment=False)
    val_JHU = JHUSparseDataset(split="val", augment=False)
    
    print(f"  Part A val: {len(val_A)} images")
    print(f"  Part B val: {len(val_B)} images")
    print(f"  JHU Sparse val: {len(val_JHU)} images")
    
    print(f"\n  Total training: {len(train_set)} images")

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
    
    val_loaders = {
        "Part_A": DataLoader(val_A, batch_size=4, shuffle=False, num_workers=0),
        "Part_B": DataLoader(val_B, batch_size=4, shuffle=False, num_workers=0),
        "JHU_Sparse": DataLoader(val_JHU, batch_size=4, shuffle=False, num_workers=0),
    }

    model = CrowdDensityNet(freeze_layers=10).to(device)
    print(f"Loading weights from {BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    # Lower learning rate for fine-tuning
    optimizer = optim.Adam([
        {"params": model.encoder.parameters(), "lr": 1e-6},
        {"params": model.decoder.parameters(), "lr": 1e-5},
    ])

    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    epochs = 15
    best_mae = float("inf")

    print(f"\n{'═' * 70}")
    print(f"  Starting fine-tuning: {epochs} epochs, batch_size=4")
    print(f"  Encoder LR: 1e-6")
    print(f"  Decoder LR: 1e-5")
    print(f"{'═' * 70}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (imgs, densities) in enumerate(train_loader):
            imgs, densities = imgs.to(device), densities.to(device)
            preds = model(imgs)
            loss = criterion(preds, densities)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(f"    Epoch {epoch:02d} — batch {batch_idx + 1}/"
                      f"{len(train_loader)}, loss: {loss.item():.6f}")

        avg_loss = total_loss / len(train_set)

        # Validate on all sets
        val_results = validate_all(model, val_loaders, device)
        combined_mae = val_results.get("Combined", float("inf"))
        
        scheduler.step(combined_mae)

        enc_lr = optimizer.param_groups[0]["lr"]
        dec_lr = optimizer.param_groups[1]["lr"]

        print(f"\nEpoch [{epoch:02d}/{epochs}]")
        print(f"  Train Loss:    {avg_loss:.6f}")
        for name, mae_val in val_results.items():
            marker = "  ←" if name == "Combined" else ""
            print(f"  Val MAE {name:>10s}: {mae_val:.2f}{marker}")
        print(f"  LR: enc={enc_lr:.2e}, dec={dec_lr:.2e}")

        if combined_mae < best_mae:
            best_mae = combined_mae
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"  ✓ Best model updated (Combined MAE: {best_mae:.2f})")

        print()

    print(f"{'═' * 70}")
    print(f"  Fine-tuning complete!")
    print(f"  Best Combined MAE: {best_mae:.2f}")
    print(f"  Updated model saved to: models/best_model.pth")
    print(f"{'═' * 70}")

if __name__ == "__main__":
    finetune()
