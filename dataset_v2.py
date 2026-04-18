"""
dataset_v2.py — Unified Dataset Pipeline (Steps 1, 2, 5, 6)
=============================================================
• ShanghaiTech A + B + JHU (filtered ≤100)
• Geometry-adaptive density maps (KDTree sigma)
• Stratified patch sampling (40% sparse / 30% medium / 30% bg)
• Multi-scale training (256 or 384) with augmentations
"""

import os
import cv2
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import torch


# ═══════════════════════════════════════════════════════════════
# STEP 2: ADAPTIVE DENSITY MAP (KDTree)
# ═══════════════════════════════════════════════════════════════

def generate_density_map_adaptive(points, img_shape):
    """
    Geometry-adaptive Gaussian density map.
    sigma per head = clamp(avg_dist_to_3_nearest * 0.3, 1.5, 10)
    """
    h, w = img_shape[:2]
    density = np.zeros((h, w), dtype=np.float32)
    n = len(points)
    if n == 0:
        return density

    points = np.array(points, dtype=np.float64)

    # Single point → fixed sigma
    if n == 1:
        x = min(max(int(round(points[0, 0])), 0), w - 1)
        y = min(max(int(round(points[0, 1])), 0), h - 1)
        density[y, x] = 1.0
        return gaussian_filter(density, sigma=4.0)

    # KDTree for neighbor distances
    k_neighbors = min(4, n)  # query self + 3 neighbors
    tree = KDTree(points)
    dists, _ = tree.query(points, k=k_neighbors)

    # dists[:, 0] is self-distance (0), use dists[:, 1:]
    if k_neighbors > 1:
        avg_dists = dists[:, 1:].mean(axis=1)
    else:
        avg_dists = np.full(n, 4.0)

    sigmas = np.clip(avg_dists * 0.3, 1.5, 10.0)

    # Stamp each head independently with its sigma
    for i in range(n):
        x = min(max(int(round(points[i, 0])), 0), w - 1)
        y = min(max(int(round(points[i, 1])), 0), h - 1)

        sigma = sigmas[i]
        # Create a small Gaussian patch around the point
        sz = int(6 * sigma + 1)  # 3-sigma each side
        x0 = max(0, x - sz)
        x1 = min(w, x + sz + 1)
        y0 = max(0, y - sz)
        y1 = min(h, y + sz + 1)

        # Build local Gaussian
        yy, xx = np.mgrid[y0:y1, x0:x1]
        kernel = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        kernel /= (kernel.sum() + 1e-8)  # normalize to sum=1

        density[y0:y1, x0:x1] += kernel.astype(np.float32)

    return density


# ═══════════════════════════════════════════════════════════════
# STEP 1: DATASET LOADERS (Shanghai A+B + JHU ≤100)
# ═══════════════════════════════════════════════════════════════

# ── Paths ──
TRAIN_A_IMG = "data/part_A_final/train_data/images"
TRAIN_A_GT  = "data/part_A_final/train_data/ground_truth"
TRAIN_B_IMG = "data/part_B_final/train_data/images"
TRAIN_B_GT  = "data/part_B_final/train_data/ground_truth"

TEST_A_IMG  = "data/part_A_final/test_data/images"
TEST_A_GT   = "data/part_A_final/test_data/ground_truth"
TEST_B_IMG  = "data/part_B_final/test_data/images"
TEST_B_GT   = "data/part_B_final/test_data/ground_truth"

JHU_TRAIN   = "jhu_crowd_v2.0/train"
JHU_VAL     = "jhu_crowd_v2.0/val"
JHU_TEST    = "jhu_crowd_v2.0/test"


def _multi_scale_crop(img, density, base_size=256):
    """Step 6: Randomly pick 256 or 384 crop size."""
    crop_size = np.random.choice([256, 384])
    h, w = img.shape[:2]

    if h >= crop_size and w >= crop_size:
        y = np.random.randint(0, h - crop_size + 1)
        x = np.random.randint(0, w - crop_size + 1)
        img_c = img[y:y+crop_size, x:x+crop_size]
        den_c = density[y:y+crop_size, x:x+crop_size]
    else:
        # Pad if image is smaller
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        img_c = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
        den_c = np.pad(density, ((0, pad_h), (0, pad_w)), mode="constant")
        img_c = img_c[:crop_size, :crop_size]
        den_c = den_c[:crop_size, :crop_size]

    # Resize to base_size for model
    if crop_size != base_size:
        scale = base_size / crop_size
        img_c = cv2.resize(img_c, (base_size, base_size),
                           interpolation=cv2.INTER_LINEAR)
        # Preserve count: density sums must stay the same
        gt_sum = den_c.sum()
        den_c = cv2.resize(den_c, (base_size, base_size),
                           interpolation=cv2.INTER_LINEAR)
        if den_c.sum() > 1e-6:
            den_c = den_c / den_c.sum() * gt_sum

    return img_c, den_c


def _augment(img, density):
    """Step 6: Horizontal flip + brightness jitter."""
    if np.random.rand() > 0.5:
        img = np.flip(img, 1).copy()
        density = np.flip(density, 1).copy()
    # Brightness jitter ±15%
    factor = 1.0 + np.random.uniform(-0.15, 0.15)
    img = np.clip(img * factor, 0, 1)
    return img, density


class ShanghaiTechV2(Dataset):
    """ShanghaiTech Part A or B with adaptive density + multi-scale."""

    def __init__(self, img_dir, gt_dir, augment=False, base_size=256):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.augment = augment
        self.base_size = base_size
        self.images = sorted(
            f for f in os.listdir(img_dir) if f.endswith(".jpg")
        )
        # Pre-compute counts for stratified sampling
        self.counts = []
        for name in self.images:
            gt_path = os.path.join(gt_dir,
                                   "GT_" + name.replace(".jpg", ".mat"))
            try:
                mat = sio.loadmat(gt_path)
                pts = mat["image_info"][0][0][0][0][0]
                self.counts.append(len(pts))
            except Exception:
                self.counts.append(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img = cv2.imread(os.path.join(self.img_dir, name),
                         cv2.IMREAD_GRAYSCALE)
        mat = sio.loadmat(
            os.path.join(self.gt_dir,
                         "GT_" + name.replace(".jpg", ".mat")))
        points = mat["image_info"][0][0][0][0][0].astype(np.float64)
        return self._process(img, points)

    def _process(self, img, points):
        h, w = img.shape[:2]
        density = generate_density_map_adaptive(points, (h, w))
        gt_count = float(density.sum())

        img_f = img.astype(np.float32) / 255.0

        if self.augment:
            img_f, density = _multi_scale_crop(img_f, density, self.base_size)
            img_f, density = _augment(img_f, density)
            gt_count = float(density.sum())
        else:
            # Validation: center crop or resize
            if h >= self.base_size and w >= self.base_size:
                cy = (h - self.base_size) // 2
                cx = (w - self.base_size) // 2
                img_f = img_f[cy:cy+self.base_size, cx:cx+self.base_size]
                density = density[cy:cy+self.base_size, cx:cx+self.base_size]
            else:
                old_sum = density.sum()
                img_f = cv2.resize(img_f, (self.base_size, self.base_size))
                density = cv2.resize(density,
                                     (self.base_size, self.base_size))
                if density.sum() > 1e-6:
                    density = density / density.sum() * old_sum
            gt_count = float(density.sum())

        img_t = torch.tensor(
            np.stack([img_f] * 3, axis=0), dtype=torch.float32)
        den_t = torch.tensor(density, dtype=torch.float32).unsqueeze(0)
        cnt_t = torch.tensor(gt_count, dtype=torch.float32)

        return img_t, den_t, cnt_t


class JHUFilteredDataset(Dataset):
    """
    Step 1: JHU-Crowd V2.0 filtered to images with ≤100 people.
    GT format: one line per person → x y w h sigma_x sigma_y
    """

    def __init__(self, split_dir, max_count=100, augment=False,
                 base_size=256):
        self.split_dir = split_dir
        self.augment = augment
        self.base_size = base_size

        labels_path = os.path.join(split_dir, "image_labels.txt")
        self.items = []  # (image_id, count)
        self.counts = []

        with open(labels_path) as f:
            for line in f:
                parts = line.strip().split(",")
                img_id = parts[0]
                count = int(parts[1])
                if count <= max_count:
                    img_path = os.path.join(split_dir, "images",
                                            f"{img_id}.jpg")
                    gt_path = os.path.join(split_dir, "gt",
                                           f"{img_id}.txt")
                    if os.path.exists(img_path) and os.path.exists(gt_path):
                        self.items.append((img_id, count, img_path, gt_path))
                        self.counts.append(count)

        print(f"  JHU {os.path.basename(split_dir)}: "
              f"{len(self.items)} images with ≤{max_count} people")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_id, count, img_path, gt_path = self.items[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback to next
            return self.__getitem__((idx + 1) % len(self))

        # Parse JHU GT: each line is "x y w h sigma_x sigma_y"
        points = []
        with open(gt_path) as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) >= 2:
                    points.append([float(vals[0]), float(vals[1])])
        points = np.array(points) if points else np.zeros((0, 2))

        return self._process(img, points)

    def _process(self, img, points):
        h, w = img.shape[:2]
        density = generate_density_map_adaptive(points, (h, w))
        gt_count = float(density.sum())

        img_f = img.astype(np.float32) / 255.0

        if self.augment:
            img_f, density = _multi_scale_crop(img_f, density, self.base_size)
            img_f, density = _augment(img_f, density)
            gt_count = float(density.sum())
        else:
            if h >= self.base_size and w >= self.base_size:
                cy = (h - self.base_size) // 2
                cx = (w - self.base_size) // 2
                img_f = img_f[cy:cy+self.base_size, cx:cx+self.base_size]
                density = density[cy:cy+self.base_size, cx:cx+self.base_size]
            else:
                old_sum = density.sum()
                img_f = cv2.resize(img_f, (self.base_size, self.base_size))
                density = cv2.resize(density,
                                     (self.base_size, self.base_size))
                if density.sum() > 1e-6:
                    density = density / density.sum() * old_sum
            gt_count = float(density.sum())

        img_t = torch.tensor(
            np.stack([img_f] * 3, axis=0), dtype=torch.float32)
        den_t = torch.tensor(density, dtype=torch.float32).unsqueeze(0)
        cnt_t = torch.tensor(gt_count, dtype=torch.float32)

        return img_t, den_t, cnt_t


# ═══════════════════════════════════════════════════════════════
# STEP 5: STRATIFIED SAMPLING
# ═══════════════════════════════════════════════════════════════

def build_stratified_sampler(dataset):
    """
    40% patches 1–20 people, 30% patches 21–50, 30% everything else.
    Works with any dataset that has a .counts attribute.
    """
    counts = np.array(dataset.counts, dtype=float)
    weights = np.ones(len(counts), dtype=float)

    mask_sparse = (counts >= 1) & (counts <= 20)
    mask_medium = (counts > 20) & (counts <= 50)
    mask_other  = ~mask_sparse & ~mask_medium

    n_sparse = mask_sparse.sum()
    n_medium = mask_medium.sum()
    n_other  = mask_other.sum()

    # Target: 40/30/30 split
    if n_sparse > 0:
        weights[mask_sparse] = 0.4 / n_sparse
    if n_medium > 0:
        weights[mask_medium] = 0.3 / n_medium
    if n_other > 0:
        weights[mask_other] = 0.3 / n_other

    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(counts),
        replacement=True,
    )


class ConcatDatasetWithCounts(ConcatDataset):
    """ConcatDataset that merges .counts lists for stratified sampling."""

    @property
    def counts(self):
        all_counts = []
        for ds in self.datasets:
            if hasattr(ds, "counts"):
                all_counts.extend(ds.counts)
            else:
                all_counts.extend([0] * len(ds))
        return all_counts


# ═══════════════════════════════════════════════════════════════
# BUILDER FUNCTION
# ═══════════════════════════════════════════════════════════════

def build_dataloaders(batch_size=4, base_size=256, num_workers=0):
    """Build train (stratified) + val dataloaders."""

    # ── Train datasets ──
    train_sets = []

    if os.path.isdir(TRAIN_A_IMG):
        ds = ShanghaiTechV2(TRAIN_A_IMG, TRAIN_A_GT, augment=True,
                            base_size=base_size)
        train_sets.append(ds)
        print(f"  ✓ ShanghaiTech A train: {len(ds)} images")

    if os.path.isdir(TRAIN_B_IMG):
        ds = ShanghaiTechV2(TRAIN_B_IMG, TRAIN_B_GT, augment=True,
                            base_size=base_size)
        train_sets.append(ds)
        print(f"  ✓ ShanghaiTech B train: {len(ds)} images")

    if os.path.isdir(JHU_TRAIN):
        ds = JHUFilteredDataset(JHU_TRAIN, max_count=100, augment=True,
                                base_size=base_size)
        train_sets.append(ds)

    if os.path.isdir(JHU_VAL):
        ds = JHUFilteredDataset(JHU_VAL, max_count=100, augment=True,
                                base_size=base_size)
        train_sets.append(ds)

    train_dataset = ConcatDatasetWithCounts(train_sets)
    sampler = build_stratified_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    # ── Val: Part B test (sparse-focused validation) ──
    val_dataset = ShanghaiTechV2(TEST_B_IMG, TEST_B_GT, augment=False,
                                 base_size=base_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    total = len(train_dataset)
    print(f"\n  Total training images: {total}")
    print(f"  Validation images: {len(val_dataset)} (Part B test)")

    return train_loader, val_loader
