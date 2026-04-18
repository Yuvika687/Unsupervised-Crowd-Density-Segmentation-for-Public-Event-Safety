import torch
import torch.nn as nn
import numpy as np
import cv2
import scipy.io as sio
import os
from torchvision import models

IMG_SIZE   = 256
MODEL_PATH = "models/best_model.pth"
TEST_IMG_B = "data/part_B_final/test_data/images"
TEST_GT_B  = "data/part_B_final/test_data/ground_truth"
TEST_IMG_A = "data/part_A_final/test_data/images"
TEST_GT_A  = "data/part_A_final/test_data/ground_truth"

class CrowdDensityNet(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.encoder = nn.Sequential(
            *list(vgg.features.children())[:24])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,2,1),
            nn.BatchNorm2d(256),nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(128),nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,32,4,2,1),
            nn.BatchNorm2d(32),nn.ReLU(inplace=True),
            nn.Conv2d(32,1,1),nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = CrowdDensityNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu",
                      weights_only=True))
model.eval()
print("Model loaded")
print("="*60)

def get_count(gt_dir, fname):
    mat = sio.loadmat(os.path.join(gt_dir,
          "GT_" + fname.replace(".jpg",".mat")))
    return len(mat["image_info"][0][0][0][0][0])

def run_image(img_path):
    img  = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res  = cv2.resize(gray,(IMG_SIZE,IMG_SIZE)).astype(np.float32)/255.0
    t    = torch.tensor(np.stack([res,res,res],axis=0),
                        dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(t)
    dm = out.squeeze().numpy()
    return dm

ratios = []

print("PART B (sparse 9-578 people):")
print("-"*60)
files = sorted(os.listdir(TEST_IMG_B))[:10]
for f in files:
    if not f.endswith(".jpg"): continue
    actual  = get_count(TEST_GT_B, f)
    dm      = run_image(os.path.join(TEST_IMG_B, f))
    raw_sum = float(dm.sum())
    ratio   = actual / (raw_sum + 1e-9)
    ratios.append(ratio)
    print(f"{f[:20]:20s} | actual={actual:4d} | "
          f"raw_sum={raw_sum:10.4f} | "
          f"ratio={ratio:8.2f}")

print()
print("PART A (dense 300-3000 people):")
print("-"*60)
files_a = sorted(os.listdir(TEST_IMG_A))[:5]
for f in files_a:
    if not f.endswith(".jpg"): continue
    actual  = get_count(TEST_GT_A, f)
    dm      = run_image(os.path.join(TEST_IMG_A, f))
    raw_sum = float(dm.sum())
    ratio   = actual / (raw_sum + 1e-9)
    ratios.append(ratio)
    print(f"{f[:20]:20s} | actual={actual:4d} | "
          f"raw_sum={raw_sum:10.4f} | "
          f"ratio={ratio:8.2f}")

scale = float(np.median(ratios))
print()
print("="*60)
print(f"Recommended scale factor: {scale:.4f}")
print(f"Add this to app.py predict_density:")
print(f"  dm = dm * {scale:.4f}")

with open("models/scale_factor.txt", "w") as f:
    f.write(str(scale))
print("Saved to models/scale_factor.txt")

