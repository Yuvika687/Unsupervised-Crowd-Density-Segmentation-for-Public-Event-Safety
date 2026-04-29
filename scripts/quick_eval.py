import torch
import torch.nn.functional as F
import cv2
import numpy as np
import scipy.io as sio
import os
from train import CrowdDensityNet

device = torch.device("cpu")
model  = CrowdDensityNet()
model.load_state_dict(torch.load("models/best_model.pth",
                     map_location=device))
model.eval()
print("Model loaded!")

TEST_IMG = "data/part_A_final/test_data/images"
TEST_GT  = "data/part_A_final/test_data/ground_truth"
preds, gts = [], []

files = sorted([f for f in os.listdir(TEST_IMG)
                if f.endswith(".jpg")])[:50]
print(f"Running on {len(files)} images...")

for fname in files:
    img = cv2.imread(os.path.join(TEST_IMG, fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)
        out = F.interpolate(out, size=(256,256),
                           mode="bilinear", align_corners=False)
    preds.append(float(out.sum()))

    gt_path = os.path.join(TEST_GT,
              "GT_" + fname.replace(".jpg", ".mat"))
    mat = sio.loadmat(gt_path)
    gts.append(len(mat['image_info'][0][0][0][0][0]))

mae = np.mean(np.abs(np.array(preds) - np.array(gts)))
mse = np.mean((np.array(preds) - np.array(gts))**2)

print(f"\nResults using your VGG-16 model:")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"Avg predicted count : {np.mean(preds):.1f}")
print(f"Avg ground truth    : {np.mean(gts):.1f}")
