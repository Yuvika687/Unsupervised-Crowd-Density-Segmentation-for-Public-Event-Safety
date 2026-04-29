import os
import scipy.io as sio

base = "data"

expected = {
    "part_A_final/train_data": 300,
    "part_A_final/test_data": 182,
    "part_B_final/train_data": 400,
    "part_B_final/test_data": 316,
}

for folder, count in expected.items():
    img_dir = os.path.join(base, folder, "images")
    gt_dir = os.path.join(base, folder, "ground_truth")

    imgs = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    gts = len([f for f in os.listdir(gt_dir) if f.endswith(".mat")])

    print(folder, ":", imgs, "images,", gts, "ground truth files")

print("Dataset verification complete.")
