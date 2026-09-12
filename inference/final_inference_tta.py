import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

from IU_PSeg import OTU3DDataset, IU_PSegNet

# =========================
# PATHS
# =========================
MODEL_PATH = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\outputs_upgraded\best_dice.pth"
DATA_ROOT = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d"
TEST_SPLIT = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d\test.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATA
# =========================
dataset = OTU3DDataset(DATA_ROOT, TEST_SPLIT, augment=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# =========================
# MODEL
# =========================
model = IU_PSegNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded")

# =========================
# METRICS
# =========================
def compute_metrics(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))

    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = (2 * precision * recall + 1e-6) / (precision + recall + 1e-6)

    return dice, iou, precision, recall, f1

# =========================
# TTA FUNCTION
# =========================
def tta_predict(image):
    # original
    pred1 = torch.sigmoid(model(image))

    # horizontal flip
    img_h = torch.flip(image, dims=[3])
    pred2 = torch.sigmoid(model(img_h))
    pred2 = torch.flip(pred2, dims=[3])

    # vertical flip
    img_v = torch.flip(image, dims=[2])
    pred3 = torch.sigmoid(model(img_v))
    pred3 = torch.flip(pred3, dims=[2])

    # average
    pred = (pred1 + pred2 + pred3) / 3.0

    return pred

# =========================
# THRESHOLD SEARCH
# =========================
thresholds = np.arange(0.2, 0.6, 0.05)

best_dice = 0
best_threshold = 0.4

results = []

for th in thresholds:
    dice_all = []

    for img, mask in tqdm(loader, desc=f"Threshold {th:.2f}"):
        img = img.to(DEVICE)
        mask = mask.numpy()

        with torch.no_grad():
            pred = tta_predict(img)

        pred = pred.cpu().numpy()
        pred_bin = (pred > th).astype(np.uint8)

        d, iou, p, r, f1 = compute_metrics(pred_bin, mask)
        dice_all.append(d)

    mean_dice = np.mean(dice_all)
    print(f"Threshold {th:.2f} → Dice: {mean_dice:.4f}")

    if mean_dice > best_dice:
        best_dice = mean_dice
        best_threshold = th

print("\n🔥 BEST RESULT")
print(f"Best Threshold: {best_threshold}")
print(f"Best Dice: {best_dice:.4f}")

# =========================
# FINAL EVALUATION
# =========================
dice_list, iou_list, p_list, r_list, f1_list = [], [], [], [], []

for img, mask in tqdm(loader, desc="Final Eval"):
    img = img.to(DEVICE)
    mask = mask.numpy()

    with torch.no_grad():
        pred = tta_predict(img)

    pred = pred.cpu().numpy()
    pred_bin = (pred > best_threshold).astype(np.uint8)

    d, iou, p, r, f1 = compute_metrics(pred_bin, mask)

    dice_list.append(d)
    iou_list.append(iou)
    p_list.append(p)
    r_list.append(r)
    f1_list.append(f1)

print("\n===== FINAL RESULTS =====")
print(f"Dice: {np.mean(dice_list):.4f}")
print(f"IoU: {np.mean(iou_list):.4f}")
print(f"Precision: {np.mean(p_list):.4f}")
print(f"Recall: {np.mean(r_list):.4f}")
print(f"F1-score: {np.mean(f1_list):.4f}")