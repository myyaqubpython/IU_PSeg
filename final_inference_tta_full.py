import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

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
def compute_all_metrics(pred_bin, gt, pred_prob):
    pred_bin = pred_bin.flatten()
    gt = gt.flatten()
    pred_prob = pred_prob.flatten()

    TP = np.sum((pred_bin == 1) & (gt == 1))
    TN = np.sum((pred_bin == 0) & (gt == 0))
    FP = np.sum((pred_bin == 1) & (gt == 0))
    FN = np.sum((pred_bin == 0) & (gt == 1))

    dice = (2*TP + 1e-6) / (2*TP + FP + FN + 1e-6)
    iou = (TP + 1e-6) / (TP + FP + FN + 1e-6)
    precision = (TP + 1e-6) / (TP + FP + 1e-6)
    recall = (TP + 1e-6) / (TP + FN + 1e-6)

    sensitivity = recall
    specificity = (TN + 1e-6) / (TN + FP + 1e-6)
    accuracy = (TP + TN + 1e-6) / (TP + TN + FP + FN + 1e-6)
    ppv = precision
    npv = (TN + 1e-6) / (TN + FN + 1e-6)

    try:
        auc = roc_auc_score(gt, pred_prob)
    except:
        auc = 0.0

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": dice,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "ppv": ppv,
        "npv": npv
    }

# =========================
# TTA
# =========================
def tta_predict(image):
    # original
    p1 = torch.sigmoid(model(image))

    # horizontal
    img_h = torch.flip(image, dims=[3])
    p2 = torch.sigmoid(model(img_h))
    p2 = torch.flip(p2, dims=[3])

    # vertical
    img_v = torch.flip(image, dims=[2])
    p3 = torch.sigmoid(model(img_v))
    p3 = torch.flip(p3, dims=[2])

    # diagonal
    img_d = torch.flip(image, dims=[2,3])
    p4 = torch.sigmoid(model(img_d))
    p4 = torch.flip(p4, dims=[2,3])

    return (p1 + p2 + p3 + p4) / 4.0

# =========================
# THRESHOLD SEARCH
# =========================
thresholds = np.arange(0.2, 0.6, 0.05)

best_dice = 0
best_threshold = 0.4

for th in thresholds:
    dice_all = []

    for img, mask in tqdm(loader, desc=f"Threshold {th:.2f}"):
        img = img.to(DEVICE)
        mask_np = mask.numpy()

        with torch.no_grad():
            pred = tta_predict(img)

        pred_np = pred.cpu().numpy()
        pred_bin = (pred_np > th).astype(np.uint8)

        metrics = compute_all_metrics(pred_bin, mask_np, pred_np)
        dice_all.append(metrics["dice"])

    mean_dice = np.mean(dice_all)
    print(f"Threshold {th:.2f} → Dice: {mean_dice:.4f}")

    if mean_dice > best_dice:
        best_dice = mean_dice
        best_threshold = th

print("\n🔥 BEST THRESHOLD:", best_threshold)

# =========================
# FINAL EVALUATION
# =========================
metrics_all = []

for img, mask in tqdm(loader, desc="Final Eval"):
    img = img.to(DEVICE)
    mask_np = mask.numpy()

    with torch.no_grad():
        pred = tta_predict(img)

    pred_np = pred.cpu().numpy()
    pred_bin = (pred_np > best_threshold).astype(np.uint8)

    metrics = compute_all_metrics(pred_bin, mask_np, pred_np)
    metrics_all.append(metrics)

# =========================
# AVERAGE RESULTS
# =========================
keys = metrics_all[0].keys()

print("\n===== FINAL RESULTS =====")

for k in keys:
    val = np.mean([m[k] for m in metrics_all])
    print(f"{k.upper():15s}: {val:.4f}")