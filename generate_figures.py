import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix
from torch.utils.data import DataLoader

from IU_PSeg import OTU3DDataset, IU_PSegNet

# =========================
# PATHS
# =========================
MODEL_PATH = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\outputs_upgraded\best_dice.pth"
DATA_ROOT = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d"
TEST_SPLIT = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d\test.txt"

SAVE_DIR = "final_figures"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# LOAD DATA
# =========================
dataset = OTU3DDataset(DATA_ROOT, TEST_SPLIT, augment=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# =========================
# LOAD MODEL
# =========================
model = IU_PSegNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded")

# =========================
# TTA
# =========================
def tta_predict(image):
    p1 = torch.sigmoid(model(image))

    img_h = torch.flip(image, dims=[3])
    p2 = torch.sigmoid(model(img_h))
    p2 = torch.flip(p2, dims=[3])

    img_v = torch.flip(image, dims=[2])
    p3 = torch.sigmoid(model(img_v))
    p3 = torch.flip(p3, dims=[2])

    img_d = torch.flip(image, dims=[2,3])
    p4 = torch.sigmoid(model(img_d))
    p4 = torch.flip(p4, dims=[2,3])

    return (p1 + p2 + p3 + p4) / 4.0

# =========================
# COLLECT DATA
# =========================
y_true = []
y_scores = []
y_pred = []

qualitative_samples = []

for i, (img, mask) in enumerate(tqdm(loader)):
    img = img.to(DEVICE)
    mask_np = mask.numpy()[0,0]

    with torch.no_grad():
        pred = tta_predict(img)

    pred_np = pred.cpu().numpy()[0,0]

    # Flatten for ROC
    y_true.extend(mask_np.flatten())
    y_scores.extend(pred_np.flatten())

    pred_bin = (pred_np > 0.25).astype(np.uint8)
    y_pred.extend(pred_bin.flatten())

    # Save few samples for qualitative
    if len(qualitative_samples) < 5:
        qualitative_samples.append((img.cpu().numpy()[0], mask_np, pred_bin))

# =========================
# 1️⃣ ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", linewidth=2)
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()

plt.savefig(os.path.join(SAVE_DIR, "roc_curve.png"), dpi=600, bbox_inches='tight')
plt.close()

print("✅ ROC saved")

# =========================
# 2️⃣ CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red')

plt.xlabel("Predicted")
plt.ylabel("Ground Truth")

plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=600, bbox_inches='tight')
plt.close()

print("✅ Confusion matrix saved")

# =========================
# 3️⃣ QUALITATIVE RESULTS
# =========================
fig, axes = plt.subplots(len(qualitative_samples), 3, figsize=(9, 3*len(qualitative_samples)))

for i, (img, gt, pred) in enumerate(qualitative_samples):
    img = img.transpose(1,2,0)

    axes[i,0].imshow(img)
    axes[i,0].set_title("Input")
    axes[i,0].axis("off")

    axes[i,1].imshow(gt, cmap='gray')
    axes[i,1].set_title("Ground Truth")
    axes[i,1].axis("off")

    axes[i,2].imshow(pred, cmap='gray')
    axes[i,2].set_title("Prediction")
    axes[i,2].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "qualitative_results.png"), dpi=600)
plt.close()

print("✅ Qualitative results saved")

print("\n🔥 ALL FIGURES GENERATED (600 DPI)")