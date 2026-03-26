import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from IU_PSeg import OTU3DDataset, IU_PSegNet

# ========================
# PATHS
# ========================
MODEL_PATH = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\outputs_upgraded\best_dice.pth"
DATA_ROOT = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d"
SPLIT_FILE = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d\test.txt"

SAVE_PATH = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\final_best_overlay.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# LOAD
# ========================
dataset = OTU3DDataset(DATA_ROOT, SPLIT_FILE, augment=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = IU_PSegNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded")

# ========================
# METRIC
# ========================
def compute_dice(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    TP = np.sum((pred == 1) & (gt == 1))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))
    return (2 * TP) / (2 * TP + FP + FN + 1e-6)

# ========================
# OVERLAY FUNCTION
# ========================
def create_overlay(image, mask):
    overlay = image.copy()

    # Red overlay for prediction
    overlay[mask == 1] = [1, 0, 0]  # red

    # Blend
    blended = 0.7 * image + 0.3 * overlay
    return blended

# ========================
# SELECT BEST SAMPLES
# ========================
best_samples = []

for img, gt in loader:
    img = img.to(DEVICE)
    gt = gt.to(DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(model(img))
        pred = (pred > 0.25).float()

    dice = compute_dice(pred.cpu().numpy(), gt.cpu().numpy())

    if dice > 0.90:
        best_samples.append((img.cpu(), gt.cpu(), pred.cpu(), dice))

    if len(best_samples) >= 6:
        break

# ========================
# PLOT FIGURE
# ========================
fig, axes = plt.subplots(len(best_samples), 4, figsize=(12, 3*len(best_samples)))

for i, (img, gt, pred, dice) in enumerate(best_samples):

    img_np = img[0].permute(1,2,0).numpy()
    gt_np = gt[0,0].numpy()
    pred_np = pred[0,0].numpy()

    overlay = create_overlay(img_np, pred_np)

    # Input
    axes[i,0].imshow(img_np)
    axes[i,0].set_title(f"Input (Dice={dice:.3f})")
    axes[i,0].axis("off")

    # GT
    axes[i,1].imshow(gt_np, cmap='gray')
    axes[i,1].set_title("Ground Truth")
    axes[i,1].axis("off")

    # Prediction
    axes[i,2].imshow(pred_np, cmap='gray')
    axes[i,2].set_title("Prediction")
    axes[i,2].axis("off")

    # Overlay
    axes[i,3].imshow(overlay)
    axes[i,3].set_title("Overlay")
    axes[i,3].axis("off")

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=600, bbox_inches='tight')
plt.show()

print("🔥 Overlay figure saved (600 DPI)")