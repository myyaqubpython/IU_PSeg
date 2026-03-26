import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import DataLoader

# ========================
# PATH CONFIG (YOUR PATHS)
# ========================
MODEL_PATH = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\outputs\best_dice.pth"

DATA_ROOT = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d"
SPLIT_FILE = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d\train.txt"

SAVE_DIR = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

# ========================
# IMPORT FROM YOUR FILE
# ========================
from IU_PSeg import OTU3DDataset, UNet

# ========================
# LOAD DATA
# ========================
dataset = OTU3DDataset(DATA_ROOT, SPLIT_FILE, augment=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ========================
# LOAD MODEL
# ========================
model = UNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully!")

# ========================
# METRICS
# ========================
def dice_score(pred, gt):
    smooth = 1e-6
    intersection = np.sum(pred * gt)
    return (2. * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)

def iou_score(pred, gt):
    smooth = 1e-6
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return (intersection + smooth) / (union + smooth)

# ========================
# VISUALIZATION
# ========================
def visualize(image, gt, pred, save_path):
    image = image.permute(1, 2, 0).cpu().numpy()
    gt = gt.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(image)
    ax[0].set_title("Input")
    ax[0].axis("off")

    ax[1].imshow(gt, cmap="gray")
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    ax[2].imshow(pred, cmap="gray")
    ax[2].set_title("Prediction")
    ax[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ========================
# INFERENCE LOOP
# ========================
dice_list = []
iou_list = []

for idx, (image, gt) in enumerate(tqdm(loader)):
    image = image.to(DEVICE)
    gt = gt.to(DEVICE)

    with torch.no_grad():
        output = model(image)
        pred = torch.sigmoid(output)
        pred = (pred > 0.5).float()

    # Metrics
    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()

    d = dice_score(pred_np, gt_np)
    i = iou_score(pred_np, gt_np)

    dice_list.append(d)
    iou_list.append(i)

    # Save visualization
    save_path = os.path.join(SAVE_DIR, f"{idx}.png")
    visualize(image.squeeze(0), gt, pred.squeeze(0), save_path)

# ========================
# FINAL RESULTS
# ========================
print("\n==========================")
print(f"Average Dice: {np.mean(dice_list):.4f}")
print(f"Average IoU : {np.mean(iou_list):.4f}")
print("==========================")