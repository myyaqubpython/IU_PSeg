import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image

from IU_PSeg import OTU3DDataset, IU_PSegNet

# ========================
# PATHS
# ========================
MODEL_PATH = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\outputs_upgraded\best_dice.pth"

DATA_ROOT = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d"
SPLIT_FILE = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d\test.txt"

SAVE_DIR = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\roi_cases"
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_FIG = os.path.join(SAVE_DIR, "best_cases.png")
WORST_FIG = os.path.join(SAVE_DIR, "failure_cases.png")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# SETTINGS
# ========================
BEST_THRESHOLD = 0.90
WORST_THRESHOLD = 0.50
MAX_SAMPLES = 6

# ========================
# LOAD DATA & MODEL
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
def compute_metrics(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()

    TP = np.sum((pred == 1) & (gt == 1))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)
    iou = TP / (TP + FP + FN + 1e-6)

    return dice, iou

# ========================
# VISUALIZATION FUNCTION
# ========================
def create_figure(image, gt, pred, dice, iou, save_path):
    image_np = image.permute(1, 2, 0).cpu().numpy()
    gt_np = gt.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy()

    error_map = np.abs(gt_np - pred_np)

    h, w = gt_np.shape
    cx, cy = w // 2, h // 2
    size = 60

    fig, ax = plt.subplots(figsize=(5, 7))

    # Main image
    ax.imshow(image_np)

    # ROI box
    rect = patches.Rectangle(
        (cx-size//2, cy-size//2),
        size, size,
        linewidth=1.5,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)

    # 🔥 Always show Dice + IoU
    ax.text(
        5, 20,
        f'Dice: {dice:.3f}\nIoU: {iou:.3f}',
        color='yellow',
        fontsize=10,
        fontweight='bold',
        bbox=dict(facecolor='black', alpha=0.6, pad=3)
    )

    ax.axis("off")

    # ROI Prediction
    ax_in1 = fig.add_axes([0.05, 0.05, 0.4, 0.25])
    zoom_pred = pred_np[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
    ax_in1.imshow(zoom_pred, cmap='gray')
    ax_in1.axis("off")

    # Error Map
    ax_in2 = fig.add_axes([0.5, 0.05, 0.4, 0.25])
    zoom_err = error_map[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
    im = ax_in2.imshow(zoom_err, cmap='hot')
    ax_in2.axis("off")

    fig.colorbar(im, ax=ax_in2, fraction=0.046, pad=0.04)

    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

# ========================
# COLLECT SAMPLES
# ========================
best_imgs = []
worst_imgs = []

for idx, (image, gt) in enumerate(tqdm(loader)):
    image = image.to(DEVICE)
    gt = gt.to(DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(model(image))
        pred = (pred > 0.25).float()

    dice, iou = compute_metrics(pred.cpu().numpy(), gt.cpu().numpy())

    if dice > BEST_THRESHOLD and len(best_imgs) < MAX_SAMPLES:
        best_imgs.append((image.squeeze(0), gt, pred.squeeze(0), dice, iou))

    if dice < WORST_THRESHOLD and len(worst_imgs) < MAX_SAMPLES:
        worst_imgs.append((image.squeeze(0), gt, pred.squeeze(0), dice, iou))

# ========================
# COMBINE FIGURES (FIXED)
# ========================
def make_combined_figure(samples, save_path):
    if len(samples) == 0:
        print(f"⚠️ No samples for {save_path}")
        return

    fig, ax = plt.subplots(1, len(samples), figsize=(3*len(samples), 4))

    # 🔥 FIX: always iterable
    if len(samples) == 1:
        ax = [ax]

    for i, (img, gt, pred, dice, iou) in enumerate(samples):
        temp_path = os.path.join(SAVE_DIR, f"temp_{i}.png")
        create_figure(img, gt, pred, dice, iou, temp_path)

        im = Image.open(temp_path)
        ax[i].imshow(im)
        ax[i].axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.001)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

# ========================
# SAVE FIGURES
# ========================
make_combined_figure(best_imgs, BEST_FIG)
make_combined_figure(worst_imgs, WORST_FIG)

print("\n🔥 DONE: Best & Failure figures saved (600 DPI)")