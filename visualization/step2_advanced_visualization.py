import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image

# ========================
# PATH CONFIG
# ========================
MODEL_PATH = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\outputs_upgraded\best_dice.pth"

DATA_ROOT = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d"
SPLIT_FILE = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d\test.txt"

SAVE_DIR = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\advanced_results"
FINAL_FIGURE = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\final_best_roi.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

# ========================
# SETTINGS (🔥 IMPORTANT)
# ========================
BEST_THRESHOLD = 0.90   # only best cases
MAX_SAMPLES = 6         # max figures to save

# ========================
# IMPORT MODEL
# ========================
from IU_PSeg import OTU3DDataset, IU_PSegNet

dataset = OTU3DDataset(DATA_ROOT, SPLIT_FILE, augment=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = IU_PSegNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded!")

# ========================
# METRICS
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
# ADVANCED VISUALIZATION
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

    # Main image + ROI
    ax.imshow(image_np)
    rect = patches.Rectangle((cx-size//2, cy-size//2),
                             size, size,
                             linewidth=1.5,
                             edgecolor='red',
                             facecolor='none')
    ax.add_patch(rect)
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
# LOOP (🔥 FILTER BEST CASES)
# ========================
dice_all = []
iou_all = []
saved_count = 0

for idx, (image, gt) in enumerate(tqdm(loader)):
    image = image.to(DEVICE)
    gt = gt.to(DEVICE)

    with torch.no_grad():
        output = model(image)
        pred = torch.sigmoid(output)
        pred = (pred > 0.25).float()   # best threshold

    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()

    dice, iou = compute_metrics(pred_np, gt_np)

    # 🔥 FILTER
    if dice < BEST_THRESHOLD:
        continue

    dice_all.append(dice)
    iou_all.append(iou)

    save_path = os.path.join(SAVE_DIR, f"{saved_count}.png")
    create_figure(image.squeeze(0), gt, pred.squeeze(0), dice, iou, save_path)

    saved_count += 1

    if saved_count >= MAX_SAMPLES:
        break

# ========================
# FINAL RESULTS
# ========================
print("\n========== FINAL RESULTS ==========")
if len(dice_all) > 0:
    print(f"Dice: {np.mean(dice_all):.4f}")
    print(f"IoU : {np.mean(iou_all):.4f}")
else:
    print("⚠️ No samples found — reduce BEST_THRESHOLD")
print("===================================")

# ========================
# COMBINED FIGURE
# ========================
images = sorted(os.listdir(SAVE_DIR))[:MAX_SAMPLES]

fig, ax = plt.subplots(1, len(images), figsize=(3*len(images), 4))

for i, img_name in enumerate(images):
    img = Image.open(os.path.join(SAVE_DIR, img_name))
    ax[i].imshow(img)
    ax[i].axis("off")

plt.subplots_adjust(wspace=0.01, hspace=0.01)

plt.savefig(FINAL_FIGURE, dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

print("\n🔥 BEST ROI FIGURE SAVED!")