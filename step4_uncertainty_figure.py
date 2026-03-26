import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ========================
# PATHS
# ========================
DATA_ROOT = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d"
VIS_DIR = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\uncertainty\vis"
EPI_DIR = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\uncertainty\epistemic"
ALE_DIR = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\uncertainty\aleatoric"

SAVE_PATH = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\uncertainty_final.png"

# ========================
# SELECT BEST SAMPLES
# ========================
files = sorted(os.listdir(VIS_DIR))[:4]  # top 4 samples

rows = []

for f in files:
    vis_img = Image.open(os.path.join(VIS_DIR, f))
    vis_np = np.array(vis_img)

    # split the panel (5 parts)
    w = vis_np.shape[1] // 5

    img = vis_np[:, 0*w:1*w]
    gt = vis_np[:, 1*w:2*w]
    pred = vis_np[:, 2*w:3*w]

    epi = cv2.imread(os.path.join(EPI_DIR, f))
    ale = cv2.imread(os.path.join(ALE_DIR, f))

    epi = cv2.cvtColor(epi, cv2.COLOR_BGR2RGB)
    ale = cv2.cvtColor(ale, cv2.COLOR_BGR2RGB)

    rows.append((img, gt, pred, epi, ale))

# ========================
# PLOT
# ========================
fig, axes = plt.subplots(len(rows), 5, figsize=(12, 3*len(rows)))

for i, (img, gt, pred, epi, ale) in enumerate(rows):

    axes[i, 0].imshow(img)
    axes[i, 0].axis("off")

    axes[i, 1].imshow(gt)
    axes[i, 1].axis("off")

    axes[i, 2].imshow(pred)
    axes[i, 2].axis("off")

    im1 = axes[i, 3].imshow(epi, vmin=0, vmax=255)
    axes[i, 3].axis("off")

    im2 = axes[i, 4].imshow(ale, vmin=0, vmax=255)
    axes[i, 4].axis("off")

# ========================
# COLORBARS (SIDE)
# ========================
cbar_ax1 = fig.add_axes([0.92, 0.55, 0.015, 0.3])
cbar1 = fig.colorbar(im1, cax=cbar_ax1)
cbar1.set_label("Epistemic")

cbar_ax2 = fig.add_axes([0.92, 0.15, 0.015, 0.3])
cbar2 = fig.colorbar(im2, cax=cbar_ax2)
cbar2.set_label("Aleatoric")

# ========================
# SPACING
# ========================
plt.subplots_adjust(
    left=0.02,
    right=0.9,
    top=0.98,
    bottom=0.02,
    wspace=0.05,
    hspace=0.08
)

plt.savefig(SAVE_PATH, dpi=600, bbox_inches='tight')
plt.show()

print("🔥 Uncertainty figure saved (600 DPI)")