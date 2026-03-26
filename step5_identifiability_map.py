import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========================
# PATHS
# ========================
MEAN_DIR = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\uncertainty\mean"
EPI_DIR = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\uncertainty\epistemic"
ALE_DIR = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\uncertainty\aleatoric"

SAVE_PATH = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\identifiability.png"

# ========================
# LOAD FILES
# ========================
files = sorted(os.listdir(MEAN_DIR))[:4]

rows = []

for f in files:
    mean = cv2.imread(os.path.join(MEAN_DIR, f), 0) / 255.0
    epi = cv2.imread(os.path.join(EPI_DIR, f), 0) / 255.0
    ale = cv2.imread(os.path.join(ALE_DIR, f), 0) / 255.0

    uncertainty = epi + ale
    uncertainty = np.clip(uncertainty, 0, 1)

    ident = mean * (1 - uncertainty)

    rows.append((mean, uncertainty, ident))

# ========================
# PLOT
# ========================
fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3*len(rows)))

for i, (mean, unc, ident) in enumerate(rows):

    axes[i, 0].imshow(mean, cmap='gray', vmin=0, vmax=1)
    axes[i, 0].axis("off")

    im1 = axes[i, 1].imshow(unc, cmap='hot', vmin=0, vmax=1)
    axes[i, 1].axis("off")

    im2 = axes[i, 2].imshow(ident, cmap='jet', vmin=0, vmax=1)
    axes[i, 2].axis("off")

# ========================
# COLORBARS
# ========================
cbar_ax1 = fig.add_axes([0.92, 0.55, 0.015, 0.3])
fig.colorbar(im1, cax=cbar_ax1)

cbar_ax2 = fig.add_axes([0.92, 0.15, 0.015, 0.3])
fig.colorbar(im2, cax=cbar_ax2)

# ========================
# LAYOUT
# ========================
plt.subplots_adjust(left=0.02, right=0.9, wspace=0.05, hspace=0.08)

plt.savefig(SAVE_PATH, dpi=600, bbox_inches='tight')
plt.show()

print("🔥 Identifiability map saved!")