import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========================
# PATHS
# ========================
MEAN_DIR = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\uncertainty\mean"
EPI_DIR  = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\uncertainty\epistemic"
ALE_DIR  = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\uncertainty\aleatoric"

SAVE_PATH = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\identifiability_final.png"

# ========================
# LOAD FILES
# ========================
files = sorted(os.listdir(MEAN_DIR))

selected = []

print("Selecting best samples...")

for f in files:

    mean = cv2.imread(os.path.join(MEAN_DIR, f), 0)
    epi  = cv2.imread(os.path.join(EPI_DIR, f), 0)
    ale  = cv2.imread(os.path.join(ALE_DIR, f), 0)

    if mean is None or epi is None or ale is None:
        continue

    # normalize to [0,1]
    mean = mean.astype(np.float32) / 255.0
    epi  = epi.astype(np.float32) / 255.0
    ale  = ale.astype(np.float32) / 255.0

    # ------------------------
    # FIX 1: Normalize maps
    # ------------------------
    mean = (mean - mean.min()) / (mean.max() + 1e-6)

    uncertainty = (epi + ale) / 2.0
    uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() + 1e-6)

    # ------------------------
    # FIX 2: Identifiability
    # ------------------------
    ident = mean * (1 - uncertainty)
    ident = np.clip(ident, 0, 1)

    # ------------------------
    # FIX 3: FILTER BAD SAMPLES
    # ------------------------
    if mean.mean() < 0.05:
        continue
    if ident.mean() < 0.02:
        continue

    selected.append((mean, uncertainty, ident))

# take best 4 samples
selected = selected[:4]

print(f"Selected {len(selected)} good samples")

# ========================
# PLOT
# ========================
rows = len(selected)

fig, axes = plt.subplots(rows, 3, figsize=(9, 3*rows))

for i, (mean, unc, ident) in enumerate(selected):

    # Mean prediction
    axes[i, 0].imshow(mean, cmap='gray', vmin=0, vmax=1)
    axes[i, 0].axis("off")

    # Uncertainty
    im1 = axes[i, 1].imshow(unc, cmap='hot', vmin=0, vmax=1)
    axes[i, 1].axis("off")

    # Identifiability
    im2 = axes[i, 2].imshow(ident, cmap='jet', vmin=0, vmax=1)
    axes[i, 2].axis("off")

# ========================
# COLORBARS (SIDE ONLY)
# ========================
cbar_ax1 = fig.add_axes([0.92, 0.55, 0.015, 0.3])
cbar1 = fig.colorbar(im1, cax=cbar_ax1)
cbar1.set_label("Uncertainty", fontsize=10)

cbar_ax2 = fig.add_axes([0.92, 0.15, 0.015, 0.3])
cbar2 = fig.colorbar(im2, cax=cbar_ax2)
cbar2.set_label("Identifiability", fontsize=10)

# ========================
# LAYOUT FIX (IMPORTANT)
# ========================
plt.subplots_adjust(
    left=0.02,
    right=0.9,
    top=0.98,
    bottom=0.02,
    wspace=0.02,
    hspace=0.05
)

# ========================
# SAVE (600 DPI)
# ========================
plt.savefig(SAVE_PATH, dpi=600, bbox_inches='tight')
plt.show()

print("🔥 Identifiability figure saved (600 DPI)")