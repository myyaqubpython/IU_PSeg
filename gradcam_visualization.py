import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from IU_PSeg import OTU3DDataset, IU_PSegNet

# ========================
# PATHS
# ========================
MODEL_PATH = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\outputs_upgraded\best_dice.pth"
DATA_ROOT = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d"
SPLIT_FILE = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d\test.txt"

SAVE_PATH = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\gradcam_final.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# LOAD MODEL
# ========================
model = IU_PSegNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded")

# ========================
# AUTO FIND TARGET LAYER
# ========================
target_layer = None
for m in reversed(list(model.modules())):
    if isinstance(m, torch.nn.Conv2d):
        target_layer = m
        break

print("🔥 Using layer:", target_layer)

# ========================
# HOOK STORAGE
# ========================
activations = []
gradients = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# ========================
# DATA
# ========================
dataset = OTU3DDataset(DATA_ROOT, SPLIT_FILE, augment=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ========================
# GRAD-CAM FUNCTION
# ========================
def generate_gradcam(input_tensor):
    activations.clear()
    gradients.clear()

    output = model(input_tensor)

    # segmentation objective (global activation)
    target = torch.mean(output)

    model.zero_grad()
    target.backward()

    grads = gradients[0].detach().cpu().numpy()[0]
    acts = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)

    # resize to image
    cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return cam

# ========================
# GENERATE SAMPLES
# ========================
samples = []

for img, _ in loader:
    img = img.to(DEVICE)

    cam = generate_gradcam(img)

    img_np = img[0].permute(1, 2, 0).cpu().numpy()

    # heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap / 255.0

    # overlay
    overlay = 0.6 * img_np + 0.4 * heatmap

    samples.append((img_np, heatmap, overlay))

    if len(samples) >= 4:
        break

# ========================
# PLOT FIGURE (CLEAN)
# ========================
fig, axes = plt.subplots(len(samples), 3, figsize=(9, 3 * len(samples)))

for i, (img, heat, overlay) in enumerate(samples):

    axes[i, 0].imshow(img)
    axes[i, 0].axis("off")

    im = axes[i, 1].imshow(heat, vmin=0, vmax=1)
    axes[i, 1].axis("off")

    axes[i, 2].imshow(overlay)
    axes[i, 2].axis("off")

# ========================
# 🔥 CREATE SIDE COLORBAR AXIS (NOT ON IMAGES)
# ========================
cbar_ax = fig.add_axes([1, 0.10, 0.015, 0.7])  
# [left, bottom, width, height]

cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)


# 🔥 remove gaps completely
plt.subplots_adjust(left=0, right=0.98, top=0.98, bottom=0, wspace=0.001, hspace=0.001)

plt.savefig(SAVE_PATH, dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

print("🔥 Grad-CAM figure saved (600 DPI)")