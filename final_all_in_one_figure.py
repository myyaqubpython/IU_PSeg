import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ========================
# PATHS
# ========================
MODEL_PATH = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\outputs_upgraded\best_dice.pth"
DATA_ROOT  = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d"
SPLIT_TXT  = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d\test.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# IMPORT
# ========================
from IU_PSeg import OTU3DDataset, IU_PSegNet

dataset = OTU3DDataset(DATA_ROOT, SPLIT_TXT, augment=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ========================
# MODEL
# ========================
model = IU_PSegNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded")

# ========================
# GRAD-CAM SETUP
# ========================
target_layer = None
for name, m in model.named_modules():
    if isinstance(m, torch.nn.Conv2d):
        target_layer = m

activations = []
gradients = []

def forward_hook(module, input, output):
    activations.clear()
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.clear()
    gradients.append(grad_output[0])

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# ========================
# FUNCTIONS
# ========================
def normalize(x):
    return (x - x.min()) / (x.max() + 1e-6)

def get_gradcam(img):
    img = img.clone()
    img.requires_grad = True

    output = model(img)
    loss = output.mean()

    model.zero_grad()
    loss.backward()

    act = activations[0].detach().cpu().numpy()[0]
    grad = gradients[0].detach().cpu().numpy()[0]

    weights = np.mean(grad, axis=(1,2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img.shape[3], img.shape[2]))
    cam = normalize(cam)

    return cam

# ========================
# FIXED OVERLAY
# ========================
def overlay(image, mask):
    mask = (mask > 0).astype(np.uint8)
    mask_3 = np.stack([mask]*3, axis=-1)

    image = np.where(
        mask_3 == 1,
        image * 0.3 + np.array([0,255,0]) * 0.7,
        image
    )

    return image.astype(np.uint8)

# ========================
# FIXED IMAGE CONVERSION
# ========================
def convert_image(image):
    img_np = image.detach().cpu().numpy()[0]

    if img_np.shape[0] == 1:
        img_np = img_np[0]
        img_np = np.stack([img_np]*3, axis=-1)

    elif img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1,2,0))

    img_np = (img_np * 255).astype(np.uint8)

    return img_np

# ========================
# SAMPLE COLLECTION
# ========================
samples = []

for image, gt in loader:

    image = image.to(DEVICE)
    gt = gt.to(DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(model(image))

    pred_bin = (pred > 0.25).float()

    # SAFE conversion
    mean = pred.detach().cpu().numpy()[0][0]
    gt_np = gt.detach().cpu().numpy()[0][0]

    mean = normalize(mean)

    # uncertainty
    unc = cv2.GaussianBlur(mean, (21,21), 0)
    unc = normalize(unc)

    # identifiability
    ident = mean * (1 - unc)
    ident = np.clip(ident, 0, 1)

    # filter bad samples
    if mean.mean() < 0.05:
        continue
    if ident.mean() < 0.02:
        continue

    # Grad-CAM
    cam = get_gradcam(image)

    # image conversion (FIXED)
    img_np = convert_image(image)

    pred_np = pred_bin.detach().cpu().numpy()[0][0]

    ov = overlay(img_np.copy(), pred_np)

    samples.append((img_np, gt_np, pred_np, ov, cam, unc, ident))

# take best samples
samples = samples[:4]

print(f"Selected {len(samples)} samples")

# ========================
# PLOT
# ========================
rows = len(samples)
cols = 7

fig, axes = plt.subplots(rows, cols, figsize=(14, 3*rows))

for i, (img, gt, pred, ov, cam, unc, ident) in enumerate(samples):

    axes[i,0].imshow(img); axes[i,0].axis("off")
    axes[i,1].imshow(gt, cmap='gray'); axes[i,1].axis("off")
    axes[i,2].imshow(pred, cmap='gray'); axes[i,2].axis("off")
    axes[i,3].imshow(ov); axes[i,3].axis("off")

    axes[i,4].imshow(cam, cmap='jet'); axes[i,4].axis("off")

    im1 = axes[i,5].imshow(unc, cmap='hot', vmin=0, vmax=1)
    axes[i,5].axis("off")

    im2 = axes[i,6].imshow(ident, cmap='jet', vmin=0, vmax=1)
    axes[i,6].axis("off")

# ========================
# COLORBARS
# ========================
cbar_ax1 = fig.add_axes([0.92, 0.55, 0.01, 0.3])
fig.colorbar(im1, cax=cbar_ax1)

cbar_ax2 = fig.add_axes([0.92, 0.15, 0.01, 0.3])
fig.colorbar(im2, cax=cbar_ax2)

# ========================
# LAYOUT
# ========================
plt.subplots_adjust(
    left=0.02,
    right=0.9,
    wspace=0.01,
    hspace=0.04
)

# ========================
# SAVE
# ========================
SAVE_PATH = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\FINAL_ALL_IN_ONE_FINAL_CLEAN.png"

plt.savefig(SAVE_PATH, dpi=600, bbox_inches='tight')
plt.show()

print("🔥 FINAL FIGURE GENERATED SUCCESSFULLY (600 DPI)")