
# ============================================================
# IU-PSeg for USOV3D VTK Dataset - V3
# Main V3 updates:
# 1) 5-fold case-level cross-validation
# 2) Stronger but safe ultrasound augmentations
# 3) Saves best-Dice and best-Reliability checkpoints per fold
# 4) Reports per-class metrics for F-mask and O-mask
# 5) Uses balanced positive/background slice sampling
# 6) Optional ensemble external-test inference using all folds
# 7) IU-PSeg maps: mask, aleatoric, epistemic, total uncertainty,
#    identifiability, boundary response, voxel reliability, reliability score
# ============================================================

import os
import re
import csv
import json
import random
import warnings
from functools import lru_cache

import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk
from skimage.segmentation import find_boundaries

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

ROOT = r"C:\Users\M YAQUB\CSU2026\USOV3D"

TRAIN_DIR = os.path.join(ROOT, "Training_Set_2019", "Training_Set_1")
TEST_DIR  = os.path.join(ROOT, "Test_Set_2019", "Test_Set_1")

SAVE_DIR = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\USOV3D_RESULTS_V3"
os.makedirs(SAVE_DIR, exist_ok=True)

CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
PRED_DIR = os.path.join(SAVE_DIR, "predictions")
MAP_DIR = os.path.join(SAVE_DIR, "iupseg_maps")
METRIC_DIR = os.path.join(SAVE_DIR, "metrics")

for d in [CHECKPOINT_DIR, PRED_DIR, MAP_DIR, METRIC_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 120
PATIENCE = 30

LR = 1e-4
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 2                 # channel 0 = f, channel 1 = o
N_FOLDS = 5

THRESHOLD = 0.5
REVIEW_THRESHOLD = 0.55

POSITIVE_MIN_PIXELS = 20
NEGATIVE_PER_POSITIVE = 1.25    # keep background slices but not too many

MC_SAMPLES = 10                 # use 10 for speed; change to 20 for final maps
RUN_EXTERNAL_ENSEMBLE = True    # after CV, run external inference using fold best-Dice models
SAVE_VISUALS = True             # saves 600 dpi map figures
MAX_VIS_SLICES_PER_CASE = 10

# Optional warm-start checkpoint. Usually keep None for clean 5-fold CV.
RESUME_CHECKPOINT = None
# Example:
# RESUME_CHECKPOINT = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\USOV3D_RESULTS_V2\checkpoints\best_iupseg_usov3d_v2.pth"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
torch.backends.cudnn.benchmark = True


# ============================================================
# VTK READER
# ============================================================

def read_vtk_volume(path):
    mesh = pv.read(path)

    if len(mesh.point_data.keys()) > 0:
        key = list(mesh.point_data.keys())[0]
        arr = np.asarray(mesh.point_data[key])
    elif len(mesh.cell_data.keys()) > 0:
        key = list(mesh.cell_data.keys())[0]
        arr = np.asarray(mesh.cell_data[key])
    else:
        raise ValueError(f"No scalar data found in {path}")

    dims = mesh.dimensions
    if len(dims) != 3:
        raise ValueError(f"Unexpected VTK dimensions in {path}: {dims}")

    # USOV3D diagnostic showed shape as [D, H, W]
    vol = arr.reshape((dims[2], dims[1], dims[0]))
    return vol


def get_base_cases(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".vtk")]
    cases = []
    for f in files:
        if re.match(r"vol\d+\.vtk$", f):
            cases.append(f.replace(".vtk", ""))
    return sorted(cases, key=lambda x: int(x.replace("vol", "")))


@lru_cache(maxsize=96)
def load_case_image(folder, case_id):
    img_path = os.path.join(folder, f"{case_id}.vtk")
    img = read_vtk_volume(img_path).astype(np.float32)
    img = np.clip(img, 0, 255) / 255.0
    return img


@lru_cache(maxsize=96)
def load_case_mask(folder, case_id):
    f_r1 = (read_vtk_volume(os.path.join(folder, f"{case_id}_f_r1.vtk")) > 0).astype(np.float32)
    f_r2 = (read_vtk_volume(os.path.join(folder, f"{case_id}_f_r2.vtk")) > 0).astype(np.float32)
    o_r1 = (read_vtk_volume(os.path.join(folder, f"{case_id}_o_r1.vtk")) > 0).astype(np.float32)
    o_r2 = (read_vtk_volume(os.path.join(folder, f"{case_id}_o_r2.vtk")) > 0).astype(np.float32)

    # Soft labels keep inter-rater variability: 0, 0.5, 1.
    f_soft = (f_r1 + f_r2) / 2.0
    o_soft = (o_r1 + o_r2) / 2.0

    mask = np.stack([f_soft, o_soft], axis=0)  # [2, D, H, W]
    return mask.astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class USOV3DSliceDataset(Dataset):
    def __init__(
        self,
        folder,
        cases,
        image_size=256,
        train=True,
        balanced=True,
        negative_per_positive=1.25,
        positive_min_pixels=20
    ):
        self.folder = folder
        self.cases = list(cases)
        self.image_size = image_size
        self.train = train

        positive = []
        negative = []

        for case in self.cases:
            img = load_case_image(folder, case)
            mask = load_case_mask(folder, case)
            D = img.shape[0]

            for z in range(D):
                pix = float((mask[:, z] > 0.5).sum())
                if pix >= positive_min_pixels:
                    positive.append((case, z, 1))
                else:
                    negative.append((case, z, 0))

        if train and balanced:
            random.shuffle(negative)
            n_neg = min(int(len(positive) * negative_per_positive), len(negative))
            self.index = positive + negative[:n_neg]
            random.shuffle(self.index)
        else:
            self.index = positive + negative

        self.n_pos = sum([x[2] for x in self.index])
        self.n_neg = len(self.index) - self.n_pos

        print(
            f"Dataset: {len(self.index)} slices | cases={len(self.cases)} | "
            f"positive={self.n_pos}, negative={self.n_neg}, train={train}"
        )

    def __len__(self):
        return len(self.index)

    def random_aug(self, img, mask):
        # flips
        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=2).copy()

        if random.random() < 0.5:
            img = np.flip(img, axis=0).copy()
            mask = np.flip(mask, axis=1).copy()

        # 90-degree rotation
        if random.random() < 0.35:
            k = random.choice([1, 2, 3])
            img = np.rot90(img, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(1, 2)).copy()

        # intensity augmentation for ultrasound
        if random.random() < 0.45:
            factor = random.uniform(0.80, 1.20)
            bias = random.uniform(-0.07, 0.07)
            img = np.clip(img * factor + bias, 0, 1)

        # mild gaussian speckle/noise
        if random.random() < 0.35:
            noise = np.random.normal(0, random.uniform(0.005, 0.025), size=img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)

        # gamma correction
        if random.random() < 0.25:
            gamma = random.uniform(0.75, 1.35)
            img = np.clip(img, 1e-6, 1.0) ** gamma

        return img, mask

    def __getitem__(self, idx):
        case, z, _ = self.index[idx]

        img3d = load_case_image(self.folder, case)
        mask3d = load_case_mask(self.folder, case)

        img = img3d[z]       # [H, W]
        mask = mask3d[:, z]  # [2, H, W]

        if self.train:
            img, mask = self.random_aug(img, mask)

        img = torch.from_numpy(img).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float()

        img = F.interpolate(
            img.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        mask = F.interpolate(
            mask.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest"
        ).squeeze(0)

        return img, mask, case, z


# ============================================================
# MODEL
# ============================================================

def norm_layer(ch):
    groups = 8 if ch % 8 == 0 else 1
    return nn.GroupNorm(groups, ch)


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.fc1 = nn.Conv2d(ch, mid, 1)
        self.fc2 = nn.Conv2d(mid, ch, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ResDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            norm_layer(out_ch),
            SEBlock(out_ch),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        y = self.conv(x) + self.shortcut(x)
        y = F.relu(y, inplace=True)
        return self.drop(y)


class IUPSegNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, base=32, dropout=0.15):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.enc1 = ResDoubleConv(in_ch, base, dropout)
        self.enc2 = ResDoubleConv(base, base * 2, dropout)
        self.enc3 = ResDoubleConv(base * 2, base * 4, dropout)
        self.enc4 = ResDoubleConv(base * 4, base * 8, dropout)

        self.bottleneck = ResDoubleConv(base * 8, base * 16, dropout)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.dec4 = ResDoubleConv(base * 16, base * 8, dropout)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = ResDoubleConv(base * 8, base * 4, dropout)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = ResDoubleConv(base * 4, base * 2, dropout)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = ResDoubleConv(base * 2, base, dropout)

        self.seg_head = nn.Conv2d(base, out_ch, 1)
        self.logvar_head = nn.Conv2d(base, out_ch, 1)
        self.boundary_head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.seg_head(d1)
        logvar = self.logvar_head(d1)
        boundary = self.boundary_head(d1)

        return logits, logvar, boundary


# ============================================================
# LOSSES AND METRICS
# ============================================================

def focal_bce_loss(logits, targets, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal = (alpha * targets + (1.0 - alpha) * (1.0 - targets)) * (1.0 - pt).pow(gamma) * bce
    return focal.mean()


def soft_dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = torch.sum(probs * targets, dims)
    union = torch.sum(probs + targets, dims)
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def tversky_loss(logits, targets, alpha=0.35, beta=0.65, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    tp = torch.sum(probs * targets, dims)
    fp = torch.sum(probs * (1.0 - targets), dims)
    fn = torch.sum((1.0 - probs) * targets, dims)
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tversky.mean()


def make_boundary_from_mask(mask):
    mask_np = (mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
    out = np.zeros_like(mask_np, dtype=np.float32)

    for b in range(mask_np.shape[0]):
        for c in range(mask_np.shape[1]):
            out[b, c] = find_boundaries(mask_np[b, c], mode="outer").astype(np.float32)

    return torch.from_numpy(out).to(mask.device)


def total_loss(logits, logvar, boundary_logits, masks):
    probs = torch.sigmoid(logits)

    bce = focal_bce_loss(logits, masks)
    dice = soft_dice_loss(logits, masks)
    tv = tversky_loss(logits, masks)

    boundary_target = make_boundary_from_mask(masks)
    boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target)

    # Empty-slice false-positive penalty.
    empty_slices = (masks.sum(dim=(1, 2, 3)) < 1.0)
    if empty_slices.any():
        empty_penalty = probs[empty_slices].mean()
    else:
        empty_penalty = torch.tensor(0.0, device=logits.device)

    # Aleatoric uncertainty regularization.
    alea = F.softplus(logvar)
    uncertainty_reg = torch.mean(alea * torch.abs(probs - masks) + 0.001 * alea)

    loss = (
        bce
        + 0.65 * dice
        + 0.35 * tv
        + 0.20 * boundary_loss
        + 0.30 * empty_penalty
        + 0.03 * uncertainty_reg
    )

    return loss, bce.item(), dice.item(), tv.item(), boundary_loss.item(), empty_penalty.item()


def batch_metrics(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets_bin = (targets > 0.5).float()

    B, C = targets.shape[:2]

    dice_all = []
    iou_all = []
    dice_cls = [[] for _ in range(C)]
    iou_cls = [[] for _ in range(C)]

    for b in range(B):
        for c in range(C):
            if targets_bin[b, c].sum() > 0:
                inter = (preds[b, c] * targets_bin[b, c]).sum()
                d_union = preds[b, c].sum() + targets_bin[b, c].sum()
                j_union = (preds[b, c] + targets_bin[b, c] - preds[b, c] * targets_bin[b, c]).sum()

                d = (2.0 * inter + eps) / (d_union + eps)
                j = (inter + eps) / (j_union + eps)

                dice_all.append(d.item())
                iou_all.append(j.item())
                dice_cls[c].append(d.item())
                iou_cls[c].append(j.item())

    empty = targets_bin.sum(dim=(1, 2, 3)) < 1.0
    empty_fp = None
    if empty.any():
        empty_fp = (preds[empty].sum(dim=(1, 2, 3)) > 0).float().mean().item()

    return {
        "dice": dice_all,
        "iou": iou_all,
        "dice_f": dice_cls[0],
        "dice_o": dice_cls[1],
        "iou_f": iou_cls[0],
        "iou_o": iou_cls[1],
        "empty_fp": empty_fp
    }


# ============================================================
# CHECKPOINT
# ============================================================

def load_partial_checkpoint(model, ckpt_path):
    if ckpt_path is None:
        return model

    if not os.path.exists(ckpt_path):
        print("Warm-start checkpoint not found:", ckpt_path)
        return model

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    model_dict = model.state_dict()
    loaded = {}

    for k, v in state.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            loaded[k] = v

    model_dict.update(loaded)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(loaded)} compatible layers from checkpoint.")
    return model


# ============================================================
# FOLDS
# ============================================================

def make_case_folds(cases, n_folds=5, seed=42):
    cases = list(cases)
    random.Random(seed).shuffle(cases)

    folds = [[] for _ in range(n_folds)]
    for i, case in enumerate(cases):
        folds[i % n_folds].append(case)

    return folds


def mean_or_zero(x):
    return float(np.mean(x)) if len(x) > 0 else 0.0


# ============================================================
# TRAIN / VALIDATE ONE FOLD
# ============================================================

def validate_model(model, loader):
    model.eval()

    losses = []
    all_dice = []
    all_iou = []
    dice_f = []
    dice_o = []
    iou_f = []
    iou_o = []
    fp_rates = []

    with torch.no_grad():
        for imgs, masks, _, _ in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            logits, logvar, boundary = model(imgs)
            loss, _, _, _, _, _ = total_loss(logits, logvar, boundary, masks)
            losses.append(loss.item())

            m = batch_metrics(logits, masks, threshold=THRESHOLD)
            all_dice.extend(m["dice"])
            all_iou.extend(m["iou"])
            dice_f.extend(m["dice_f"])
            dice_o.extend(m["dice_o"])
            iou_f.extend(m["iou_f"])
            iou_o.extend(m["iou_o"])
            if m["empty_fp"] is not None:
                fp_rates.append(m["empty_fp"])

    return {
        "loss": mean_or_zero(losses),
        "dice": mean_or_zero(all_dice),
        "iou": mean_or_zero(all_iou),
        "dice_f": mean_or_zero(dice_f),
        "dice_o": mean_or_zero(dice_o),
        "iou_f": mean_or_zero(iou_f),
        "iou_o": mean_or_zero(iou_o),
        "empty_fp": mean_or_zero(fp_rates)
    }


def train_one_fold(fold_id, train_cases, val_cases):
    print("\n" + "=" * 70)
    print(f"FOLD {fold_id}")
    print("Train cases:", train_cases)
    print("Val cases:", val_cases)
    print("=" * 70)

    fold_dir = os.path.join(CHECKPOINT_DIR, f"fold_{fold_id}")
    os.makedirs(fold_dir, exist_ok=True)

    train_ds = USOV3DSliceDataset(
        TRAIN_DIR,
        train_cases,
        image_size=IMAGE_SIZE,
        train=True,
        balanced=True,
        negative_per_positive=NEGATIVE_PER_POSITIVE,
        positive_min_pixels=POSITIVE_MIN_PIXELS
    )

    val_ds = USOV3DSliceDataset(
        TRAIN_DIR,
        val_cases,
        image_size=IMAGE_SIZE,
        train=False,
        balanced=False,
        positive_min_pixels=POSITIVE_MIN_PIXELS
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = IUPSegNet(in_ch=1, out_ch=NUM_CLASSES, base=32, dropout=0.15).to(DEVICE)
    model = load_partial_checkpoint(model, RESUME_CHECKPOINT)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_dice = -1.0
    best_reliability_score = -1.0
    best_dice_path = os.path.join(fold_dir, f"best_dice_fold_{fold_id}.pth")
    best_reliability_path = os.path.join(fold_dir, f"best_reliability_fold_{fold_id}.pth")

    log_path = os.path.join(METRIC_DIR, f"training_log_fold_{fold_id}_v3.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "epoch", "train_loss", "train_dice", "val_loss",
            "val_dice", "val_iou", "val_dice_f", "val_dice_o",
            "val_iou_f", "val_iou_o", "val_empty_fp", "lr"
        ])

    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_losses = []
        train_dices = []

        pbar = tqdm(train_loader, desc=f"Fold {fold_id} Epoch {epoch}/{EPOCHS}")

        for imgs, masks, _, _ in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits, logvar, boundary = model(imgs)
            loss, bce, dloss, tv, bl, empty_pen = total_loss(logits, logvar, boundary, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            m = batch_metrics(logits.detach(), masks, threshold=THRESHOLD)
            train_dices.extend(m["dice"])

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice+": f"{mean_or_zero(train_dices):.4f}",
                "empty": f"{empty_pen:.4f}"
            })

        scheduler.step()

        val = validate_model(model, val_loader)
        train_loss = mean_or_zero(train_losses)
        train_dice = mean_or_zero(train_dices)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nFold {fold_id} Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, Train Dice+={train_dice:.4f}, "
            f"Val Loss={val['loss']:.4f}, Val Dice+={val['dice']:.4f}, "
            f"Val IoU+={val['iou']:.4f}, F-Dice={val['dice_f']:.4f}, "
            f"O-Dice={val['dice_o']:.4f}, Empty-FP={val['empty_fp']:.4f}"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                fold_id, epoch, train_loss, train_dice, val["loss"],
                val["dice"], val["iou"], val["dice_f"], val["dice_o"],
                val["iou_f"], val["iou_o"], val["empty_fp"], current_lr
            ])

        # Best Dice checkpoint.
        if val["dice"] > best_dice:
            best_dice = val["dice"]
            no_improve = 0

            torch.save({
                "fold": fold_id,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "criterion": "best_dice",
                "val_metrics": val,
                "train_cases": train_cases,
                "val_cases": val_cases,
                "config": {
                    "image_size": IMAGE_SIZE,
                    "num_classes": NUM_CLASSES,
                    "threshold": THRESHOLD,
                    "review_threshold": REVIEW_THRESHOLD
                }
            }, best_dice_path)

            print("Saved best Dice model:", best_dice_path)
        else:
            no_improve += 1

        # Best reliability-aware checkpoint.
        reliability_score = val["dice"] - 0.15 * val["empty_fp"]
        if reliability_score > best_reliability_score:
            best_reliability_score = reliability_score

            torch.save({
                "fold": fold_id,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "criterion": "best_reliability",
                "val_metrics": val,
                "reliability_score": reliability_score,
                "train_cases": train_cases,
                "val_cases": val_cases,
                "config": {
                    "image_size": IMAGE_SIZE,
                    "num_classes": NUM_CLASSES,
                    "threshold": THRESHOLD,
                    "review_threshold": REVIEW_THRESHOLD
                }
            }, best_reliability_path)

            print("Saved best Reliability model:", best_reliability_path)

        if no_improve >= PATIENCE:
            print(f"Early stopping fold {fold_id} at epoch {epoch}.")
            break

    return {
        "fold": fold_id,
        "best_dice": best_dice,
        "best_dice_path": best_dice_path,
        "best_reliability_score": best_reliability_score,
        "best_reliability_path": best_reliability_path,
        "train_cases": train_cases,
        "val_cases": val_cases
    }


# ============================================================
# IU-PSEG UNCERTAINTY MAPS
# ============================================================

def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
            m.train()


def refine_mask(mask):
    refined = np.zeros_like(mask, dtype=np.uint8)

    for c in range(mask.shape[0]):
        m = mask[c].astype(bool)
        m = binary_closing(m, disk(2))
        m = binary_opening(m, disk(1))
        m = remove_small_objects(m, min_size=64)
        m = binary_fill_holes(m)
        refined[c] = m.astype(np.uint8)

    return refined


def compute_iupseg_maps_single_model(model, img_tensor, threshold=0.5, mc_samples=10):
    probs_mc = []
    alea_mc = []
    boundary_mc = []

    model.eval()
    enable_mc_dropout(model)

    with torch.no_grad():
        for _ in range(mc_samples):
            logits, logvar, boundary_logits = model(img_tensor)

            prob = torch.sigmoid(logits)
            alea = torch.sigmoid(logvar) * (4.0 * prob * (1.0 - prob))
            boundary = torch.sigmoid(boundary_logits)

            probs_mc.append(prob.cpu().numpy())
            alea_mc.append(alea.cpu().numpy())
            boundary_mc.append(boundary.cpu().numpy())

    probs_mc = np.concatenate(probs_mc, axis=0)       # [T, C, H, W]
    alea_mc = np.concatenate(alea_mc, axis=0)
    boundary_mc = np.concatenate(boundary_mc, axis=0)

    return probs_mc, alea_mc, boundary_mc


def compute_iupseg_maps_ensemble(models, img_tensor, threshold=0.5, mc_samples=10):
    all_probs = []
    all_alea = []
    all_boundary = []

    for model in models:
        p, a, b = compute_iupseg_maps_single_model(
            model,
            img_tensor,
            threshold=threshold,
            mc_samples=mc_samples
        )
        all_probs.append(p)
        all_alea.append(a)
        all_boundary.append(b)

    probs_mc = np.concatenate(all_probs, axis=0)
    alea_mc = np.concatenate(all_alea, axis=0)
    boundary_mc = np.concatenate(all_boundary, axis=0)

    mean_prob = probs_mc.mean(axis=0)                 # [C, H, W]
    epistemic = probs_mc.var(axis=0) * 4.0
    aleatoric = alea_mc.mean(axis=0)
    boundary_response = boundary_mc.mean(axis=0)

    epistemic = np.clip(epistemic, 0, 1)
    aleatoric = np.clip(aleatoric, 0, 1)

    total_unc = np.clip(0.5 * epistemic + 0.5 * aleatoric, 0, 1)

    raw_mask = (mean_prob > threshold).astype(np.uint8)
    refined = refine_mask(raw_mask)

    # IU-PSeg identifiability definition:
    # identifiability = mean prediction x (1 - normalized uncertainty)
    identifiability = np.clip(mean_prob * (1.0 - total_unc), 0, 1)

    voxel_reliability = identifiability * (1.0 - 0.25 * boundary_response)
    voxel_reliability = np.clip(voxel_reliability, 0, 1)

    if refined.sum() > 0:
        case_reliability = float(voxel_reliability[refined > 0].mean())
    else:
        case_reliability = 0.0

    expert_review_flag = case_reliability < REVIEW_THRESHOLD

    return {
        "mean_prob": mean_prob,
        "refined_mask": refined,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "total_uncertainty": total_unc,
        "identifiability": identifiability,
        "boundary_response": boundary_response,
        "voxel_reliability": voxel_reliability,
        "case_reliability": case_reliability,
        "expert_review_flag": expert_review_flag
    }


def threshold_sensitivity(mean_prob):
    results = {}
    for t in [0.30, 0.40, 0.50, 0.60, 0.70]:
        results[str(t)] = int((mean_prob > t).sum())
    return results


def save_slice_visual(case_id, z, img, maps, out_dir):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    items = [
        ("Image", img.squeeze(), "gray"),
        ("Mask-F", maps["refined_mask"][0], "gray"),
        ("Mask-O", maps["refined_mask"][1], "gray"),
        ("Aleatoric", maps["aleatoric"].mean(axis=0), "hot"),
        ("Epistemic", maps["epistemic"].mean(axis=0), "hot"),
        ("Total Unc.", maps["total_uncertainty"].mean(axis=0), "hot"),
        ("Identifiability", maps["identifiability"].mean(axis=0), "viridis"),
        ("Boundary", maps["boundary_response"].mean(axis=0), "gray"),
        ("Reliability", maps["voxel_reliability"].mean(axis=0), "viridis"),
        ("Probability", maps["mean_prob"].mean(axis=0), "gray"),
    ]

    for ax, (title, data, cmap) in zip(axes.flatten(), items):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"{case_id} slice {z} | Reliability={maps['case_reliability']:.3f} | "
        f"Review={maps['expert_review_flag']}",
        fontsize=12
    )

    save_path = os.path.join(out_dir, f"{case_id}_slice_{z}_iupseg_maps_v3.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()


# ============================================================
# LOAD TRAINED FOLD MODELS
# ============================================================

def load_models_from_paths(paths):
    models = []
    for p in paths:
        model = IUPSegNet(in_ch=1, out_ch=NUM_CLASSES, base=32, dropout=0.15).to(DEVICE)
        ckpt = torch.load(p, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)
    return models


# ============================================================
# EXTERNAL TEST ENSEMBLE INFERENCE
# ============================================================

def run_external_test_inference(model_paths):
    print("\nLoading ensemble models:")
    for p in model_paths:
        print(" ", p)

    models = load_models_from_paths(model_paths)
    test_cases = get_base_cases(TEST_DIR)

    report_path = os.path.join(SAVE_DIR, "USOV3D_IUPSeg_external_test_report_v3.csv")
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "slice_id", "case_reliability", "expert_review_flag",
            "thr_0.3", "thr_0.4", "thr_0.5", "thr_0.6", "thr_0.7"
        ])

    case_summary = []

    for case in test_cases:
        print("\nExternal ensemble inference:", case)

        case_out = os.path.join(MAP_DIR, case)
        os.makedirs(case_out, exist_ok=True)

        vol = load_case_image(TEST_DIR, case)
        D = vol.shape[0]

        all_masks = []
        all_reliability = []

        step = max(1, D // MAX_VIS_SLICES_PER_CASE)
        selected_slices = set(list(range(0, D, step)))

        for z in tqdm(range(D), desc=case):
            img = torch.from_numpy(vol[z]).float().unsqueeze(0).unsqueeze(0)

            img_rs = F.interpolate(
                img,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False
            ).to(DEVICE)

            maps = compute_iupseg_maps_ensemble(
                models,
                img_rs,
                threshold=THRESHOLD,
                mc_samples=MC_SAMPLES
            )

            all_masks.append(maps["refined_mask"])
            all_reliability.append(maps["case_reliability"])

            sens = threshold_sensitivity(maps["mean_prob"])

            with open(report_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    case, z, f"{maps['case_reliability']:.4f}",
                    maps["expert_review_flag"],
                    sens["0.3"], sens["0.4"], sens["0.5"], sens["0.6"], sens["0.7"]
                ])

            if SAVE_VISUALS and z in selected_slices:
                save_slice_visual(case, z, img_rs.cpu().numpy()[0, 0], maps, case_out)

        all_masks = np.stack(all_masks, axis=1)  # [C, D, H, W]
        np.save(os.path.join(PRED_DIR, f"{case}_refined_mask_v3.npy"), all_masks)

        mean_rel = float(np.mean(all_reliability))
        review_rate = float(np.mean(np.array(all_reliability) < REVIEW_THRESHOLD))
        case_summary.append([case, mean_rel, review_rate, D])

        print(f"{case} mean reliability: {mean_rel:.4f} | review rate: {review_rate:.4f}")

    summary_path = os.path.join(SAVE_DIR, "USOV3D_IUPSeg_case_summary_v3.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "mean_reliability", "review_rate", "num_slices"])
        writer.writerows(case_summary)

    print("\nSaved external report:", report_path)
    print("Saved case summary:", summary_path)


# ============================================================
# CROSS-VALIDATION MAIN
# ============================================================

def run_cross_validation():
    all_cases = get_base_cases(TRAIN_DIR)
    folds = make_case_folds(all_cases, n_folds=N_FOLDS, seed=42)

    print("\nAll labeled cases:", all_cases)
    print("Fold split:", folds)

    split_path = os.path.join(METRIC_DIR, "fold_split_v3.json")
    with open(split_path, "w") as f:
        json.dump({"all_cases": all_cases, "folds": folds}, f, indent=2)

    fold_results = []
    best_dice_paths = []
    best_reliability_paths = []

    for fold_id in range(N_FOLDS):
        val_cases = folds[fold_id]
        train_cases = []
        for j in range(N_FOLDS):
            if j != fold_id:
                train_cases.extend(folds[j])

        result = train_one_fold(fold_id + 1, train_cases, val_cases)
        fold_results.append(result)
        best_dice_paths.append(result["best_dice_path"])
        best_reliability_paths.append(result["best_reliability_path"])

    cv_path = os.path.join(METRIC_DIR, "cross_validation_summary_v3.csv")
    with open(cv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "best_dice", "best_reliability_score",
            "best_dice_path", "best_reliability_path",
            "train_cases", "val_cases"
        ])

        for r in fold_results:
            writer.writerow([
                r["fold"],
                r["best_dice"],
                r["best_reliability_score"],
                r["best_dice_path"],
                r["best_reliability_path"],
                "|".join(r["train_cases"]),
                "|".join(r["val_cases"])
            ])

    mean_dice = float(np.mean([r["best_dice"] for r in fold_results]))
    std_dice = float(np.std([r["best_dice"] for r in fold_results]))

    print("\n" + "=" * 70)
    print("5-FOLD CASE-LEVEL CROSS-VALIDATION FINISHED")
    print(f"Mean Dice+ = {mean_dice:.4f} ± {std_dice:.4f}")
    print("Saved:", cv_path)
    print("=" * 70)

    return best_dice_paths, best_reliability_paths


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Using device:", DEVICE)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    best_dice_paths, best_reliability_paths = run_cross_validation()

    if RUN_EXTERNAL_ENSEMBLE:
        # For manuscript, use best-Dice ensemble first.
        run_external_test_inference(best_dice_paths)

    print("\nV3 completed.")
    print("Results folder:", SAVE_DIR)
