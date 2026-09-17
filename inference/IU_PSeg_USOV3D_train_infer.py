import os
import re
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
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

ROOT = r"C:\Users\M YAQUB\CSU2026\USOV3D"

TRAIN_DIR = os.path.join(ROOT, "Training_Set_2019", "Training_Set_1")
TEST_DIR  = os.path.join(ROOT, "Test_Set_2019", "Test_Set_1")

SAVE_DIR = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\USOV3D_RESULTS"
os.makedirs(SAVE_DIR, exist_ok=True)

CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
PRED_DIR = os.path.join(SAVE_DIR, "predictions")
MAP_DIR = os.path.join(SAVE_DIR, "iupseg_maps")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(MAP_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 80
LR = 1e-4
NUM_CLASSES = 2          # channel 0 = f, channel 1 = o
MC_SAMPLES = 20
THRESHOLD = 0.5

# optional: put your previous IU-PSeg checkpoint path here
RESUME_CHECKPOINT = None
# Example:
# RESUME_CHECKPOINT = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\best_model.pth"


print("Using device:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


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
        raise ValueError(f"Unexpected dimensions in {path}: {dims}")

    vol = arr.reshape((dims[2], dims[1], dims[0]))
    return vol


def get_base_cases(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".vtk")]
    cases = []
    for f in files:
        if re.match(r"vol\d+\.vtk$", f):
            cases.append(f.replace(".vtk", ""))
    return sorted(cases, key=lambda x: int(x.replace("vol", "")))


@lru_cache(maxsize=32)
def load_case_image(case_id, folder):
    img_path = os.path.join(folder, f"{case_id}.vtk")
    img = read_vtk_volume(img_path).astype(np.float32)
    img = np.clip(img, 0, 255)
    img = img / 255.0
    return img


@lru_cache(maxsize=32)
def load_case_mask(case_id, folder):
    f_r1 = read_vtk_volume(os.path.join(folder, f"{case_id}_f_r1.vtk")) > 0
    f_r2 = read_vtk_volume(os.path.join(folder, f"{case_id}_f_r2.vtk")) > 0
    o_r1 = read_vtk_volume(os.path.join(folder, f"{case_id}_o_r1.vtk")) > 0
    o_r2 = read_vtk_volume(os.path.join(folder, f"{case_id}_o_r2.vtk")) > 0

    # soft/union voting from two raters
    f_mask = ((f_r1.astype(np.uint8) + f_r2.astype(np.uint8)) >= 1).astype(np.float32)
    o_mask = ((o_r1.astype(np.uint8) + o_r2.astype(np.uint8)) >= 1).astype(np.float32)

    mask = np.stack([f_mask, o_mask], axis=0)  # [2, D, H, W]
    return mask


# ============================================================
# DATASET
# ============================================================

class USOV3DSliceDataset(Dataset):
    def __init__(self, folder, cases, image_size=256, train=True, only_positive=True):
        self.folder = folder
        self.cases = cases
        self.image_size = image_size
        self.train = train
        self.only_positive = only_positive
        self.index = []

        for case in cases:
            img = load_case_image(case, folder)
            mask = load_case_mask(case, folder)

            D = img.shape[0]
            for z in range(D):
                if only_positive:
                    if mask[:, z].sum() > 0:
                        self.index.append((case, z))
                else:
                    self.index.append((case, z))

        print(f"Dataset built: {len(self.index)} slices from {len(cases)} cases")

    def __len__(self):
        return len(self.index)

    def random_aug(self, img, mask):
        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=2).copy()

        if random.random() < 0.5:
            img = np.flip(img, axis=0).copy()
            mask = np.flip(mask, axis=1).copy()

        if random.random() < 0.3:
            factor = random.uniform(0.85, 1.15)
            img = np.clip(img * factor, 0, 1)

        return img, mask

    def __getitem__(self, idx):
        case, z = self.index[idx]

        img3d = load_case_image(case, self.folder)
        mask3d = load_case_mask(case, self.folder)

        img = img3d[z]             # [H, W]
        mask = mask3d[:, z]        # [2, H, W]

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
# IU-PSEG MODEL: U-NET + DROPOUT
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class IUPSegNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, base=32, dropout=0.15):
        super().__init__()

        self.enc1 = DoubleConv(in_ch, base, dropout)
        self.enc2 = DoubleConv(base, base * 2, dropout)
        self.enc3 = DoubleConv(base * 2, base * 4, dropout)
        self.enc4 = DoubleConv(base * 4, base * 8, dropout)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16, dropout)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8, dropout)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4, dropout)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2, dropout)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base, dropout)

        self.seg_head = nn.Conv2d(base, out_ch, 1)

        # aleatoric uncertainty head
        self.logvar_head = nn.Conv2d(base, out_ch, 1)

        # boundary response head
        self.boundary_head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.seg_head(d1)
        logvar = self.logvar_head(d1)
        boundary = self.boundary_head(d1)

        return logits, logvar, boundary


# ============================================================
# LOSSES AND METRICS
# ============================================================

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = torch.sum(probs * targets, dims)
    union = torch.sum(probs + targets, dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def dice_score(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    dims = (0, 2, 3)
    inter = torch.sum(preds * targets, dims)
    union = torch.sum(preds + targets, dims)
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


def iou_score(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    dims = (0, 2, 3)
    inter = torch.sum(preds * targets, dims)
    union = torch.sum(preds + targets - preds * targets, dims)
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()


def make_boundary_from_mask(mask):
    # mask: [B, C, H, W]
    mask_np = mask.detach().cpu().numpy()
    out = np.zeros_like(mask_np, dtype=np.float32)

    for b in range(mask_np.shape[0]):
        for c in range(mask_np.shape[1]):
            bd = find_boundaries(mask_np[b, c] > 0.5, mode="outer")
            out[b, c] = bd.astype(np.float32)

    return torch.from_numpy(out).to(mask.device)


def total_loss(logits, logvar, boundary_logits, masks):
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    dloss = dice_loss(logits, masks)

    boundary_target = make_boundary_from_mask(masks)
    bl = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target)

    # aleatoric regularization
    prob = torch.sigmoid(logits)
    alea = torch.exp(logvar)
    uncertainty_reg = torch.mean(alea * torch.abs(prob - masks) + 0.01 * logvar ** 2)

    loss = bce + dloss + 0.2 * bl + 0.05 * uncertainty_reg
    return loss, bce.item(), dloss.item(), bl.item()


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_partial_checkpoint(model, ckpt_path):
    if ckpt_path is None:
        return model

    if not os.path.exists(ckpt_path):
        print("Checkpoint not found:", ckpt_path)
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
# TRAINING
# ============================================================

def split_cases(cases, seed=42):
    random.seed(seed)
    cases = cases.copy()
    random.shuffle(cases)

    n = len(cases)
    n_train = int(0.75 * n)
    n_val = int(0.125 * n)

    train_cases = cases[:n_train]
    val_cases = cases[n_train:n_train+n_val]
    internal_test_cases = cases[n_train+n_val:]

    return train_cases, val_cases, internal_test_cases


def train_model():
    all_cases = get_base_cases(TRAIN_DIR)
    train_cases, val_cases, internal_test_cases = split_cases(all_cases)

    print("Train cases:", train_cases)
    print("Val cases:", val_cases)
    print("Internal test cases:", internal_test_cases)

    train_ds = USOV3DSliceDataset(TRAIN_DIR, train_cases, IMAGE_SIZE, train=True, only_positive=True)
    val_ds = USOV3DSliceDataset(TRAIN_DIR, val_cases, IMAGE_SIZE, train=False, only_positive=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = IUPSegNet(in_ch=1, out_ch=NUM_CLASSES, base=32, dropout=0.15).to(DEVICE)
    model = load_partial_checkpoint(model, RESUME_CHECKPOINT)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_dice = 0.0
    best_path = os.path.join(CHECKPOINT_DIR, "best_iupseg_usov3d.pth")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss = 0.0
        train_dice = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for imgs, masks, _, _ in pbar:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits, logvar, boundary = model(imgs)
            loss, bce, dloss, bl = total_loss(logits, logvar, boundary, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            dsc = dice_score(logits.detach(), masks)

            train_loss += loss.item()
            train_dice += dsc

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{dsc:.4f}",
                "bce": f"{bce:.4f}"
            })

        scheduler.step()

        train_loss /= max(1, len(train_loader))
        train_dice /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for imgs, masks, _, _ in val_loader:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                logits, logvar, boundary = model(imgs)
                loss, _, _, _ = total_loss(logits, logvar, boundary, masks)

                val_loss += loss.item()
                val_dice += dice_score(logits, masks)
                val_iou += iou_score(logits, masks)

        val_loss /= max(1, len(val_loader))
        val_dice /= max(1, len(val_loader))
        val_iou /= max(1, len(val_loader))

        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, Train Dice={train_dice:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}, Val IoU={val_iou:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "train_cases": train_cases,
                "val_cases": val_cases,
                "internal_test_cases": internal_test_cases
            }, best_path)

            print("Saved best model:", best_path)

    print("Best validation Dice:", best_dice)
    return best_path


# ============================================================
# IU-PSEG UNCERTAINTY INFERENCE
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


def compute_iupseg_maps(model, img_tensor, threshold=0.5, mc_samples=20):
    """
    img_tensor: [1, 1, H, W]
    """
    probs_mc = []
    alea_mc = []
    boundary_mc = []

    model.eval()
    enable_mc_dropout(model)

    with torch.no_grad():
        for _ in range(mc_samples):
            logits, logvar, boundary_logits = model(img_tensor)

            prob = torch.sigmoid(logits)
            alea = torch.sigmoid(logvar)
            boundary = torch.sigmoid(boundary_logits)

            probs_mc.append(prob.cpu().numpy())
            alea_mc.append(alea.cpu().numpy())
            boundary_mc.append(boundary.cpu().numpy())

    probs_mc = np.concatenate(probs_mc, axis=0)      # [T, C, H, W]
    alea_mc = np.concatenate(alea_mc, axis=0)
    boundary_mc = np.concatenate(boundary_mc, axis=0)

    mean_prob = probs_mc.mean(axis=0)                # [C, H, W]
    epistemic = probs_mc.var(axis=0)                 # [C, H, W]
    aleatoric = alea_mc.mean(axis=0)                 # [C, H, W]
    total_unc = epistemic + aleatoric

    raw_mask = (mean_prob > threshold).astype(np.uint8)
    refined = refine_mask(raw_mask)

    boundary_response = boundary_mc.mean(axis=0)

    confidence = np.maximum(mean_prob, 1.0 - mean_prob)
    identifiability = 1.0 - total_unc
    identifiability = np.clip(identifiability, 0, 1)

    voxel_reliability = confidence * identifiability * (1.0 - boundary_response)
    voxel_reliability = np.clip(voxel_reliability, 0, 1)

    positive_area = refined.sum()
    if positive_area > 0:
        case_reliability = float(voxel_reliability[refined > 0].mean())
    else:
        case_reliability = float(voxel_reliability.mean())

    expert_review_flag = case_reliability < 0.75

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
        mask = (mean_prob > t).astype(np.uint8)
        results[str(t)] = int(mask.sum())
    return results


def save_slice_visual(case_id, z, img, maps, out_dir):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    img_show = img.squeeze()

    items = [
        ("Image", img_show, "gray"),
        ("Refined Mask F", maps["refined_mask"][0], "gray"),
        ("Refined Mask O", maps["refined_mask"][1], "gray"),
        ("Aleatoric", maps["aleatoric"].mean(axis=0), "hot"),
        ("Epistemic", maps["epistemic"].mean(axis=0), "hot"),
        ("Total Unc.", maps["total_uncertainty"].mean(axis=0), "hot"),
        ("Identifiability", maps["identifiability"].mean(axis=0), "viridis"),
        ("Boundary", maps["boundary_response"].mean(axis=0), "gray"),
        ("Voxel Reliability", maps["voxel_reliability"].mean(axis=0), "viridis"),
        ("Probability", maps["mean_prob"].mean(axis=0), "gray"),
    ]

    for ax, (title, data, cmap) in zip(axes.flatten(), items):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"{case_id} slice {z} | Reliability={maps['case_reliability']:.3f} | "
        f"Expert Review={maps['expert_review_flag']}",
        fontsize=12
    )

    save_path = os.path.join(out_dir, f"{case_id}_slice_{z}_iupseg_maps.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()


def run_external_test_inference(checkpoint_path):
    model = IUPSegNet(in_ch=1, out_ch=NUM_CLASSES, base=32, dropout=0.15).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    test_cases = get_base_cases(TEST_DIR)

    report_lines = []
    report_lines.append("case_id,slice_id,case_reliability,expert_review_flag,thr_0.3,thr_0.4,thr_0.5,thr_0.6,thr_0.7\n")

    for case in test_cases:
        print("Running inference:", case)

        case_out = os.path.join(MAP_DIR, case)
        os.makedirs(case_out, exist_ok=True)

        vol = load_case_image(case, TEST_DIR)
        D = vol.shape[0]

        all_masks = []
        all_reliability = []

        # save maps for center and positive-like slices only
        selected_slices = list(range(0, D, max(1, D // 10)))

        for z in tqdm(range(D), desc=case):
            img = torch.from_numpy(vol[z]).float().unsqueeze(0).unsqueeze(0)

            img_rs = F.interpolate(
                img,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False
            ).to(DEVICE)

            maps = compute_iupseg_maps(
                model,
                img_rs,
                threshold=THRESHOLD,
                mc_samples=MC_SAMPLES
            )

            all_masks.append(maps["refined_mask"])
            all_reliability.append(maps["case_reliability"])

            sens = threshold_sensitivity(maps["mean_prob"])

            report_lines.append(
                f"{case},{z},{maps['case_reliability']:.4f},"
                f"{maps['expert_review_flag']},"
                f"{sens['0.3']},{sens['0.4']},{sens['0.5']},{sens['0.6']},{sens['0.7']}\n"
            )

            if z in selected_slices:
                save_slice_visual(
                    case,
                    z,
                    img_rs.cpu().numpy()[0, 0],
                    maps,
                    case_out
                )

        all_masks = np.stack(all_masks, axis=1)  # [C, D, H, W]
        np.save(os.path.join(PRED_DIR, f"{case}_refined_mask.npy"), all_masks)

        case_score = float(np.mean(all_reliability))
        print(f"{case} reliability score: {case_score:.4f}")

    report_path = os.path.join(SAVE_DIR, "USOV3D_IUPSeg_external_test_report.csv")
    with open(report_path, "w") as f:
        f.writelines(report_lines)

    print("Saved report:", report_path)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Step 1: train model
    best_model_path = train_model()

    # Step 2: run IU-PSeg inference on official test set
    run_external_test_inference(best_model_path)