
# ============================================================
# EVALUATE_MR3DUS_VOLUME_METRICS_V3.py
#
# Purpose:
#   Calculate complete 3D volume-wise metrics for MR-3DUS using the
#   trained V3 5-fold IU-PSeg checkpoints.
#
# Metrics:
#   Dice, IoU, Precision, Recall, F1, Accuracy, Sensitivity,
#   Specificity, PPV, NPV, AUC, HD95, ASD
#
# Outputs:
#   1) volume_metrics_per_target_v3.csv
#   2) volume_metrics_per_case_v3.csv
#   3) volume_summary_mean_sd_v3.csv
#
# Expected checkpoint paths:
#   C:\Users\M YAQUB\CSU2026\IU-PSeg\USOV3D_RESULTS_V3\checkpoints\fold_1\best_dice_fold_1.pth
#   ...
#   C:\Users\M YAQUB\CSU2026\IU-PSeg\USOV3D_RESULTS_V3\checkpoints\fold_5\best_dice_fold_5.pth
#
# Run:
#   cd /d "C:\Users\M YAQUB\CSU2026\IU-PSeg"
#   python EVALUATE_MR3DUS_VOLUME_METRICS_V3.py
# ============================================================

import os
import re
import csv
import json
import math
import warnings
from functools import lru_cache

import numpy as np
import pyvista as pv

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    binary_fill_holes,
)
from skimage.morphology import (
    remove_small_objects,
    binary_opening,
    binary_closing,
    disk,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

ROOT = r"C:\Users\M YAQUB\CSU2026\USOV3D"
TRAIN_DIR = os.path.join(ROOT, "Training_Set_2019", "Training_Set_1")

RESULT_ROOT = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\USOV3D_RESULTS_V3"
CHECKPOINT_DIR = os.path.join(RESULT_ROOT, "checkpoints")
METRIC_DIR = os.path.join(RESULT_ROOT, "metrics")
os.makedirs(METRIC_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 256
NUM_CLASSES = 2
THRESHOLD = 0.50

# Choose "best_dice" or "best_reliability"
MODEL_SELECTION = "best_dice"

# Consensus rule:
# If both raters agree, soft label = 1.0. If one rater marks it, soft = 0.5.
# Validation code used target > 0.5, so this script uses strict consensus.
GT_POSITIVE_THRESHOLD = 0.50

# Post-processing matches the generated V3 external maps.
APPLY_POSTPROCESSING = True
POSTPROCESS_MIN_SIZE = 64

# Surface-distance penalty if GT exists but prediction is empty.
# If True, HD95/ASD for complete missed target = physical image diagonal.
PENALIZE_COMPLETE_MISS = True

# Save optional prediction volumes as .npy for checking.
SAVE_PREDICTION_NPY = False
PRED_SAVE_DIR = os.path.join(METRIC_DIR, "volume_predictions_v3")
os.makedirs(PRED_SAVE_DIR, exist_ok=True)


# ============================================================
# MODEL ARCHITECTURE: SAME AS V3
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
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )
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
# VTK READING
# ============================================================

def read_vtk_volume_with_spacing(path):
    mesh = pv.read(path)

    if len(mesh.point_data.keys()) > 0:
        key = list(mesh.point_data.keys())[0]
        arr = np.asarray(mesh.point_data[key])
    elif len(mesh.cell_data.keys()) > 0:
        key = list(mesh.cell_data.keys())[0]
        arr = np.asarray(mesh.cell_data[key])
    else:
        raise ValueError(f"No scalar data found in: {path}")

    dims = mesh.dimensions
    if len(dims) != 3:
        raise ValueError(f"Unexpected VTK dimensions in {path}: {dims}")

    # Existing V3 code uses volume order [D, H, W] = [z, y, x].
    vol = arr.reshape((dims[2], dims[1], dims[0]))

    spacing_xyz = getattr(mesh, "spacing", (1.0, 1.0, 1.0))
    if spacing_xyz is None or len(spacing_xyz) != 3:
        spacing_xyz = (1.0, 1.0, 1.0)

    # Convert from VTK spacing order (x, y, z) to numpy volume order (D, H, W).
    spacing_dhw = (
        float(spacing_xyz[2]),
        float(spacing_xyz[1]),
        float(spacing_xyz[0]),
    )
    spacing_dhw = tuple(s if np.isfinite(s) and s > 0 else 1.0 for s in spacing_dhw)

    return vol, spacing_dhw


@lru_cache(maxsize=64)
def load_image(case_id):
    img, spacing_dhw = read_vtk_volume_with_spacing(
        os.path.join(TRAIN_DIR, f"{case_id}.vtk")
    )
    img = np.clip(img.astype(np.float32), 0, 255) / 255.0
    return img, spacing_dhw


@lru_cache(maxsize=64)
def load_gt_mask(case_id):
    f_r1, _ = read_vtk_volume_with_spacing(os.path.join(TRAIN_DIR, f"{case_id}_f_r1.vtk"))
    f_r2, _ = read_vtk_volume_with_spacing(os.path.join(TRAIN_DIR, f"{case_id}_f_r2.vtk"))
    o_r1, _ = read_vtk_volume_with_spacing(os.path.join(TRAIN_DIR, f"{case_id}_o_r1.vtk"))
    o_r2, _ = read_vtk_volume_with_spacing(os.path.join(TRAIN_DIR, f"{case_id}_o_r2.vtk"))

    f_soft = ((f_r1 > 0).astype(np.float32) + (f_r2 > 0).astype(np.float32)) / 2.0
    o_soft = ((o_r1 > 0).astype(np.float32) + (o_r2 > 0).astype(np.float32)) / 2.0

    f_gt = f_soft > GT_POSITIVE_THRESHOLD
    o_gt = o_soft > GT_POSITIVE_THRESHOLD

    return np.stack([f_gt, o_gt], axis=0).astype(bool)


# ============================================================
# CHECKPOINT / FOLD LOADING
# ============================================================

def checkpoint_path_for_fold(fold_id):
    return os.path.join(
        CHECKPOINT_DIR,
        f"fold_{fold_id}",
        f"{MODEL_SELECTION}_fold_{fold_id}.pth"
    )


def load_fold_model(fold_id):
    ckpt_path = checkpoint_path_for_fold(fold_id)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{ckpt_path}\n"
            "Please confirm V3 training completed and MODEL_SELECTION is correct."
        )

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = IUPSegNet(in_ch=1, out_ch=NUM_CLASSES, base=32, dropout=0.15).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_cases = list(ckpt["val_cases"])
    epoch = ckpt.get("epoch", "NA")
    val_metrics = ckpt.get("val_metrics", {})

    return model, val_cases, ckpt_path, epoch, val_metrics


# ============================================================
# PREDICTION
# ============================================================

def refine_mask_2d(mask_2d):
    m = mask_2d.astype(bool)
    m = binary_closing(m, disk(2))
    m = binary_opening(m, disk(1))
    m = remove_small_objects(m, min_size=POSTPROCESS_MIN_SIZE)
    m = binary_fill_holes(m)
    return m.astype(np.uint8)


def predict_volume(model, image):
    """
    image: [D, H, W], normalized [0, 1]
    returns:
      prob_volume: [C, D, H, W], float32 probability at native resolution
      pred_volume: [C, D, H, W], bool binary prediction at native resolution
    """
    d, h, w = image.shape
    prob_volume = np.zeros((NUM_CLASSES, d, h, w), dtype=np.float32)
    pred_volume = np.zeros((NUM_CLASSES, d, h, w), dtype=np.uint8)

    with torch.no_grad():
        for z in tqdm(range(d), desc="Predicting slices", leave=False):
            x = torch.from_numpy(image[z]).float().unsqueeze(0).unsqueeze(0)

            x256 = F.interpolate(
                x,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            ).to(DEVICE)

            logits, _, _ = model(x256)
            prob256 = torch.sigmoid(logits)

            prob_native = F.interpolate(
                prob256,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )[0].cpu().numpy().astype(np.float32)

            mask256 = (prob256 > THRESHOLD).float()

            if APPLY_POSTPROCESSING:
                mask_np = mask256[0].cpu().numpy().astype(np.uint8)
                for c in range(NUM_CLASSES):
                    mask_np[c] = refine_mask_2d(mask_np[c])
                mask256 = torch.from_numpy(mask_np).float().unsqueeze(0).to(DEVICE)

            mask_native = F.interpolate(
                mask256,
                size=(h, w),
                mode="nearest",
            )[0].cpu().numpy().astype(np.uint8)

            prob_volume[:, z] = prob_native
            pred_volume[:, z] = mask_native

    return prob_volume, pred_volume.astype(bool)


# ============================================================
# METRICS
# ============================================================

def physical_diagonal(mask_shape, spacing_dhw):
    d, h, w = mask_shape
    sd, sh, sw = spacing_dhw
    return float(
        math.sqrt(
            ((max(d - 1, 1) * sd) ** 2)
            + ((max(h - 1, 1) * sh) ** 2)
            + ((max(w - 1, 1) * sw) ** 2)
        )
    )


def surface_distance_metrics(pred, gt, spacing_dhw):
    """
    Calculates symmetric HD95 and ASD in physical units if spacing exists.
    Otherwise values are in voxel/pixel units.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_any = bool(pred.any())
    gt_any = bool(gt.any())

    if not gt_any:
        return np.nan, np.nan, "gt_empty"

    if gt_any and not pred_any:
        if PENALIZE_COMPLETE_MISS:
            penalty = physical_diagonal(gt.shape, spacing_dhw)
            return penalty, penalty, "complete_miss_penalty"
        return np.nan, np.nan, "complete_miss_nan"

    structure = np.ones((3, 3, 3), dtype=bool)
    pred_surface = pred ^ binary_erosion(pred, structure=structure, border_value=0)
    gt_surface = gt ^ binary_erosion(gt, structure=structure, border_value=0)

    if not pred_surface.any() or not gt_surface.any():
        if PENALIZE_COMPLETE_MISS:
            penalty = physical_diagonal(gt.shape, spacing_dhw)
            return penalty, penalty, "surface_failure_penalty"
        return np.nan, np.nan, "surface_failure_nan"

    dt_to_gt = distance_transform_edt(~gt_surface, sampling=spacing_dhw)
    dt_to_pred = distance_transform_edt(~pred_surface, sampling=spacing_dhw)

    d_pred_to_gt = dt_to_gt[pred_surface]
    d_gt_to_pred = dt_to_pred[gt_surface]

    distances = np.concatenate([d_pred_to_gt, d_gt_to_pred]).astype(np.float64)

    hd95 = float(np.percentile(distances, 95))
    asd = float(np.mean(distances))

    return hd95, asd, "valid"


def safe_div(num, den, default=np.nan):
    if den == 0:
        return default
    return float(num / den)


def binary_metrics_3d(prob, pred, gt, spacing_dhw):
    """
    prob, pred, gt: [D, H, W]
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    prob = prob.astype(np.float32)

    gt_present = int(gt.any())
    pred_present = int(pred.any())

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())

    if gt_present == 0:
        return {
            "Dice": np.nan,
            "IoU": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "Accuracy": np.nan,
            "Sensitivity": np.nan,
            "Specificity": np.nan,
            "PPV": np.nan,
            "NPV": np.nan,
            "AUC": np.nan,
            "HD95": np.nan,
            "ASD": np.nan,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "GT_voxels": int(gt.sum()),
            "Pred_voxels": int(pred.sum()),
            "GT_present": gt_present,
            "Pred_present": pred_present,
            "Empty_GT_false_positive": int(pred_present),
            "Distance_status": "gt_empty",
        }

    dice = safe_div(2 * tp, 2 * tp + fp + fn, default=0.0)
    iou = safe_div(tp, tp + fp + fn, default=0.0)
    precision = safe_div(tp, tp + fp, default=0.0)
    recall = safe_div(tp, tp + fn, default=0.0)
    f1 = dice
    accuracy = safe_div(tp + tn, tp + tn + fp + fn, default=0.0)
    sensitivity = recall
    specificity = safe_div(tn, tn + fp, default=0.0)
    ppv = precision
    npv = safe_div(tn, tn + fn, default=0.0)

    try:
        # AUC requires both labels present in GT.
        if len(np.unique(gt.reshape(-1).astype(np.uint8))) == 2:
            auc = float(roc_auc_score(gt.reshape(-1).astype(np.uint8), prob.reshape(-1)))
        else:
            auc = np.nan
    except Exception:
        auc = np.nan

    hd95, asd, status = surface_distance_metrics(pred, gt, spacing_dhw)

    return {
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "AUC": auc,
        "HD95": hd95,
        "ASD": asd,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "GT_voxels": int(gt.sum()),
        "Pred_voxels": int(pred.sum()),
        "GT_present": gt_present,
        "Pred_present": pred_present,
        "Empty_GT_false_positive": 0,
        "Distance_status": status,
    }


def mean_sd(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return np.nan, np.nan, 0
    if len(arr) == 1:
        return float(arr[0]), 0.0, 1
    return float(arr.mean()), float(arr.std(ddof=1)), int(len(arr))


def format_pm(mean, sd, decimals=4):
    if not np.isfinite(mean):
        return "NA"
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


# ============================================================
# CSV OUTPUT
# ============================================================

METRIC_KEYS = [
    "Dice", "IoU", "Precision", "Recall", "F1", "Accuracy",
    "Sensitivity", "Specificity", "PPV", "NPV", "AUC", "HD95", "ASD"
]


def write_per_target_csv(rows):
    out_path = os.path.join(METRIC_DIR, f"volume_metrics_per_target_{MODEL_SELECTION}_v3.csv")

    fields = [
        "Fold", "Case", "Target"
    ] + METRIC_KEYS + [
        "TP", "FP", "FN", "TN",
        "GT_voxels", "Pred_voxels",
        "GT_present", "Pred_present",
        "Empty_GT_false_positive",
        "Distance_status"
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def write_per_case_csv(rows):
    out_path = os.path.join(METRIC_DIR, f"volume_metrics_per_case_{MODEL_SELECTION}_v3.csv")

    case_map = {}
    for r in rows:
        if int(r["GT_present"]) == 1:
            key = (r["Fold"], r["Case"])
            case_map.setdefault(key, []).append(r)

    case_rows = []
    for (fold, case), rs in sorted(case_map.items(), key=lambda x: (int(x[0][0]), x[0][1])):
        row = {"Fold": fold, "Case": case, "Positive_targets": len(rs)}
        for m in METRIC_KEYS:
            row[m] = float(np.nanmean([float(r[m]) for r in rs]))
        case_rows.append(row)

    fields = ["Fold", "Case", "Positive_targets"] + METRIC_KEYS

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(case_rows)

    return out_path


def write_summary_csv(rows):
    out_path = os.path.join(METRIC_DIR, f"volume_summary_mean_sd_{MODEL_SELECTION}_v3.csv")

    groups = {
        "F target": [r for r in rows if r["Target"] == "F" and int(r["GT_present"]) == 1],
        "O target": [r for r in rows if r["Target"] == "O" and int(r["GT_present"]) == 1],
        "Overall": [r for r in rows if int(r["GT_present"]) == 1],
    }

    summary_rows = []
    for group_name, group_rows in groups.items():
        srow = {
            "Group": group_name,
            "N_target_positive": len(group_rows),
            "Complete_misses": sum(str(r["Distance_status"]).startswith("complete_miss") for r in group_rows),
            "Surface_failures": sum(str(r["Distance_status"]).startswith("surface_failure") for r in group_rows),
        }

        # count empty GT cases and false positives among all rows in this group type
        if group_name == "F target":
            all_group = [r for r in rows if r["Target"] == "F"]
        elif group_name == "O target":
            all_group = [r for r in rows if r["Target"] == "O"]
        else:
            all_group = rows

        srow["Empty_GT_cases"] = sum(int(r["GT_present"]) == 0 for r in all_group)
        srow["Empty_GT_false_positive_cases"] = sum(int(r["Empty_GT_false_positive"]) for r in all_group)

        for m in METRIC_KEYS:
            mean, sd, n = mean_sd([float(r[m]) for r in group_rows])
            srow[f"{m}_mean"] = mean
            srow[f"{m}_sd"] = sd
            srow[f"{m}_mean_pm_sd"] = format_pm(mean, sd)
            srow[f"{m}_n"] = n

        summary_rows.append(srow)

    fields = [
        "Group", "N_target_positive", "Complete_misses", "Surface_failures",
        "Empty_GT_cases", "Empty_GT_false_positive_cases"
    ]

    for m in METRIC_KEYS:
        fields.extend([f"{m}_mean", f"{m}_sd", f"{m}_mean_pm_sd", f"{m}_n"])

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    return out_path, summary_rows


def print_summary(summary_rows):
    print("\n" + "=" * 90)
    print("MR-3DUS VOLUME-WISE 5-FOLD EVALUATION SUMMARY")
    print(f"Model selection: {MODEL_SELECTION}")
    print("=" * 90)

    for r in summary_rows:
        print(f"\n{r['Group']} | n={r['N_target_positive']}")
        print(f"Dice:        {r['Dice_mean_pm_sd']}")
        print(f"IoU:         {r['IoU_mean_pm_sd']}")
        print(f"Precision:   {r['Precision_mean_pm_sd']}")
        print(f"Recall:      {r['Recall_mean_pm_sd']}")
        print(f"F1:          {r['F1_mean_pm_sd']}")
        print(f"Accuracy:    {r['Accuracy_mean_pm_sd']}")
        print(f"AUC:         {r['AUC_mean_pm_sd']}")
        print(f"HD95:        {r['HD95_mean_pm_sd']}")
        print(f"ASD:         {r['ASD_mean_pm_sd']}")
        print(f"Sensitivity: {r['Sensitivity_mean_pm_sd']}")
        print(f"Specificity: {r['Specificity_mean_pm_sd']}")
        print(f"PPV:         {r['PPV_mean_pm_sd']}")
        print(f"NPV:         {r['NPV_mean_pm_sd']}")
        print(f"Complete misses: {r['Complete_misses']}")
        print(f"Empty-GT false-positive cases: {r['Empty_GT_false_positive_cases']} / {r['Empty_GT_cases']}")

    print("=" * 90)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Using device:", DEVICE)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    all_rows = []
    seen_cases = []

    for fold_id in range(1, 6):
        print("\n" + "-" * 90)
        print(f"Evaluating fold {fold_id}")
        print("-" * 90)

        model, val_cases, ckpt_path, epoch, val_metrics = load_fold_model(fold_id)

        print("Checkpoint:", ckpt_path)
        print("Checkpoint epoch:", epoch)
        print("Stored validation metrics:", val_metrics)
        print("Held-out validation cases:", val_cases)

        for case_id in val_cases:
            if case_id in seen_cases:
                raise RuntimeError(f"Case {case_id} appears in multiple validation folds.")
            seen_cases.append(case_id)

            print(f"\nCase: {case_id}")
            image, spacing_dhw = load_image(case_id)
            gt = load_gt_mask(case_id)

            prob, pred = predict_volume(model, image)

            if pred.shape != gt.shape:
                raise RuntimeError(f"Shape mismatch for {case_id}: pred={pred.shape}, gt={gt.shape}")

            if SAVE_PREDICTION_NPY:
                np.save(os.path.join(PRED_SAVE_DIR, f"{case_id}_prob.npy"), prob.astype(np.float32))
                np.save(os.path.join(PRED_SAVE_DIR, f"{case_id}_pred.npy"), pred.astype(np.uint8))
                np.save(os.path.join(PRED_SAVE_DIR, f"{case_id}_gt.npy"), gt.astype(np.uint8))

            for c, target in enumerate(["F", "O"]):
                metrics = binary_metrics_3d(
                    prob=prob[c],
                    pred=pred[c],
                    gt=gt[c],
                    spacing_dhw=spacing_dhw
                )

                row = {
                    "Fold": fold_id,
                    "Case": case_id,
                    "Target": target,
                }
                row.update(metrics)
                all_rows.append(row)

                dice_txt = "NA" if not np.isfinite(metrics["Dice"]) else f"{metrics['Dice']:.4f}"
                iou_txt = "NA" if not np.isfinite(metrics["IoU"]) else f"{metrics['IoU']:.4f}"
                hd_txt = "NA" if not np.isfinite(metrics["HD95"]) else f"{metrics['HD95']:.4f}"
                asd_txt = "NA" if not np.isfinite(metrics["ASD"]) else f"{metrics['ASD']:.4f}"

                print(
                    f"  {target}: Dice={dice_txt}, IoU={iou_txt}, "
                    f"Precision={metrics['Precision'] if np.isfinite(metrics['Precision']) else 'NA'}, "
                    f"Recall={metrics['Recall'] if np.isfinite(metrics['Recall']) else 'NA'}, "
                    f"HD95={hd_txt}, ASD={asd_txt}, "
                    f"GT voxels={metrics['GT_voxels']}, Pred voxels={metrics['Pred_voxels']}, "
                    f"Status={metrics['Distance_status']}"
                )

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    per_target_path = write_per_target_csv(all_rows)
    per_case_path = write_per_case_csv(all_rows)
    summary_path, summary_rows = write_summary_csv(all_rows)

    print_summary(summary_rows)

    print("\nSaved files:")
    print("1)", per_target_path)
    print("2)", per_case_path)
    print("3)", summary_path)


if __name__ == "__main__":
    main()
