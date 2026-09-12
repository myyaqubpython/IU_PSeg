
# ============================================================
# EVALUATE_CEUS_OLSEG_METRICS.py
#
# Purpose:
#   Evaluate the trained IU-PSeg CEUS-OLSeg checkpoint on the test split.
#
# Metrics:
#   Dice, IoU, Precision, Recall, F1, Accuracy, Sensitivity,
#   Specificity, PPV, NPV, AUC, HD95, ASD
#
# Also calculates:
#   Threshold sensitivity analysis from 0.10 to 0.90
#
# Expected default files:
#   Data root:
#     C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d
#   Checkpoint:
#     C:\Users\M YAQUB\CSU2026\IU-PSeg\SOTA_outputs\best_IU_PSeg_SOTA.pth
#
# Outputs:
#   C:\Users\M YAQUB\CSU2026\IU-PSeg\SOTA_outputs\CEUS_OLSeg_Evaluation\
#       CEUS_OLSeg_per_case_metrics.csv
#       CEUS_OLSeg_summary_mean_sd.csv
#       CEUS_OLSeg_threshold_sensitivity.csv
#
# Run:
#   cd /d "C:\Users\M YAQUB\CSU2026\IU-PSeg"
#   python EVALUATE_CEUS_OLSEG_METRICS.py
# ============================================================

import os
import csv
import math
import argparse
import warnings

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None

from scipy.ndimage import binary_erosion, distance_transform_edt

warnings.filterwarnings("ignore")


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_DATA_ROOT = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d"
DEFAULT_CHECKPOINT = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\SOTA_outputs\best_IU_PSeg_SOTA.pth"
DEFAULT_OUTPUT_DIR = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\SOTA_outputs\CEUS_OLSeg_Evaluation"

DEFAULT_IMG_SIZE = 384
DEFAULT_MIN_AREA = 80
DEFAULT_KEEP_COMPONENTS = 2


# ============================================================
# DATASET
# ============================================================

class CEUSOLSegDataset(Dataset):
    def __init__(self, root, split="test", img_size=384):
        self.root = root
        self.image_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "annotations")
        self.split = split
        self.img_size = img_size
        self.pairs = self.collect_pairs()

        if len(self.pairs) == 0:
            raise RuntimeError(f"No samples found for split: {split}")

        print(f"{split}: {len(self.pairs)} samples")

    def collect_pairs(self):
        split_file = os.path.join(self.root, f"{self.split}.txt")
        names = []

        if os.path.exists(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip().replace(",", " ")
                    if line:
                        token = line.split()[0]
                        names.append(os.path.splitext(os.path.basename(token))[0])
        else:
            for f in os.listdir(self.image_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    names.append(os.path.splitext(f)[0])

        pairs = []
        for name in names:
            img_path, mask_path = None, None

            for ext in [".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png"]:
                p = os.path.join(self.image_dir, name + ext)
                if os.path.exists(p):
                    img_path = p
                    break

            for ext in [".PNG", ".png", ".JPG", ".jpg", ".JPEG", ".jpeg"]:
                p = os.path.join(self.mask_dir, name + ext)
                if os.path.exists(p):
                    mask_path = p
                    break

            if img_path is not None and mask_path is not None:
                pairs.append((img_path, mask_path))

        return pairs

    def load_image_mask(self, img_path, mask_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_rgb = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if mask_rgb is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)

        # Same mask conversion used in the CEUS training code:
        # any non-background color channel becomes lesion foreground.
        mask = (
            (mask_rgb[:, :, 0] > 20) |
            (mask_rgb[:, :, 1] > 20) |
            (mask_rgb[:, :, 2] > 20)
        ).astype(np.float32)

        original_shape = mask.shape[:2]  # H, W

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1)).copy()
        mask = mask[None, :, :].astype(np.float32)

        return img, mask, original_shape

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img, mask, original_shape = self.load_image_mask(img_path, mask_path)
        name = os.path.splitext(os.path.basename(img_path))[0]
        return (
            torch.from_numpy(img).float(),
            torch.from_numpy(mask).float(),
            name,
            original_shape[0],
            original_shape[1],
        )


# ============================================================
# MODEL
# ============================================================

def build_model(model_name="unetplusplus", encoder="resnet50"):
    if smp is None:
        raise ImportError(
            "Please install required packages:\n"
            "pip install segmentation-models-pytorch timm"
        )

    model_name = model_name.lower()

    def create(weights):
        if model_name == "unetplusplus":
            return smp.UnetPlusPlus(
                encoder_name=encoder,
                encoder_weights=weights,
                in_channels=3,
                classes=1,
                activation=None,
                decoder_attention_type="scse"
            )

        if model_name == "deeplabv3plus":
            return smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=weights,
                in_channels=3,
                classes=1,
                activation=None
            )

        if model_name == "fpn":
            return smp.FPN(
                encoder_name=encoder,
                encoder_weights=weights,
                in_channels=3,
                classes=1,
                activation=None
            )

        raise ValueError("model must be unetplusplus, deeplabv3plus, or fpn")

    # For evaluation checkpoint loading, encoder weights are not needed.
    return create(None)


# ============================================================
# POSTPROCESSING
# ============================================================

def postprocess_binary(mask, min_area=80, keep_components=2):
    mask = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num <= 1:
        return mask

    comps = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            comps.append((area, i))

    if not comps:
        return np.zeros_like(mask)

    comps = sorted(comps, reverse=True)[:keep_components]
    out = np.zeros_like(mask)
    for _, idx in comps:
        out[labels == idx] = 1

    return out


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict_tta(model, x):
    model.eval()
    probs = []

    p = torch.sigmoid(model(x))
    probs.append(p)

    xf = torch.flip(x, dims=[3])
    p = torch.sigmoid(model(xf))
    probs.append(torch.flip(p, dims=[3]))

    xf = torch.flip(x, dims=[2])
    p = torch.sigmoid(model(xf))
    probs.append(torch.flip(p, dims=[2]))

    xf = torch.flip(x, dims=[2, 3])
    p = torch.sigmoid(model(xf))
    probs.append(torch.flip(p, dims=[2, 3]))

    return torch.stack(probs, dim=0).mean(dim=0)


# ============================================================
# 2D SURFACE DISTANCE METRICS
# ============================================================

def surface_distances_2d(pred, gt, spacing=(1.0, 1.0)):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if not gt.any():
        return np.nan, np.nan, "gt_empty"

    if gt.any() and not pred.any():
        h, w = gt.shape
        penalty = math.sqrt(((h - 1) * spacing[0]) ** 2 + ((w - 1) * spacing[1]) ** 2)
        return float(penalty), float(penalty), "complete_miss_penalty"

    structure = np.ones((3, 3), dtype=bool)

    pred_surface = pred ^ binary_erosion(pred, structure=structure, border_value=0)
    gt_surface = gt ^ binary_erosion(gt, structure=structure, border_value=0)

    if not pred_surface.any() or not gt_surface.any():
        h, w = gt.shape
        penalty = math.sqrt(((h - 1) * spacing[0]) ** 2 + ((w - 1) * spacing[1]) ** 2)
        return float(penalty), float(penalty), "surface_failure_penalty"

    dt_to_gt = distance_transform_edt(~gt_surface, sampling=spacing)
    dt_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)

    d_pred_to_gt = dt_to_gt[pred_surface]
    d_gt_to_pred = dt_to_pred[gt_surface]

    distances = np.concatenate([d_pred_to_gt, d_gt_to_pred]).astype(np.float64)

    hd95 = float(np.percentile(distances, 95))
    asd = float(np.mean(distances))

    return hd95, asd, "valid"


# ============================================================
# METRICS
# ============================================================

def safe_div(num, den, default=0.0):
    if den == 0:
        return float(default)
    return float(num / den)


def compute_metrics(prob, gt, threshold=0.5, min_area=80, keep_components=2):
    """
    prob: 2D probability map
    gt: 2D binary ground truth
    """
    gt = (gt > 0.5).astype(np.uint8)

    raw_pred = (prob >= threshold).astype(np.uint8)
    pred = postprocess_binary(raw_pred, min_area=min_area, keep_components=keep_components)

    tp = int(np.logical_and(pred == 1, gt == 1).sum())
    fp = int(np.logical_and(pred == 1, gt == 0).sum())
    fn = int(np.logical_and(pred == 0, gt == 1).sum())
    tn = int(np.logical_and(pred == 0, gt == 0).sum())

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
        if len(np.unique(gt.reshape(-1))) == 2:
            auc = float(roc_auc_score(gt.reshape(-1), prob.reshape(-1)))
        else:
            auc = np.nan
    except Exception:
        auc = np.nan

    hd95, asd, dist_status = surface_distances_2d(pred, gt, spacing=(1.0, 1.0))

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
        "GT_pixels": int(gt.sum()),
        "Pred_pixels": int(pred.sum()),
        "Distance_status": dist_status,
    }


def mean_sd(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return np.nan, np.nan, 0
    if len(arr) == 1:
        return float(arr[0]), 0.0, 1
    return float(arr.mean()), float(arr.std(ddof=1)), int(len(arr))


def format_pm(m, s, decimals=4):
    if not np.isfinite(m):
        return "NA"
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


# ============================================================
# EVALUATION
# ============================================================

METRIC_KEYS = [
    "Dice", "IoU", "Precision", "Recall", "F1", "Accuracy",
    "Sensitivity", "Specificity", "PPV", "NPV", "AUC", "HD95", "ASD"
]


@torch.no_grad()
def evaluate_test_set(model, loader, device, threshold, min_area, keep_components, output_dir):
    per_case_rows = []

    for x, y, name, orig_h, orig_w in tqdm(loader, desc="Evaluating CEUS-OLSeg test set"):
        x = x.to(device, non_blocking=True)

        prob = predict_tta(model, x)
        prob_np = prob[0, 0].detach().cpu().numpy()
        gt_np = y[0, 0].numpy()

        metrics = compute_metrics(
            prob_np,
            gt_np,
            threshold=threshold,
            min_area=min_area,
            keep_components=keep_components
        )

        row = {
            "Case": name[0],
            "Threshold": threshold,
            "Original_H": int(orig_h[0]),
            "Original_W": int(orig_w[0]),
        }
        row.update(metrics)
        per_case_rows.append(row)

    per_case_path = os.path.join(output_dir, "CEUS_OLSeg_per_case_metrics.csv")
    fields = ["Case", "Threshold", "Original_H", "Original_W"] + METRIC_KEYS + [
        "TP", "FP", "FN", "TN", "GT_pixels", "Pred_pixels", "Distance_status"
    ]

    with open(per_case_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_case_rows)

    summary = {
        "Group": "CEUS-OLSeg test set",
        "N_cases": len(per_case_rows),
    }

    for m in METRIC_KEYS:
        mean, sd, n = mean_sd([float(r[m]) for r in per_case_rows])
        summary[f"{m}_mean"] = mean
        summary[f"{m}_sd"] = sd
        summary[f"{m}_mean_pm_sd"] = format_pm(mean, sd)
        summary[f"{m}_n"] = n

    summary_path = os.path.join(output_dir, "CEUS_OLSeg_summary_mean_sd.csv")
    fields = ["Group", "N_cases"]
    for m in METRIC_KEYS:
        fields.extend([f"{m}_mean", f"{m}_sd", f"{m}_mean_pm_sd", f"{m}_n"])

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(summary)

    return per_case_path, summary_path, summary, per_case_rows


@torch.no_grad()
def threshold_sensitivity(model, loader, device, thresholds, min_area, keep_components, output_dir):
    rows = []

    for th in thresholds:
        metric_values = {m: [] for m in METRIC_KEYS}

        for x, y, name, orig_h, orig_w in tqdm(loader, desc=f"Threshold {th:.2f}", leave=False):
            x = x.to(device, non_blocking=True)

            prob = predict_tta(model, x)
            prob_np = prob[0, 0].detach().cpu().numpy()
            gt_np = y[0, 0].numpy()

            m = compute_metrics(
                prob_np,
                gt_np,
                threshold=float(th),
                min_area=min_area,
                keep_components=keep_components
            )

            for key in METRIC_KEYS:
                if np.isfinite(m[key]):
                    metric_values[key].append(m[key])

        row = {"Threshold": float(th)}
        for key in METRIC_KEYS:
            mean, sd, n = mean_sd(metric_values[key])
            row[f"{key}_mean"] = mean
            row[f"{key}_sd"] = sd
            row[f"{key}_mean_pm_sd"] = format_pm(mean, sd)
            row[f"{key}_n"] = n

        rows.append(row)

    out_path = os.path.join(output_dir, "CEUS_OLSeg_threshold_sensitivity.csv")
    fields = ["Threshold"]
    for m in METRIC_KEYS:
        fields.extend([f"{m}_mean", f"{m}_sd", f"{m}_mean_pm_sd", f"{m}_n"])

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return out_path, rows


def print_summary(summary, threshold):
    print("\n" + "=" * 90)
    print("CEUS-OLSeg TEST EVALUATION SUMMARY")
    print("=" * 90)
    print(f"Threshold: {threshold:.4f}")
    print(f"N cases: {summary['N_cases']}")
    for m in METRIC_KEYS:
        print(f"{m}: {summary[f'{m}_mean_pm_sd']}")
    print("=" * 90)


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--threshold", type=float, default=-1.0,
                        help="If < 0, use checkpoint best_threshold.")
    parser.add_argument("--min_area", type=int, default=DEFAULT_MIN_AREA)
    parser.add_argument("--keep_components", type=int, default=DEFAULT_KEEP_COMPONENTS)

    parser.add_argument("--model", type=str, default="",
                        choices=["", "unetplusplus", "deeplabv3plus", "fpn"])
    parser.add_argument("--encoder", type=str, default="")

    parser.add_argument("--skip_threshold_sensitivity", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found:\n{args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)

    ckpt_args = ckpt.get("args", {})
    model_name = args.model if args.model else ckpt_args.get("model", "unetplusplus")
    encoder = args.encoder if args.encoder else ckpt_args.get("encoder", "resnet50")

    threshold = args.threshold
    if threshold < 0:
        threshold = float(ckpt.get("best_threshold", 0.5))

    print("Checkpoint:", args.checkpoint)
    print("Model:", model_name)
    print("Encoder:", encoder)
    print("Best validation Dice:", ckpt.get("best_val_dice", "NA"))
    print("Evaluation threshold:", threshold)

    model = build_model(model_name=model_name, encoder=encoder).to(device)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    model.eval()

    test_ds = CEUSOLSegDataset(args.data_root, "test", args.img_size)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    per_case_path, summary_path, summary, _ = evaluate_test_set(
        model=model,
        loader=test_loader,
        device=device,
        threshold=threshold,
        min_area=args.min_area,
        keep_components=args.keep_components,
        output_dir=args.output_dir,
    )

    print_summary(summary, threshold)

    if not args.skip_threshold_sensitivity:
        thresholds = np.arange(0.10, 0.91, 0.05)
        th_path, th_rows = threshold_sensitivity(
            model=model,
            loader=test_loader,
            device=device,
            thresholds=thresholds,
            min_area=args.min_area,
            keep_components=args.keep_components,
            output_dir=args.output_dir,
        )

        best_row = max(th_rows, key=lambda r: r["Dice_mean"] if np.isfinite(r["Dice_mean"]) else -1)
        print("\nThreshold sensitivity saved:", th_path)
        print(
            f"Best threshold by test-set Dice sensitivity: "
            f"{best_row['Threshold']:.2f}, Dice={best_row['Dice_mean_pm_sd']}"
        )

    print("\nSaved files:")
    print("1)", per_case_path)
    print("2)", summary_path)
    if not args.skip_threshold_sensitivity:
        print("3)", th_path)


if __name__ == "__main__":
    main()
