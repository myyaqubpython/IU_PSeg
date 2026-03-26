import os
import argparse
import csv
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn

# ✅ FIXED MODEL
from IU_PSeg import IU_PSegNet


# ========================
# METRICS
# ========================
def dice_iou(pred, gt, eps=1e-6):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    inter = (pred * gt).sum()
    dice = (2 * inter + eps) / (pred.sum() + gt.sum() + eps)

    union = pred.sum() + gt.sum() - inter
    iou = (inter + eps) / (union + eps)

    return dice, iou


# ========================
# UTILS
# ========================
def normalize(x):
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def colorize(x):
    x = normalize(x)
    x = (x * 255).astype(np.uint8)
    return cv2.applyColorMap(x, cv2.COLORMAP_JET)


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


# ========================
# LOAD IDS
# ========================
def load_ids(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


# ========================
# LOAD IMAGE (FIXED FOR YOUR DATASET)
# ========================
def load_sample(data_root, sid):
    img_path = os.path.join(data_root, "images", sid + ".JPG")
    mask_path = os.path.join(data_root, "annotations", sid + ".PNG")

    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0)

    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (256, 256))
    mask = (mask > 0).astype(np.uint8)

    return img, mask


# ========================
# MAIN
# ========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_txt", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_root", required=True)

    parser.add_argument("--mc_samples", type=int, default=20)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ========================
    # OUTPUT DIRS
    # ========================
    mean_dir = os.path.join(args.out_root, "mean")
    epi_dir = os.path.join(args.out_root, "epistemic")
    ale_dir = os.path.join(args.out_root, "aleatoric")
    vis_dir = os.path.join(args.out_root, "vis")

    for d in [mean_dir, epi_dir, ale_dir, vis_dir]:
        os.makedirs(d, exist_ok=True)

    # ========================
    # LOAD IDS
    # ========================
    ids = load_ids(args.split_txt)

    # ========================
    # MODEL
    # ========================
    model = IU_PSegNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    enable_mc_dropout(model)

    print("✅ Model loaded")

    # ========================
    # INFERENCE
    # ========================
    csv_rows = []
    csv_path = os.path.join(args.out_root, "uncertainty_metrics.csv")

    with torch.no_grad():
        for sid in tqdm(ids):

            img, gt = load_sample(args.data_root, sid)
            img = img.to(device)

            preds = []

            for _ in range(args.mc_samples):
                p = torch.sigmoid(model(img))
                preds.append(p.squeeze().cpu().numpy())

            preds = np.stack(preds, axis=0)

            mean_prob = preds.mean(axis=0)
            epistemic = preds.var(axis=0)
            aleatoric = mean_prob * (1.0 - mean_prob)

            pred_bin = (mean_prob > 0.25).astype(np.uint8)  # 🔥 best threshold

            # metrics
            dice, iou = dice_iou(pred_bin, gt)

            csv_rows.append([sid, dice, iou,
                             float(epistemic.mean()),
                             float(aleatoric.mean())])

            # save maps
            cv2.imwrite(os.path.join(mean_dir, f"{sid}.png"),
                        (mean_prob * 255).astype(np.uint8))

            cv2.imwrite(os.path.join(epi_dir, f"{sid}.png"),
                        colorize(epistemic))

            cv2.imwrite(os.path.join(ale_dir, f"{sid}.png"),
                        colorize(aleatoric))

            # visualization panel
            img_vis = (img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gt_rgb = cv2.cvtColor(gt * 255, cv2.COLOR_GRAY2BGR)
            pred_rgb = cv2.cvtColor(pred_bin * 255, cv2.COLOR_GRAY2BGR)

            panel = np.concatenate([
                img_vis,
                gt_rgb,
                pred_rgb,
                colorize(epistemic),
                colorize(aleatoric)
            ], axis=1)

            cv2.imwrite(os.path.join(vis_dir, f"{sid}.png"), panel)

    # ========================
    # SAVE CSV
    # ========================
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "dice", "iou", "epistemic", "aleatoric"])
        writer.writerows(csv_rows)

    print("🔥 DONE: Uncertainty results saved!")


if __name__ == "__main__":
    main()