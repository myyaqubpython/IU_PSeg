import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# ✅ FIXED IMPORT
from IU_PSeg import OTU3DDataset, IU_PSegNet

# ========================
# PATHS
# ========================
DATA_ROOT = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d"
SPLIT_FILE = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d\test.txt"  # 🔥 use test for paper
SAVE_ROOT = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\ablation_outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_ROOT, exist_ok=True)

# ========================
# MODELS
# ========================
MODELS = {
    "Baseline":   r"C:\Users\M YAQUB\CSU2026\IU-PSeg\outputs\baseline_best.pth",
    "No_Attention": r"C:\Users\M YAQUB\CSU2026\IU-PSeg\outputs\ablation1_best.pth",
    "No_Refinement": r"C:\Users\M YAQUB\CSU2026\IU-PSeg\outputs\ablation2_best.pth",
    "IU_PSeg_Ours": r"C:\Users\M YAQUB\CSU2026\IU-PSeg\outputs_upgraded\best_dice.pth",
}

# ========================
# DATA
# ========================
dataset = OTU3DDataset(DATA_ROOT, SPLIT_FILE, augment=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ========================
# METRICS
# ========================
def compute_metrics(pred, gt):
    pred = (pred > 0.25).astype(np.float32)  # 🔥 best threshold
    gt = gt.astype(np.float32)

    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))

    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    acc = (tp + tn + 1e-6) / (tp + tn + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = (2 * precision * recall + 1e-6) / (precision + recall + 1e-6)

    return dice, iou, acc, precision, recall, f1

# ========================
# EVALUATION
# ========================
def eval_model(ckpt_path, model_name):
    if not os.path.exists(ckpt_path):
        print(f"[WARNING] Missing checkpoint: {ckpt_path}")
        return None, None

    # ✅ FIXED MODEL
    model = IU_PSegNet().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    all_metrics = []

    with torch.no_grad():
        for image, gt in tqdm(loader, desc=model_name):
            image = image.to(DEVICE)
            gt = gt.to(DEVICE)

            pred = torch.sigmoid(model(image))

            pred_np = pred.squeeze().cpu().numpy()
            gt_np = gt.squeeze().cpu().numpy()

            metrics = compute_metrics(pred_np, gt_np)
            all_metrics.append(metrics)

    all_metrics = np.array(all_metrics)

    means = np.mean(all_metrics, axis=0)
    stds = np.std(all_metrics, axis=0)

    return {
        "model": model_name,
        "dice_mean": means[0], "dice_std": stds[0],
        "iou_mean": means[1], "iou_std": stds[1],
        "acc_mean": means[2], "acc_std": stds[2],
        "precision_mean": means[3], "precision_std": stds[3],
        "recall_mean": means[4], "recall_std": stds[4],
        "f1_mean": means[5], "f1_std": stds[5],
    }, all_metrics

# ========================
# RUN ALL
# ========================
summaries = []
dice_data = {}
iou_data = {}

for model_name, ckpt_path in MODELS.items():
    summary, all_metrics = eval_model(ckpt_path, model_name)

    if summary is not None:
        summaries.append(summary)
        dice_data[model_name] = all_metrics[:, 0]
        iou_data[model_name] = all_metrics[:, 1]

# ========================
# SAVE CSV
# ========================
summary_csv = os.path.join(SAVE_ROOT, "ablation_summary.csv")

with open(summary_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "model",
        "dice_mean", "dice_std",
        "iou_mean", "iou_std",
        "acc_mean", "acc_std",
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "f1_mean", "f1_std"
    ])

    for s in summaries:
        writer.writerow(list(s.values()))

print(f"Saved: {summary_csv}")

# ========================
# BOXPLOTS
# ========================
plt.figure(figsize=(10,5))
plt.boxplot(dice_data.values(), tick_labels=dice_data.keys())
plt.ylabel("Dice")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "dice_boxplot.png"), dpi=600)

plt.figure(figsize=(10,5))
plt.boxplot(iou_data.values(), tick_labels=iou_data.keys())
plt.ylabel("IoU")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_ROOT, "iou_boxplot.png"), dpi=600)

print("🔥 Ablation study completed successfully!")