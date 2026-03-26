import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from medpy.metric.binary import hd95, assd
from skimage.metrics import structural_similarity as ssim

# ===============================
# 🔹 METRIC FUNCTIONS
# ===============================

def dice_score(gt, pred):
    intersection = np.sum(gt * pred)
    return (2. * intersection) / (np.sum(gt) + np.sum(pred) + 1e-8)

def iou_score(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred) - intersection
    return intersection / (union + 1e-8)

# ===============================
# 🔹 LOAD DATA (NPY FORMAT)
# ===============================

def load_data(folder):
    images = []
    gts = []
    preds = []
    uncertainties = []

    for file in os.listdir(folder):
        if file.endswith("_img.npy"):
            base = file.replace("_img.npy", "")
            images.append(np.load(os.path.join(folder, base + "_img.npy")))
            gts.append(np.load(os.path.join(folder, base + "_gt.npy")))
            preds.append(np.load(os.path.join(folder, base + "_pred.npy")))
            uncertainties.append(np.load(os.path.join(folder, base + "_unc.npy")))

    return images, gts, preds, uncertainties

# ===============================
# 🔹 EVALUATION LOOP
# ===============================

def evaluate_all(images, gts, preds):
    dice_list, iou_list, hd_list, asd_list = [], [], [], []

    for i in range(len(images)):
        gt = gts[i]
        pred = preds[i]

        gt = (gt > 0).astype(np.uint8)
        pred = (pred > 0).astype(np.uint8)

        dice = dice_score(gt, pred)
        iou = iou_score(gt, pred)

        try:
            hd = hd95(pred, gt)
            asd = assd(pred, gt)
        except:
            hd, asd = 0, 0

        dice_list.append(dice)
        iou_list.append(iou)
        hd_list.append(hd)
        asd_list.append(asd)

    return dice_list, iou_list, hd_list, asd_list

# ===============================
# 🔹 VISUALIZATION FUNCTIONS
# ===============================

def visualize_sample(img, gt, pred, unc, idx, save_dir):
    error_map = np.abs(gt - pred)

    fig, ax = plt.subplots(1, 5, figsize=(20, 4))

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("MRI")

    ax[1].imshow(gt, cmap='gray')
    ax[1].set_title("Ground Truth")

    ax[2].imshow(pred, cmap='gray')
    ax[2].set_title("Prediction")

    ax[3].imshow(error_map, cmap='hot')
    ax[3].set_title("Error Map")

    ax[4].imshow(unc, cmap='jet')
    ax[4].set_title("Uncertainty")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"sample_{idx}.png"), dpi=300)
    plt.close()

# ===============================
# 🔹 GRAPH PLOTTING
# ===============================

def plot_metrics(dice, iou, hd, asd, save_dir):
    df = pd.DataFrame({
        "Dice": dice,
        "IoU": iou,
        "HD95": hd,
        "ASD": asd
    })

    # Box Plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df)
    plt.title("Segmentation Performance Distribution")
    plt.savefig(os.path.join(save_dir, "boxplot_metrics.png"), dpi=300)
    plt.close()

    # Bar Plot (Mean)
    means = df.mean()
    plt.figure(figsize=(6, 4))
    means.plot(kind='bar')
    plt.title("Average Metrics")
    plt.savefig(os.path.join(save_dir, "bar_metrics.png"), dpi=300)
    plt.close()

# ===============================
# 🔹 MAIN
# ===============================

if __name__ == "__main__":

    data_path = "./results/"   # folder with .npy files
    save_dir = "./outputs/"
    os.makedirs(save_dir, exist_ok=True)

    images, gts, preds, uncertainties = load_data(data_path)

    dice, iou, hd, asd = evaluate_all(images, gts, preds)

    print("==== RESULTS ====")
    print("Dice:", np.mean(dice))
    print("IoU:", np.mean(iou))
    print("HD95:", np.mean(hd))
    print("ASD:", np.mean(asd))

    # Save visualizations (first 5 samples)
    for i in range(min(5, len(images))):
        visualize_sample(images[i], gts[i], preds[i], uncertainties[i], i, save_dir)

    # Plot graphs
    plot_metrics(dice, iou, hd, asd, save_dir)

    print("Results saved in:", save_dir)