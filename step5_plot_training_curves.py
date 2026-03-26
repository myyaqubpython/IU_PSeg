import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\outputs\metrics.csv"
SAVE_DIR = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\training_plots"

os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# -----------------------
# Dice curve
# -----------------------
plt.figure(figsize=(7, 5))
plt.plot(df["epoch"], df["dice"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.title("Training Dice Curve")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "dice_curve.png"), dpi=600)
plt.close()

# -----------------------
# IoU curve
# -----------------------
plt.figure(figsize=(7, 5))
plt.plot(df["epoch"], df["iou"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("Training IoU Curve")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "iou_curve.png"), dpi=600)
plt.close()

# -----------------------
# Precision / Recall / F1
# -----------------------
plt.figure(figsize=(7, 5))
plt.plot(df["epoch"], df["precision"], label="Precision")
plt.plot(df["epoch"], df["recall"], label="Recall")
plt.plot(df["epoch"], df["f1"], label="F1-score")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Training Precision / Recall / F1")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "precision_recall_f1_curve.png"), dpi=600)
plt.close()

# -----------------------
# HD95 curve
# -----------------------
plt.figure(figsize=(7, 5))
plt.plot(df["epoch"], df["hd95"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("HD95")
plt.title("Training HD95 Curve")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "hd95_curve.png"), dpi=600)
plt.close()

# -----------------------
# Loss curve if available
# -----------------------
if "loss" in df.columns:
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"), dpi=600)
    plt.close()
    print("Loss curve saved.")
else:
    print("No 'loss' column found in metrics.csv, so loss curve was not plotted.")

print(f"All plots saved in: {SAVE_DIR}")