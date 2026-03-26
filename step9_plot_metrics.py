import pandas as pd
import matplotlib.pyplot as plt

log_path = "OUT_STEP7_NEW/train_log.csv"
df = pd.read_csv(log_path)

plt.figure(figsize=(6,4))
plt.plot(df["epoch"], df["dice"], label="Dice", linewidth=2)
plt.plot(df["epoch"], df["iou"], label="IoU", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_dice_iou_vs_epoch.png", dpi=600)
plt.close()

plt.figure(figsize=(6,4))
plt.plot(df["epoch"], df["hd95"], color="red", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("HD95 (pixels)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_hd95_vs_epoch.png", dpi=600)
plt.close()
