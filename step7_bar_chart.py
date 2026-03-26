import pandas as pd
import matplotlib.pyplot as plt

CSV = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\ablation_outputs\ablation_summary.csv"

df = pd.read_csv(CSV)

models = df["model"]
dice = df["dice_mean"]
dice_std = df["dice_std"]

plt.figure(figsize=(8,5))
plt.bar(models, dice, yerr=dice_std)

plt.ylabel("Dice Score")
plt.title("Model Comparison (Dice)")
plt.xticks(rotation=20)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("bar_chart_dice.png", dpi=600)
plt.show()