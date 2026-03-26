import os
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

SAVE_ROOT = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\ablation_outputs"

OURS = os.path.join(SAVE_ROOT, "IU_PSeg_Ours_per_image_metrics.csv")
BASELINES = {
    "Baseline_UNet": os.path.join(SAVE_ROOT, "Baseline_UNet_per_image_metrics.csv"),
    "Ablation_1": os.path.join(SAVE_ROOT, "Ablation_1_per_image_metrics.csv"),
    "Ablation_2": os.path.join(SAVE_ROOT, "Ablation_2_per_image_metrics.csv"),
}

METRICS = ["dice", "iou", "precision", "recall", "f1"]

df_ours = pd.read_csv(OURS)

rows = []

for name, path in BASELINES.items():
    if not os.path.exists(path):
        print(f"[WARNING] Missing file: {path}")
        continue

    df_base = pd.read_csv(path)

    for metric in METRICS:
        x = df_ours[metric].values
        y = df_base[metric].values

        t_stat, t_p = ttest_rel(x, y)
        try:
            w_stat, w_p = wilcoxon(x, y)
        except Exception:
            w_stat, w_p = None, None

        rows.append({
            "comparison": f"IU_PSeg_Ours vs {name}",
            "metric": metric,
            "ours_mean": x.mean(),
            "baseline_mean": y.mean(),
            "paired_ttest_p": t_p,
            "wilcoxon_p": w_p
        })

out_csv = os.path.join(SAVE_ROOT, "statistical_significance_results.csv")
pd.DataFrame(rows).to_csv(out_csv, index=False)

print(f"Saved: {out_csv}")
print("\nInterpretation:")
print("p < 0.05  => statistically significant")
print("p < 0.01  => highly significant")