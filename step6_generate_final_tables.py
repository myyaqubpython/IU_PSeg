import pandas as pd

# ========================
# PATHS
# ========================
SUMMARY_CSV = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\ablation_outputs\ablation_summary.csv"
PVAL_CSV = r"C:\Users\M YAQUB\CSU2026\IU-PSeg\ablation_outputs\statistical_significance_results.csv"

# ========================
# LOAD DATA
# ========================
df = pd.read_csv(SUMMARY_CSV)

# Format Mean ± Std
def fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"

table = []

for _, row in df.iterrows():
    table.append([
        row["model"],
        fmt(row["dice_mean"], row["dice_std"]),
        fmt(row["iou_mean"], row["iou_std"]),
        fmt(row["precision_mean"], row["precision_std"]),
        fmt(row["recall_mean"], row["recall_std"]),
        fmt(row["f1_mean"], row["f1_std"]),
    ])

final_df = pd.DataFrame(table, columns=[
    "Model", "Dice", "IoU", "Precision", "Recall", "F1"
])

print("\n===== FINAL TABLE =====")
print(final_df)

# Save
final_df.to_csv("final_ablation_table.csv", index=False)
print("\nSaved: final_ablation_table.csv")

# ========================
# P-VALUE TABLE
# ========================
pval_df = pd.read_csv(PVAL_CSV)

print("\n===== P-VALUE RESULTS =====")
print(pval_df)

pval_df.to_csv("final_pvalues.csv", index=False)