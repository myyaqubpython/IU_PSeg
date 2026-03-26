import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def safe_col(df, name):
    if name not in df.columns:
        raise ValueError(f"Missing column '{name}' in CSV. Found: {list(df.columns)}")
    return df[name].values

# -----------------------------
# Main
# -----------------------------
def main(args):
    ensure_dir(args.out_root)

    # Load CSVs
    base = pd.read_csv(args.step1_metrics)
    gain = pd.read_csv(args.step8_metrics)

    # Normalize column names
    base_cols = base.columns.str.lower()
    gain_cols = gain.columns.str.lower()
    base.columns = base_cols
    gain.columns = gain_cols

    # Required columns
    base_case = safe_col(base, "stem")
    base_dice = safe_col(base, "dice")

    gain_case = safe_col(gain, "case")
    dice_before = safe_col(gain, "dice_before")
    dice_after = safe_col(gain, "dice_after")
    delta = safe_col(gain, "delta_dice")

    # -----------------------------
    # 1️⃣ Dice Boxplot
    # -----------------------------
    plt.figure(figsize=(6,5))
    plt.boxplot([dice_before, dice_after],
                labels=["Before Retraining", "After Retraining"],
                patch_artist=True,
                boxprops=dict(facecolor="#4C72B0"),
                medianprops=dict(color="black"))
    plt.ylabel("Dice Score")
    plt.title("Segmentation Performance Improvement")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_root, "dice_boxplot.png"), dpi=300)
    plt.close()

    # -----------------------------
    # 2️⃣ Hard Sample Recovery
    # -----------------------------
    hard_mask = dice_before < args.failure_thr
    recovered = np.sum((dice_after > args.failure_thr) & hard_mask)
    total_hard = np.sum(hard_mask)

    plt.figure(figsize=(6,4))
    plt.bar(["Hard Samples", "Recovered"],
            [total_hard, recovered],
            color=["#DD8452", "#55A868"])
    plt.ylabel("Number of Samples")
    plt.title("Hard Sample Recovery After Retraining")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_root, "hard_sample_recovery.png"), dpi=300)
    plt.close()

    # -----------------------------
    # 3️⃣ Failure Rate Reduction
    # -----------------------------
    fail_before = np.mean(dice_before < args.failure_thr)
    fail_after = np.mean(dice_after < args.failure_thr)

    plt.figure(figsize=(6,4))
    plt.bar(["Before", "After"],
            [fail_before, fail_after],
            color=["#C44E52", "#4C72B0"])
    plt.ylabel("Failure Rate")
    plt.title("Failure Rate Reduction")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_root, "failure_rate_reduction.png"), dpi=300)
    plt.close()

    # -----------------------------
    # 4️⃣ Summary CSV
    # -----------------------------
    summary = pd.DataFrame({
        "metric": [
            "Mean Dice (Before)",
            "Mean Dice (After)",
            "Median Dice (Before)",
            "Median Dice (After)",
            "Failure Rate (Before)",
            "Failure Rate (After)",
            "Recovered Hard Samples"
        ],
        "value": [
            np.mean(dice_before),
            np.mean(dice_after),
            np.median(dice_before),
            np.median(dice_after),
            fail_before,
            fail_after,
            recovered
        ]
    })

    summary.to_csv(os.path.join(args.out_root, "step9_summary.csv"), index=False)

    print("STEP-9 COMPLETED SUCCESSFULLY")
    print(f"[OK] Figures + summary saved to: {args.out_root}")

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step1_metrics", required=True)
    parser.add_argument("--step8_metrics", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--failure_thr", type=float, default=0.5)

    args = parser.parse_args()
    main(args)
