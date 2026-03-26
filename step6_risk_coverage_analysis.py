import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser("STEP-6: Risk–Coverage Analysis")
    parser.add_argument("--step1_root", required=True)
    parser.add_argument("--step5_root", required=True)
    parser.add_argument("--out_root", required=True)
    return parser.parse_args()


def find_dice_column(df):
    for c in ["dice", "dice_x", "dice_y", "Dice", "DICE"]:
        if c in df.columns:
            return c
    raise KeyError(
        f"No Dice column found. Available columns: {list(df.columns)}"
    )


def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    # ---- Load CSVs
    step1_csv = os.path.join(args.step1_root, "step1_metrics.csv")
    step5_csv = os.path.join(args.step5_root, "hardness_ranking.csv")

    if not os.path.exists(step1_csv):
        raise FileNotFoundError(step1_csv)
    if not os.path.exists(step5_csv):
        raise FileNotFoundError(step5_csv)

    m1 = pd.read_csv(step1_csv)
    m5 = pd.read_csv(step5_csv)

    # ---- Sanity checks
    assert "stem" in m1.columns, "Missing 'stem' in STEP-1 CSV"
    assert "stem" in m5.columns, "Missing 'stem' in STEP-5 CSV"
    assert "hardness" in m5.columns, "Missing 'hardness' in STEP-5 CSV"

    # ---- Merge
    df = pd.merge(m5, m1, on="stem", how="inner")

    print("[DEBUG] Merged columns:", list(df.columns))

    dice_col = find_dice_column(df)
    print(f"[OK] Using Dice column: {dice_col}")

    # ---- Sort by hardness (hardest first)
    df = df.sort_values("hardness", ascending=False).reset_index(drop=True)

    dice_vals = df[dice_col].values
    N = len(dice_vals)

    # ---- Risk–Coverage
    coverages, risks = [], []

    for k in range(1, N + 1):
        selected = dice_vals[:k]
        risk = 1.0 - np.mean(selected)
        coverage = k / N
        coverages.append(coverage)
        risks.append(risk)

    rc_df = pd.DataFrame({
        "coverage": coverages,
        "risk": risks
    })

    # ---- Save CSV
    csv_path = os.path.join(args.out_root, "risk_coverage.csv")
    rc_df.to_csv(csv_path, index=False)

    # ---- Plot
    plt.figure(figsize=(6, 5))
    plt.plot(coverages, risks, marker="o", linewidth=2)
    plt.xlabel("Coverage", fontsize=12)
    plt.ylabel("Risk (1 − Dice)", fontsize=12)
    plt.title("Risk–Coverage Curve", fontsize=13)
    plt.grid(True)
    plt.tight_layout()

    png_path = os.path.join(args.out_root, "risk_coverage.png")
    plt.savefig(png_path, dpi=300)
    plt.close()

    print("\nSTEP-6 COMPLETED SUCCESSFULLY")
    print(f"[OK] CSV  : {csv_path}")
    print(f"[OK] PNG  : {png_path}")


if __name__ == "__main__":
    main()
