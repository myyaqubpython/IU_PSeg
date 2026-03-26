import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# -------------------------------------------------
# Utility: find metrics CSV robustly
# -------------------------------------------------
def find_metrics_csv(step_root):
    step_root = Path(step_root)

    search_dirs = [
        step_root,
        step_root.parent / "OUT_STEP1"
    ]

    candidates = [
        "step2_metrics.csv",
        "metrics.csv",
        "step1_metrics.csv"
    ]

    for d in search_dirs:
        for c in candidates:
            p = d / c
            if p.exists():
                print(f"[OK] Using metrics file: {p}")
                return p

    raise FileNotFoundError(
        f"No metrics CSV found. Looked in:\n"
        + "\n".join(str(d) for d in search_dirs)
        + f"\nExpected one of: {candidates}"
    )

# -------------------------------------------------
# Hardness score computation
# -------------------------------------------------
def compute_hardness(df):
    """
    Hardness score combines:
    - low Dice (main)
    - high ECE (if exists)
    - high uncertainty (if exists)
    """

    eps = 1e-6

    # ---- Dice ----
    if "dice" not in df.columns:
        raise ValueError("Required column 'dice' not found in metrics CSV")

    dice_term = 1.0 - df["dice"].clip(0, 1)

    # ---- Epistemic uncertainty (optional) ----
    if "epi" in df.columns:
        epi_term = df["epi"] / (df["epi"].max() + eps)
    else:
        epi_term = 0.0

    # ---- Aleatoric uncertainty (optional) ----
    if "ale" in df.columns:
        ale_term = df["ale"] / (df["ale"].max() + eps)
    else:
        ale_term = 0.0

    # ---- Calibration error (optional) ----
    if "ece" in df.columns:
        ece_term = df["ece"] / (df["ece"].max() + eps)
    else:
        ece_term = 0.0

    hardness = (
        0.6 * dice_term +
        0.2 * epi_term +
        0.1 * ale_term +
        0.1 * ece_term
    )

    return hardness

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser("STEP-5: Hard Sample Mining")
    parser.add_argument("--step2_root", required=True, help="Path to OUT_STEP2 or OUT_STEP1")
    parser.add_argument("--out_root", required=True, help="Output folder for STEP-5")
    parser.add_argument("--top_k", type=int, default=20, help="Top-K hardest samples")

    args = parser.parse_args()

    step_root = Path(args.step2_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- Load metrics ----
    metrics_csv = find_metrics_csv(step_root)
    df = pd.read_csv(metrics_csv)

    # ---- Case column handling ----
    if "case" not in df.columns:
        if "name" in df.columns:
            df = df.rename(columns={"name": "case"})
        else:
            df["case"] = [f"case_{i:04d}" for i in range(len(df))]

    # ---- Compute hardness ----
    df["hardness"] = compute_hardness(df)

    # ---- Rank ----
    df = df.sort_values("hardness", ascending=False).reset_index(drop=True)

    # ---- Save full ranking ----
    full_csv = out_root / "hardness_ranking.csv"
    df.to_csv(full_csv, index=False)

    # ---- Save Top-K ----
    top_k = min(args.top_k, len(df))
    hard_df = df.iloc[:top_k]
    hard_csv = out_root / f"top_{top_k}_hard_samples.csv"
    hard_df.to_csv(hard_csv, index=False)

    # ---- Summary ----
    summary_txt = out_root / "summary.txt"
    with open(summary_txt, "w") as f:
        f.write("STEP-5: Hard Sample Mining\n")
        f.write("==========================\n")
        f.write(f"Metrics source : {metrics_csv}\n")
        f.write(f"Total samples  : {len(df)}\n")
        f.write(f"Top-K selected : {top_k}\n\n")
        f.write("Top-5 hardest cases:\n")
        for i in range(min(5, top_k)):
            row = hard_df.iloc[i]
            f.write(f"{i+1}. {row['case']} | hardness={row['hardness']:.4f} | dice={row['dice']:.4f}\n")

    print("\nSTEP-5 COMPLETED SUCCESSFULLY")
    print(f"[OK] Full ranking : {full_csv}")
    print(f"[OK] Top-K samples : {hard_csv}")

# -------------------------------------------------
if __name__ == "__main__":
    main()
