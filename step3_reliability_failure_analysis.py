import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# LOAD CSV
# =========================================================
csv_path = "outputs/uncertainty/step2_uncertainty_metrics.csv"

df = pd.read_csv(csv_path)

print("\nDataset loaded:")
print(df.head())

# =========================================================
# BASIC STATS
# =========================================================
print("\n=== SUMMARY ===")
print(df.describe())

# =========================================================
# UNCERTAINTY vs PERFORMANCE
# =========================================================
plt.figure(figsize=(6, 5))
plt.scatter(df["dice"], df["epistemic"], alpha=0.7)

plt.xlabel("Dice Score")
plt.ylabel("Epistemic Uncertainty")
plt.title("Uncertainty vs Dice")

plt.grid(True)
plt.savefig("outputs/uncertainty/uncertainty_vs_dice.png")
plt.show()

# =========================================================
# FAILURE ANALYSIS
# =========================================================
threshold = 0.5

failures = df[df["dice"] < threshold]
success = df[df["dice"] >= threshold]

print(f"\nTotal samples: {len(df)}")
print(f"Failures (<{threshold} Dice): {len(failures)}")
print(f"Success: {len(success)}")

# =========================================================
# COMPARE UNCERTAINTY
# =========================================================
fail_mean = failures["epistemic"].mean()
success_mean = success["epistemic"].mean()

print("\n=== UNCERTAINTY COMPARISON ===")
print(f"Failure uncertainty: {fail_mean:.4f}")
print(f"Success uncertainty: {success_mean:.4f}")

# =========================================================
# HISTOGRAM
# =========================================================
plt.figure(figsize=(6, 5))

plt.hist(failures["epistemic"], bins=20, alpha=0.6, label="Failure")
plt.hist(success["epistemic"], bins=20, alpha=0.6, label="Success")

plt.xlabel("Epistemic Uncertainty")
plt.ylabel("Count")
plt.title("Uncertainty Distribution")
plt.legend()

plt.savefig("outputs/uncertainty/uncertainty_hist.png")
plt.show()

# =========================================================
# RELIABILITY (CALIBRATION-LIKE)
# =========================================================
bins = np.linspace(0, 1, 6)
df["bin"] = np.digitize(df["dice"], bins)

bin_acc = []
bin_unc = []

for b in sorted(df["bin"].unique()):
    subset = df[df["bin"] == b]
    if len(subset) == 0:
        continue

    bin_acc.append(subset["dice"].mean())
    bin_unc.append(subset["epistemic"].mean())

plt.figure(figsize=(6, 5))
plt.plot(bin_acc, bin_unc, marker="o")

plt.xlabel("Dice (Accuracy)")
plt.ylabel("Uncertainty")
plt.title("Reliability Curve")

plt.grid(True)
plt.savefig("outputs/uncertainty/reliability_curve.png")
plt.show()

# =========================================================
# CORRELATION
# =========================================================
corr = df["dice"].corr(df["epistemic"])

print("\n=== CORRELATION ===")
print(f"Correlation (Dice vs Uncertainty): {corr:.4f}")

if corr < 0:
    print("✅ GOOD: Higher uncertainty → lower Dice")
else:
    print("⚠ WARNING: Uncertainty not aligned with errors")

# =========================================================
# SAVE SUMMARY
# =========================================================
summary_path = "outputs/uncertainty/analysis_summary.txt"

with open(summary_path, "w") as f:
    f.write("=== STEP 3 ANALYSIS ===\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Failures: {len(failures)}\n")
    f.write(f"Failure uncertainty: {fail_mean:.4f}\n")
    f.write(f"Success uncertainty: {success_mean:.4f}\n")
    f.write(f"Correlation: {corr:.4f}\n")

print(f"\n✅ Analysis saved: {summary_path}")