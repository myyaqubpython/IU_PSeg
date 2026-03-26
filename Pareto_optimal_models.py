import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========================
# FULL MODEL LIST (UPDATE IF NEEDED)
# ========================
models = [
    "UNet", "UNet++", "DeepLabV3+", "Attention-UNet",
    "TransUNet", "Swin-UNet", "nnUNet",
    "DHC", "Co-BioNet", "GA", "TAK", "IU-PSeg"
]

# GFLOPs
flops = np.array([
    210, 250, 320, 230,
    450, 410, 380,
    566.76, 285.68, 143.40, 150.96, 128.35
])

# Parameters (Millions)
params = np.array([
    31, 34, 42, 33,
    105, 62, 45,
    37.8, 18.8, 18.0, 18.5, 14.72
])

# Dice scores
dice = np.array([
    0.89, 0.91, 0.92, 0.90,
    0.94, 0.945, 0.946,
    0.91, 0.93, 0.92, 0.94, 0.982
])

# ========================
# 1️⃣ 3D TRADE-OFF PLOT
# ========================
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(flops, params, dice, s=120)

# highlight OUR MODEL
ours_idx = models.index("IU-PSeg")
ax.scatter(flops[ours_idx], params[ours_idx], dice[ours_idx],
           s=200, marker='*')

# annotate
for i, name in enumerate(models):
    ax.text(flops[i], params[i], dice[i], name, fontsize=8)

ax.set_xlabel("FLOPs (G)", fontsize=11)
ax.set_ylabel("Params (M)", fontsize=11)
ax.set_zlabel("Dice Score", fontsize=11)

ax.view_init(elev=20, azim=135)

plt.tight_layout()
plt.savefig("3D_tradeoff_ALL.png", dpi=600)
plt.close()

print("✅ 3D trade-off saved")

# ========================
# 2️⃣ PARETO FRONTIER
# ========================
pareto = []

for i in range(len(models)):
    dominated = False
    for j in range(len(models)):
        if (flops[j] <= flops[i] and dice[j] >= dice[i]) and j != i:
            dominated = True
            break
    if not dominated:
        pareto.append(i)

# ========================
# SORT FOR FRONTIER LINE
# ========================
pareto_sorted = sorted(pareto, key=lambda x: flops[x])

# ========================
# PLOT
# ========================
plt.figure(figsize=(6,5))

# all models
plt.scatter(flops, dice, s=80, label="All Models")

# pareto points
plt.scatter(flops[pareto], dice[pareto],
            s=180, marker='*', label="Pareto Optimal")

# draw frontier line
pareto_flops = flops[pareto_sorted]
pareto_dice  = dice[pareto_sorted]

plt.plot(pareto_flops, pareto_dice, linestyle='--')

# highlight ours
plt.scatter(flops[ours_idx], dice[ours_idx],
            s=220, marker='*')

# annotate
for i, name in enumerate(models):
    plt.text(flops[i]+5, dice[i], name, fontsize=8)

plt.xlabel("FLOPs (G)", fontsize=11)
plt.ylabel("Dice Score", fontsize=11)

plt.title("Pareto Frontier Analysis (All Models)", fontsize=12)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("Pareto_ALL.png", dpi=600)
plt.show()

print("🔥 Pareto frontier saved")