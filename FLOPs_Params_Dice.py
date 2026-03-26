import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========================
# DATA (UPDATE IF NEEDED)
# ========================
models = ["DHC", "Co-BioNet", "GA", "TAK", "IU-PSeg"]

flops =  [566.76, 285.68, 143.40, 150.96, 128.35]   # GFLOPs
params = [37.8,   18.8,   18.0,   18.5,   14.72]     # Millions
dice =   [0.91,   0.93,   0.92,   0.94,   0.982]

# ========================
# PLOT
# ========================
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(flops, params, dice, s=120)

# annotate
for i, name in enumerate(models):
    ax.text(flops[i], params[i], dice[i], name, fontsize=9)

ax.set_xlabel("FLOPs (G)", fontsize=11)
ax.set_ylabel("Params (M)", fontsize=11)
ax.set_zlabel("Dice Score", fontsize=11)

ax.view_init(elev=20, azim=135)

plt.tight_layout()
plt.savefig("3D_tradeoff.png", dpi=600)
plt.show()