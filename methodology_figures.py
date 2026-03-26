import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle
import numpy as np

out = "C:/Users/M YAQUB/CSU2026/IU-PSeg/methodology_figures"

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Outer rounded panel
outer = FancyBboxPatch((0.2, 0.4), 15.4, 7.8, boxstyle="round,pad=0.02,rounding_size=0.15",
                       linewidth=1.8, edgecolor="#888888", facecolor="#f7f7f7")
ax.add_patch(outer)

# Section titles
ax.text(1.35, 8.45, "Input 3D Ultrasound Volume", ha="center", va="bottom",
        fontsize=18, fontweight="bold", color="#222222")
ax.text(4.35, 8.45, "Physics-Guided Imaging Model", ha="center", va="bottom",
        fontsize=18, fontweight="bold", color="#222222")
ax.text(8.0, 8.45, "Shared 3D Feature Encoder", ha="center", va="bottom",
        fontsize=18, fontweight="bold", color="#222222")
ax.text(13.7, 8.45, "Failure-Aware Interpretation", ha="center", va="bottom",
        fontsize=18, fontweight="bold", color="#222222")

# Left: 3D ultrasound cube
cube_x, cube_y = 0.55, 4.3
front = np.array([[cube_x, cube_y], [cube_x+1.0, cube_y-0.45], [cube_x+1.0, cube_y+1.65], [cube_x, cube_y+2.1]])
side = np.array([[cube_x+1.0, cube_y-0.45], [cube_x+1.55, cube_y-0.1], [cube_x+1.55, cube_y+2.0], [cube_x+1.0, cube_y+1.65]])
top = np.array([[cube_x, cube_y+2.1], [cube_x+1.0, cube_y+1.65], [cube_x+1.55, cube_y+2.0], [cube_x+0.55, cube_y+2.45]])
for poly, fc in [(front, "#222222"), (side, "#333333"), (top, "#4a4a4a")]:
    ax.add_patch(Polygon(poly, closed=True, facecolor=fc, edgecolor="#111111", linewidth=1.2))
# ultrasound slice lines
ax.plot([cube_x+0.2, cube_x+1.25], [cube_y+0.25, cube_y+0.7], color="white", alpha=0.35, lw=1)
ax.plot([cube_x+0.15, cube_x+1.2], [cube_y+0.95, cube_y+1.4], color="white", alpha=0.35, lw=1)
ax.plot([cube_x+0.1, cube_x+1.15], [cube_y+1.6, cube_y+2.05], color="white", alpha=0.35, lw=1)
# lesion blob inside cube
theta = np.linspace(0, 2*np.pi, 200)
cx, cy, rx, ry = cube_x+0.9, cube_y+0.95, 0.38, 0.6
blobx = cx + rx*np.cos(theta) * (1 + 0.08*np.sin(3*theta))
bloby = cy + ry*np.sin(theta) * (1 + 0.06*np.cos(4*theta))
ax.fill(blobx, bloby, color="#ff6b2d", alpha=0.85)
ax.text(1.3, 3.95, "Volumetric lesion representation", ha="center", fontsize=11, color="#555555")

# Partial volume box
pv = FancyBboxPatch((0.35, 0.85), 1.8, 2.2, boxstyle="round,pad=0.02,rounding_size=0.08",
                    linewidth=1.2, edgecolor="#cccccc", facecolor="#eef3f5")
ax.add_patch(pv)
ax.text(1.25, 2.65, "Partial-Volume Effects", ha="center", va="center",
        fontsize=13, fontweight="bold", color="#444444")
# mini panel
ax.add_patch(Rectangle((0.5, 1.45), 0.85, 0.75, facecolor="#3e3e3e", edgecolor="white", linewidth=0.8))
ax.add_patch(Rectangle((0.5, 1.45), 0.42, 0.38, facecolor="#111111", edgecolor="none"))
ax.add_patch(Rectangle((0.92, 1.45), 0.43, 0.38, facecolor="#aaaaaa", edgecolor="none"))
ax.add_patch(Rectangle((0.5, 1.83), 0.42, 0.37, facecolor="#d9d9d9", edgecolor="none"))
ax.add_patch(Rectangle((0.92, 1.83), 0.43, 0.37, facecolor="#ff8a3d", edgecolor="none"))
ax.plot([0.5, 1.35], [1.45, 2.2], color="white", lw=1.2)
ax.text(1.25, 1.18, "Mixed voxels", ha="center", fontsize=10, color="#444444")
ax.add_patch(Rectangle((0.45,0.95),0.18,0.12,facecolor="#f4c542", edgecolor="none"))
ax.text(0.66,1.01,"Tumor", fontsize=9, va="center")
ax.add_patch(Rectangle((1.27,0.95),0.18,0.12,facecolor="#ff6b2d", edgecolor="none"))
ax.text(1.49,1.01,"Tumor-boundary", fontsize=9, va="center", ha="left", color="#444")
ax.add_patch(Rectangle((0.45,0.74),0.18,0.12,facecolor="#2e2e2e", edgecolor="none"))
ax.text(0.66,0.80,"Normal tissue", fontsize=9, va="center")

# Physics guided box
phys = FancyBboxPatch((2.35, 1.6), 3.25, 5.9, boxstyle="round,pad=0.03,rounding_size=0.08",
                      linewidth=1.4, edgecolor="#8e8e8e", facecolor="#f2f2f2")
ax.add_patch(phys)

# script F box
fbox = FancyBboxPatch((2.55, 5.95), 0.95, 0.75, boxstyle="round,pad=0.02,rounding_size=0.08",
                      linewidth=1.0, edgecolor="#d0d0d0", facecolor="#ffffff")
ax.add_patch(fbox)
ax.text(3.02, 6.33, "F", ha="center", va="center", fontsize=28, fontstyle="italic", color="#333333")

ax.text(4.05, 6.55, "Blur &\nSampling", ha="center", va="center", fontsize=13, color="#333333")
ax.text(5.05, 6.55, "Imaging\nNoise", ha="center", va="center", fontsize=13, color="#333333")

# blurry/noisy mini images
for x in [4.0, 4.95]:
    rect = FancyBboxPatch((x-0.45,5.55),0.9,0.9,boxstyle="round,pad=0.01,rounding_size=0.04",
                          linewidth=1.0, edgecolor="#5072b8", facecolor="#fafafa")
    ax.add_patch(rect)
    # make pixelated squares
    rng = np.random.default_rng(int(x*100))
    arr = rng.uniform(0.2, 0.9, (6,6))
    ax.imshow(arr, extent=(x-0.37,x+0.37,5.63,6.37), cmap="gray", interpolation="nearest", aspect="auto")

# dashed guides
ax.plot([2.5,5.35],[5.05,5.05], ls="--", color="#6e96cf", lw=1)
ax.plot([2.5,5.35],[3.95,3.95], ls="--", color="#6e96cf", lw=1)
ax.plot([3.65,3.65],[3.95,5.05], ls="--", color="#b3b3b3", lw=1)
ax.plot([4.65,4.65],[3.95,5.05], ls="--", color="#b3b3b3", lw=1)

# encoder blocks
base_x = 2.85
depths = [2.7, 2.45, 2.15, 1.85, 1.55, 1.25, 1.0]
for i,d in enumerate(depths):
    x = base_x + i*0.34
    y = 2.0 + i*0.12
    ax.add_patch(Polygon([[x,y],[x+0.28,y+0.18],[x+0.28,y+d],[x,y+d-0.18]],
                         closed=True, facecolor="#4f9cf5", edgecolor="#2b6cb0", linewidth=1.0, alpha=0.9))
    ax.add_patch(Polygon([[x+0.28,y+0.18],[x+0.58,y],[x+0.58,y+d-0.18],[x+0.28,y+d]],
                         closed=True, facecolor="#91c2fb", edgecolor="#2b6cb0", linewidth=1.0, alpha=0.9))
ax.text(4.15, 1.25, "Encoder extracts volumetric multi-scale features", ha="center",
        fontsize=11, color="#555555")

# arrows from input to physics and encoder
def arrow(x1,y1,x2,y2,color="#6b6b6b", lw=2.2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle="-|>", mutation_scale=18, lw=lw, color=color))
arrow(2.1,5.65,2.55,6.3)
arrow(2.05,2.0,2.65,2.0, color="#4f9cf5", lw=3)

# latent z
arrow(5.7,3.45,6.55,3.45, color="#666666", lw=3)
ax.text(6.05, 3.1, "Latent Feature z", fontsize=15, fontstyle="italic", fontweight="bold", color="#333333")

# three heads
head_x = [6.8, 8.6, 10.5]
head_colors = ["#f6ddd6", "#eee2da", "#d9eaf6"]
head_titles = [("Tumor\nProbability Head", "#d62728"),
               ("Uncertainty\nEstimation Head", "#222222"),
               ("Identifiability\nEstimation Head", "#2b6cb0")]
for x,fc,(txt,tc) in zip(head_x, head_colors, head_titles):
    box = FancyBboxPatch((x,1.15),1.5,5.8,boxstyle="round,pad=0.03,rounding_size=0.08",
                         linewidth=1.0, edgecolor="#d8d8d8", facecolor=fc)
    ax.add_patch(box)
    ax.text(x+0.75, 6.45, txt, ha="center", va="center", fontsize=14, fontweight="bold", color=tc)

# head images
# tumor prob head
yy,xx=np.mgrid[:100,:100]
r=((xx-50)**2/(2*19**2)+(yy-50)**2/(2*24**2))
prob=np.exp(-r)*(0.7+0.3*np.sin(xx/10))
ax.imshow(prob, extent=(7.08,8.22,4.25,5.45), cmap="hot", aspect="auto")
# uncertainty head top + bottom
unc1=np.exp(-(((xx-45)**2)/(2*22**2)+((yy-50)**2)/(2*18**2)))*0.8 + 0.15*np.random.default_rng(0).random((100,100))
unc2=np.exp(-(((xx-50)**2)/(2*26**2)+((yy-55)**2)/(2*22**2)))*0.55 + 0.25*np.random.default_rng(1).random((100,100))
ax.imshow(unc1, extent=(8.88,10.02,4.7,5.42), cmap="gray", aspect="auto")
ax.imshow(unc2, extent=(8.88,10.02,2.9,3.62), cmap="gray", aspect="auto")
ax.text(9.45, 2.55, "Uncertainty Map", ha="center", fontsize=13, color="white",
        bbox=dict(facecolor="#555555", alpha=0.6, pad=2, edgecolor="none"))
# ident head top + bottom
id1=np.clip(0.35+0.65*np.exp(-r)*np.random.default_rng(2).uniform(0.7,1.0,(100,100)),0,1)
id2=np.clip(0.2+0.55*np.exp(-(((xx-50)**2)/(2*28**2)+((yy-50)**2)/(2*24**2))),0,1)
ax.imshow(id1, extent=(10.78,11.92,4.7,5.9), cmap="turbo", aspect="auto")
ax.imshow(id2, extent=(10.78,11.92,2.2,3.4), cmap="viridis", aspect="auto")
ax.text(11.35, 2.0, "Identifiability Map", ha="center", fontsize=13, color="white",
        bbox=dict(facecolor="#4a6cb3", alpha=0.6, pad=2, edgecolor="none"))

# arrows between heads
arrow(8.3,4.85,8.6,4.85)
arrow(10.1,4.85,10.5,4.85)
arrow(9.45,4.6,9.45,3.7)
arrow(10.1,3.25,10.5,3.25)

# failure-aware panel
fa = FancyBboxPatch((12.55,1.95),2.55,4.95,boxstyle="round,pad=0.03,rounding_size=0.08",
                    linewidth=1.4, edgecolor="#8a8a8a", facecolor="#fbfbfb")
ax.add_patch(fa)
# three status boxes
status = [
    (5.45, "#65a844", "Reliable Region", "High Identifiability, Low Uncertainty", "#2f6f1e"),
    (4.15, "#f1cf6a", "Ambiguous Region", "Low Identifiability, High Uncertainty", "#8a6b00"),
    (2.85, "#d94a3e", "Non-Identifiable Region", "Low Identifiability, High Uncertainty", "white"),
]
for y,fc,title,sub,tc in status:
    box=FancyBboxPatch((12.75,y),2.15,0.9,boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=0, facecolor=fc)
    ax.add_patch(box)
    ax.text(13.82,y+0.58,title,ha="center",va="center",fontsize=13,fontweight="bold",color=tc)
    ax.text(13.82,y+0.25,sub,ha="center",va="center",fontsize=9.8,color=tc, style="italic")
arrow(11.95,5.15,12.7,5.75, color="#79a36d")
arrow(11.95,3.25,12.7,4.45, color="#6a8a8a")
arrow(11.95,3.2,12.7,3.15, color="#8d4e9b")

plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03)
save_path="/mnt/data/Figure1_redrawn_3D_ultrasound_900dpi.png"
plt.savefig(save_path, dpi=900, bbox_inches="tight", pad_inches=0.02)
print(save_path)
