import numpy as np
import matplotlib.pyplot as plt

# ========================
# SETTINGS
# ========================
EPOCHS = 100
epochs = np.arange(1, EPOCHS + 1)

# ========================
# SIMULATED CURVES
# ========================
def smooth_decay(start, end, noise=0.02):
    curve = np.linspace(start, end, EPOCHS)
    noise_term = np.random.normal(0, noise, EPOCHS)
    return np.clip(curve + noise_term, 0, None)

def smooth(y, window=5):
    return np.convolve(y, np.ones(window)/window, mode='same')

# Loss decreases
loss = smooth(smooth_decay(1.2, 0.15))

# Dice increases (inverse trend)
dice = smooth(np.linspace(0.6, 0.90, EPOCHS) + np.random.normal(0, 0.01, EPOCHS))

# ========================
# PLOT
# ========================
plt.figure(figsize=(8,6))

plt.plot(epochs, loss, label="Total Loss")
plt.plot(epochs, dice, label="Dice Score")

plt.xlabel("Epochs")
plt.ylabel("Value")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("Loss_vs_Dice.png", dpi=600)
plt.show()