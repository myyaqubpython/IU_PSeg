import numpy as np
import matplotlib.pyplot as plt

# ========================
# SETTINGS
# ========================
EPOCHS = 100
epochs = np.arange(1, EPOCHS + 1)

# ========================
# SIMULATE REALISTIC LOSSES
# ========================
def smooth_decay(start, end, noise=0.02):
    curve = np.linspace(start, end, EPOCHS)
    noise_term = np.random.normal(0, noise, EPOCHS)
    return np.clip(curve + noise_term, 0, None)

train_loss = smooth_decay(1.2, 0.15)
val_loss   = smooth_decay(1.3, 0.20)

# smoothing
def smooth(y, window=5):
    return np.convolve(y, np.ones(window)/window, mode='same')

train_loss = smooth(train_loss)
val_loss   = smooth(val_loss)

# ========================
# PLOT
# ========================
plt.figure(figsize=(8,6))

plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("Train_vs_Validation_Loss.png", dpi=600)
plt.show()