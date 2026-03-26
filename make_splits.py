import os
import random

random.seed(42)

img_dir = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d\images"
ann_dir = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d\annotations"
out_dir = r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d"

files = []

# Collect valid pairs
for f in os.listdir(img_dir):
    name = os.path.splitext(f)[0]
    mask_path = os.path.join(ann_dir, name + ".PNG")

    if os.path.exists(mask_path):
        files.append(name)

random.shuffle(files)

n = len(files)

train_split = int(0.7 * n)
val_split = int(0.85 * n)

train = files[:train_split]
val = files[train_split:val_split]
test = files[val_split:]

# Save files
def save_txt(path, data):
    with open(path, "w") as f:
        for item in data:
            f.write(item + "\n")

save_txt(os.path.join(out_dir, "train.txt"), train)
save_txt(os.path.join(out_dir, "val.txt"), val)
save_txt(os.path.join(out_dir, "test.txt"), test)

print("✅ train.txt, val.txt, test.txt created!")
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")