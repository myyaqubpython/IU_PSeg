import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2
from tqdm import tqdm
import pandas as pd
import umap
import scanpy as sc
import matplotlib.pyplot as plt

# -----------------------------
# Model (MATCHES TRAINING)
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=64):
        super().__init__()

        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.dec3 = DoubleConv(base * 8, base * 4)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.dec1 = DoubleConv(base * 2, base)

        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x, return_embedding=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        emb = torch.mean(e4, dim=(2, 3))

        d3 = self.dec3(self.up(e4))
        d2 = self.dec2(self.up(d3))
        d1 = self.dec1(self.up(d2))
        out = self.outc(d1)

        if return_embedding:
            return out, emb
        return out


# -----------------------------
# Dataset (ROBUST)
# -----------------------------
class OTU3DDataset(Dataset):
    def __init__(self, data_root, split_txt, img_size=256):
        self.img_dir = os.path.join(data_root, "images")
        self.img_size = img_size

        with open(split_txt, "r") as f:
            self.ids = [x.strip() for x in f.readlines()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        # Try common extensions
        img = None
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            path = os.path.join(self.img_dir, name + ext)
            if os.path.exists(path):
                img = cv2.imread(path)
                break

        if img is None:
            raise FileNotFoundError(f"Missing image for case: {name}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, name


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_txt", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_root", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = OTU3DDataset(args.data_root, args.split_txt)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = TinyUNet(base=64).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    embeddings, names, skipped = [], [], []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Extract embeddings"):
            try:
                img, name = dataset[i]
                img = img.unsqueeze(0).to(device)
                _, emb = model(img, return_embedding=True)
                embeddings.append(emb.cpu().numpy())
                names.append(name)
            except Exception as e:
                skipped.append(dataset.ids[i])
                print(f"[SKIP] {dataset.ids[i]} → {e}")

    embeddings = np.vstack(embeddings)

    # -----------------------------
    # Save embeddings
    # -----------------------------
    pd.DataFrame(embeddings).assign(case=names).to_csv(
        os.path.join(args.out_root, "step4_embeddings.csv"), index=False
    )

    # -----------------------------
    # UMAP
    # -----------------------------
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_emb = reducer.fit_transform(embeddings)

    umap_df = pd.DataFrame(umap_emb, columns=["UMAP1", "UMAP2"])
    umap_df["case"] = names
    umap_df.to_csv(os.path.join(args.out_root, "step4_umap.csv"), index=False)

    plt.figure(figsize=(6, 5))
    plt.scatter(umap_emb[:, 0], umap_emb[:, 1], s=12)
    plt.title("UMAP of IU-PSeg Embeddings")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_root, "umap.png"), dpi=300)
    plt.close()

    # -----------------------------
    # PAGA
    # -----------------------------
    adata = sc.AnnData(embeddings)
    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.leiden(adata, resolution=0.5)
    sc.tl.paga(adata)

    sc.pl.paga(adata, show=False)
    plt.savefig(os.path.join(args.out_root, "paga.png"), dpi=300)
    plt.close()

    # -----------------------------
    # Log skipped cases
    # -----------------------------
    if skipped:
        with open(os.path.join(args.out_root, "skipped_cases.txt"), "w") as f:
            for s in skipped:
                f.write(s + "\n")

    print("STEP-4 COMPLETED SUCCESSFULLY")
    print(f"Valid samples: {len(names)} | Skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
