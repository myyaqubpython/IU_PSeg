import os
import csv
import cv2
import argparse
import random
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Dataset
# =========================================================
class OTU3DDataset(Dataset):
    def __init__(self, root, split_txt, img_size=256, augment=True):
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "annotations")
        self.img_size = img_size
        self.augment = augment

        with open(split_txt, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def _find_img(self, name):
        for ext in [".jpg", ".png", ".JPG", ".jpeg", ".JPEG"]:
            p = os.path.join(self.img_dir, name + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Image not found for ID: {name}")

    def _find_mask(self, name):
        for ext in [".PNG", ".png", ".jpg", ".JPG"]:
            p = os.path.join(self.mask_dir, name + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Mask not found for ID: {name}")

    def _augment(self, img, mask):
        # horizontal flip
        if np.random.rand() < 0.5:
            img = np.flip(img, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # vertical flip
        if np.random.rand() < 0.5:
            img = np.flip(img, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        # random rotation
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-25, 25)
            center = (self.img_size // 2, self.img_size // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(
                img, M, (self.img_size, self.img_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
            mask = cv2.warpAffine(
                mask, M, (self.img_size, self.img_size),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT_101
            )

        # brightness/contrast
        if np.random.rand() < 0.4:
            alpha = np.random.uniform(0.9, 1.15)  # contrast
            beta = np.random.uniform(-10, 10)     # brightness
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # gaussian blur
        if np.random.rand() < 0.25:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        return img, mask

    def __getitem__(self, idx):
        name = self.ids[idx]

        img = cv2.imread(self._find_img(name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        mask = cv2.imread(self._find_mask(name), 0)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)

        if self.augment:
            img, mask = self._augment(img, mask)

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1).copy()
        mask = np.expand_dims(mask, axis=0).copy()

        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()


# =========================================================
# Model blocks
# =========================================================
class ResidualConv(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        x = x + residual
        x = self.relu(x)
        return x


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(out_ch * 4, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.out(x)


# =========================================================
# Upgraded IU_PSeg model
# =========================================================
class IU_PSegNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        self.enc1 = ResidualConv(in_ch, 64, dropout=0.05)
        self.enc2 = ResidualConv(64, 128, dropout=0.05)
        self.enc3 = ResidualConv(128, 256, dropout=0.10)
        self.enc4 = ResidualConv(256, 512, dropout=0.10)

        self.pool = nn.MaxPool2d(2, 2)

        self.bridge = ASPP(512, 512)

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.att4 = AttentionGate(512, 512, 256)
        self.dec4 = ResidualConv(512 + 512, 256, dropout=0.10)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.att3 = AttentionGate(256, 256, 128)
        self.dec3 = ResidualConv(256 + 256, 128, dropout=0.10)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.att2 = AttentionGate(128, 128, 64)
        self.dec2 = ResidualConv(128 + 128, 64, dropout=0.05)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.att1 = AttentionGate(64, 64, 32)
        self.dec1 = ResidualConv(64 + 64, 64, dropout=0.05)

        self.final = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)              # 256
        e2 = self.enc2(self.pool(e1))  # 128
        e3 = self.enc3(self.pool(e2))  # 64
        e4 = self.enc4(self.pool(e3))  # 32

        b = self.bridge(self.pool(e4)) # 16

        d4 = self.up4(b)               # 32
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)              # 64
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)              # 128
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)              # 256
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        return self.final(d1)


# =========================================================
# Losses
# =========================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        target = target.view(-1)

        intersection = (probs * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + target.sum() + self.smooth
        )
        return 1.0 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class HybridLoss(nn.Module):
    def __init__(self, bce_w=0.3, dice_w=0.5, focal_w=0.2):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.focal_w = focal_w

    def forward(self, logits, target):
        loss = (
            self.bce_w * self.bce(logits, target) +
            self.dice_w * self.dice(logits, target) +
            self.focal_w * self.focal(logits, target)
        )
        return loss


# =========================================================
# Metrics
# =========================================================
def compute_metrics_from_probs(probs, gt, threshold=0.4):
    pred = (probs > threshold).float()

    tp = (pred * gt).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()
    tn = ((1 - pred) * (1 - gt)).sum().item()

    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    acc = (tp + tn + 1e-6) / (tp + tn + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = (2 * precision * recall + 1e-6) / (precision + recall + 1e-6)

    return dice, iou, acc, precision, recall, f1


def hd95_batch(probs, gt, threshold=0.4):
    pred = (probs > threshold).float().cpu().numpy()
    gt = gt.cpu().numpy()

    hd_vals = []
    for b in range(pred.shape[0]):
        p = np.argwhere(pred[b, 0] > 0)
        g = np.argwhere(gt[b, 0] > 0)

        if len(p) == 0 or len(g) == 0:
            hd_vals.append(0.0)
        else:
            hd_vals.append(max(
                directed_hausdorff(p, g)[0],
                directed_hausdorff(g, p)[0]
            ))
    return float(np.mean(hd_vals))


# =========================================================
# Train / Validate
# =========================================================
def run_one_epoch(model, loader, criterion, optimizer, device, train_mode=True, threshold=0.4):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    stats = np.zeros(6, dtype=np.float64)
    hd_list = []

    pbar = tqdm(loader, leave=False)
    for img, mask in pbar:
        img = img.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.float32)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            logits = model(img)
            loss = criterion(logits, mask)

            if train_mode:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            stats += np.array(compute_metrics_from_probs(probs, mask, threshold=threshold))
            hd_list.append(hd95_batch(probs, mask, threshold=threshold))

        pbar.set_description(
            f"{'Train' if train_mode else 'Val'} | Loss {loss.item():.4f}"
        )

    total_loss /= len(loader)
    stats /= len(loader)
    hd_val = float(np.mean(hd_list)) if len(hd_list) > 0 else 0.0

    return total_loss, stats, hd_val


def train(args):
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = OTU3DDataset(
        root=args.data_root,
        split_txt=args.train_split,
        img_size=args.img_size,
        augment=True
    )
    val_ds = OTU3DDataset(
        root=args.data_root,
        split_txt=args.val_split,
        img_size=args.img_size,
        augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = IU_PSegNet().to(device)

    if args.pretrained and os.path.exists(args.pretrained):
        print(f"[INFO] Loading pretrained weights: {args.pretrained}")
        state = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(state, strict=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=12, verbose=True
    )

    criterion = HybridLoss()

    os.makedirs(args.out_root, exist_ok=True)

    csv_path = os.path.join(args.out_root, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_acc",
            "train_precision", "train_recall", "train_f1", "train_hd95",
            "val_loss", "val_dice", "val_iou", "val_acc",
            "val_precision", "val_recall", "val_f1", "val_hd95",
            "lr"
        ])

        best_val_dice = 0.0
        early_stop_counter = 0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_stats, train_hd = run_one_epoch(
                model, train_loader, criterion, optimizer, device,
                train_mode=True, threshold=args.threshold
            )

            val_loss, val_stats, val_hd = run_one_epoch(
                model, val_loader, criterion, optimizer, device,
                train_mode=False, threshold=args.threshold
            )

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_stats[0])  # val Dice

            print(
                f"[Epoch {epoch:03d}] "
                f"Train Loss={train_loss:.4f} Dice={train_stats[0]:.4f} IoU={train_stats[1]:.4f} "
                f"| Val Loss={val_loss:.4f} Dice={val_stats[0]:.4f} IoU={val_stats[1]:.4f} "
                f"Prec={val_stats[3]:.4f} Recall={val_stats[4]:.4f} F1={val_stats[5]:.4f} "
                f"HD95={val_hd:.2f} LR={current_lr:.6f}"
            )

            writer.writerow([
                epoch,
                train_loss, *train_stats.tolist(), train_hd,
                val_loss, *val_stats.tolist(), val_hd,
                current_lr
            ])
            f.flush()

            # save last
            torch.save(model.state_dict(), os.path.join(args.out_root, "last.pth"))

            # save best
            if val_stats[0] > best_val_dice:
                best_val_dice = val_stats[0]
                early_stop_counter = 0
                torch.save(model.state_dict(), os.path.join(args.out_root, "best_dice.pth"))
                print(f"✅ Saved best model at epoch {epoch} | Val Dice={best_val_dice:.4f}")
            else:
                early_stop_counter += 1

            if early_stop_counter >= args.early_stop:
                print(f"🛑 Early stopping at epoch {epoch}")
                break

    print("🔥 Training completed")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--train_split", required=True, type=str)
    parser.add_argument("--val_split", required=True, type=str)
    parser.add_argument("--out_root", required=True, type=str)

    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--threshold", default=0.4, type=float)
    parser.add_argument("--early_stop", default=30, type=int)
    parser.add_argument("--pretrained", default=None, type=str)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()