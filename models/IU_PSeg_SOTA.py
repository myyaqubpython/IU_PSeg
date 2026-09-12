import os, cv2, csv, random, argparse, copy
from contextlib import nullcontext
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import segmentation_models_pytorch as smp
except Exception:
    smp = None


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Utilities
# ============================================================

def minmax_np(x, eps=1e-7):
    return (x - x.min()) / (x.max() - x.min() + eps)


def minmax_torch(x, eps=1e-7):
    b = x.shape[0]
    flat = x.view(b, -1)
    mn = flat.min(dim=1)[0].view(b, 1, 1, 1)
    mx = flat.max(dim=1)[0].view(b, 1, 1, 1)
    return (x - mn) / (mx - mn + eps)


def gradient_2d(x):
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    dy = F.pad(dy, (0, 0, 0, 1))
    dx = F.pad(dx, (0, 1, 0, 0))
    return torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)


def boundary_from_mask(mask):
    dil = F.max_pool2d(mask, 3, 1, 1)
    ero = -F.max_pool2d(-mask, 3, 1, 1)
    return torch.clamp(dil - ero, 0, 1)


def postprocess_binary(mask, min_area=80, keep_components=2):
    mask = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask

    comps = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            comps.append((area, i))

    if not comps:
        return mask

    comps = sorted(comps, reverse=True)[:keep_components]
    out = np.zeros_like(mask)
    for _, idx in comps:
        out[labels == idx] = 1
    return out


def metrics_np(prob, gt, threshold=0.5, min_area=80, keep_components=2, eps=1e-7):
    pred = (prob >= threshold).astype(np.uint8)
    pred = postprocess_binary(pred, min_area=min_area, keep_components=keep_components)
    gt = (gt > 0.5).astype(np.uint8)

    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    try:
        auc = roc_auc_score(gt.reshape(-1), prob.reshape(-1))
    except Exception:
        auc = np.nan

    return dice, iou, precision, recall, accuracy, auc


# ============================================================
# Dataset
# ============================================================

class MMOTUDataset(Dataset):
    def __init__(self, root, split="train", img_size=384, augment=True):
        self.root = root
        self.image_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "annotations")
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == "train"
        self.pairs = self.collect_pairs()

        if len(self.pairs) == 0:
            raise RuntimeError(f"No samples found for split: {split}")

        print(f"{split}: {len(self.pairs)} samples")

    def collect_pairs(self):
        split_file = os.path.join(self.root, f"{self.split}.txt")
        names = []

        if os.path.exists(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip().replace(",", " ")
                    if line:
                        token = line.split()[0]
                        names.append(os.path.splitext(os.path.basename(token))[0])
        else:
            for f in os.listdir(self.image_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    names.append(os.path.splitext(f)[0])

        pairs = []
        for name in names:
            img_path, mask_path = None, None

            for ext in [".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png"]:
                p = os.path.join(self.image_dir, name + ext)
                if os.path.exists(p):
                    img_path = p
                    break

            for ext in [".PNG", ".png", ".JPG", ".jpg"]:
                p = os.path.join(self.mask_dir, name + ext)
                if os.path.exists(p):
                    mask_path = p
                    break

            if img_path is not None and mask_path is not None:
                pairs.append((img_path, mask_path))

        return pairs

    def load_image_mask(self, img_path, mask_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_rgb = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)

        mask = (
            (mask_rgb[:, :, 0] > 20) |
            (mask_rgb[:, :, 1] > 20) |
            (mask_rgb[:, :, 2] > 20)
        ).astype(np.float32)

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, mask = self.augment_data(img, mask)

        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1)).copy()
        mask = mask[None, :, :].astype(np.float32)

        return img, mask

    def augment_data(self, img, mask):
        h, w = mask.shape

        if random.random() < 0.5:
            img = img[:, ::-1, :]
            mask = mask[:, ::-1]

        if random.random() < 0.5:
            img = img[::-1, :, :]
            mask = mask[::-1, :]

        if random.random() < 0.75:
            angle = random.uniform(-25, 25)
            scale = random.uniform(0.88, 1.15)
            tx = random.uniform(-0.06, 0.06) * w
            ty = random.uniform(-0.06, 0.06) * h
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        if random.random() < 0.55:
            alpha = random.uniform(0.80, 1.25)
            beta = random.uniform(-15, 15)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        if random.random() < 0.30:
            noise = np.random.normal(0, random.uniform(4, 14), img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if random.random() < 0.25:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        if random.random() < 0.25:
            for _ in range(random.randint(1, 5)):
                ch = random.randint(15, 45)
                cw = random.randint(15, 45)
                y0 = random.randint(0, max(0, h - ch))
                x0 = random.randint(0, max(0, w - cw))
                img[y0:y0 + ch, x0:x0 + cw] = 0

        return img.copy(), mask.copy()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img, mask = self.load_image_mask(img_path, mask_path)
        return torch.from_numpy(img).float(), torch.from_numpy(mask).float(), os.path.basename(img_path)


# ============================================================
# Model
# ============================================================

def build_model(model_name="unetplusplus", encoder="resnet50"):
    if smp is None:
        raise ImportError("Please install segmentation-models-pytorch: pip install segmentation-models-pytorch timm")

    model_name = model_name.lower()

    def create(weights):
        if model_name == "unetplusplus":
            return smp.UnetPlusPlus(
                encoder_name=encoder,
                encoder_weights=weights,
                in_channels=3,
                classes=1,
                activation=None,
                decoder_attention_type="scse"
            )
        if model_name == "deeplabv3plus":
            return smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=weights,
                in_channels=3,
                classes=1,
                activation=None
            )
        if model_name == "fpn":
            return smp.FPN(
                encoder_name=encoder,
                encoder_weights=weights,
                in_channels=3,
                classes=1,
                activation=None
            )
        raise ValueError("model must be unetplusplus, deeplabv3plus, or fpn")

    try:
        model = create("imagenet")
        print(f"Using ImageNet encoder: {encoder}")
    except Exception as e:
        print("ImageNet weights failed, using random weights.")
        print("Reason:", e)
        model = create(None)

    return model


# ============================================================
# Losses
# ============================================================

def dice_loss_logits(logits, target, eps=1e-7):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(2, 3))
    den = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


def tversky_loss_logits(logits, target, alpha=0.25, beta=0.75, eps=1e-7):
    prob = torch.sigmoid(logits)
    tp = (prob * target).sum(dim=(2, 3))
    fp = (prob * (1 - target)).sum(dim=(2, 3))
    fn = ((1 - prob) * target).sum(dim=(2, 3))
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - score.mean()


def focal_loss_logits(logits, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = target * prob + (1 - target) * (1 - prob)
    return (alpha * (1 - pt) ** gamma * bce).mean()


def boundary_loss_logits(logits, target):
    prob = torch.sigmoid(logits)
    pred_b = minmax_torch(gradient_2d(prob))
    gt_b = boundary_from_mask(target)

    ctx = torch.amp.autocast(device_type="cuda", enabled=False) if pred_b.is_cuda else nullcontext()
    with ctx:
        pred_b = pred_b.float().clamp(1e-6, 1.0 - 1e-6)
        gt_b = gt_b.float()
        loss = F.binary_cross_entropy(pred_b, gt_b)

    return loss


def combo_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss_logits(logits, target)
    tv = tversky_loss_logits(logits, target)
    focal = focal_loss_logits(logits, target)
    bnd = boundary_loss_logits(logits, target)

    return 1.00 * dice + 0.60 * bce + 0.60 * tv + 0.30 * focal + 0.15 * bnd


# ============================================================
# EMA
# ============================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model):
        state = model.state_dict()
        for k in self.shadow:
            if self.shadow[k].dtype.is_floating_point:
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * state[k].detach()
            else:
                self.shadow[k] = state[k]

    def apply_to(self, model):
        self.backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model):
        model.load_state_dict(self.backup, strict=True)


# ============================================================
# Prediction / evaluation
# ============================================================

@torch.no_grad()
def predict_tta(model, x):
    model.eval()
    probs = []

    p = torch.sigmoid(model(x))
    probs.append(p)

    xf = torch.flip(x, dims=[3])
    p = torch.sigmoid(model(xf))
    probs.append(torch.flip(p, dims=[3]))

    xf = torch.flip(x, dims=[2])
    p = torch.sigmoid(model(xf))
    probs.append(torch.flip(p, dims=[2]))

    xf = torch.flip(x, dims=[2, 3])
    p = torch.sigmoid(model(xf))
    probs.append(torch.flip(p, dims=[2, 3]))

    return torch.stack(probs, dim=0).mean(dim=0)


@torch.no_grad()
def collect_probs(model, loader, device, use_tta=False):
    model.eval()
    probs, masks, names = [], [], []

    for x, y, n in loader:
        x = x.to(device, non_blocking=True)
        p = predict_tta(model, x) if use_tta else torch.sigmoid(model(x))
        probs.append(p.detach().cpu().numpy())
        masks.append(y.numpy())
        names.extend(list(n))

    return np.concatenate(probs, axis=0), np.concatenate(masks, axis=0), names


def find_best_threshold(probs, masks, min_area=80, keep_components=2):
    best_th, best_dice = 0.5, 0.0
    for th in np.arange(0.10, 0.91, 0.01):
        dices = []
        for i in range(probs.shape[0]):
            d, _, _, _, _, _ = metrics_np(
                probs[i, 0], masks[i, 0],
                threshold=float(th),
                min_area=min_area,
                keep_components=keep_components
            )
            dices.append(d)
        md = float(np.mean(dices))
        if md > best_dice:
            best_dice = md
            best_th = float(th)
    return best_th, best_dice


def evaluate(model, loader, device, threshold, use_tta=True, min_area=80, keep_components=2):
    probs, masks, names = collect_probs(model, loader, device, use_tta=use_tta)
    vals = []
    for i in range(probs.shape[0]):
        vals.append(metrics_np(
            probs[i, 0], masks[i, 0],
            threshold=threshold,
            min_area=min_area,
            keep_components=keep_components
        ))
    vals = np.array(vals)
    return {
        "Dice": float(np.nanmean(vals[:, 0])),
        "IoU": float(np.nanmean(vals[:, 1])),
        "Precision": float(np.nanmean(vals[:, 2])),
        "Recall": float(np.nanmean(vals[:, 3])),
        "Accuracy": float(np.nanmean(vals[:, 4])),
        "AUC": float(np.nanmean(vals[:, 5])),
    }


# ============================================================
# IU-PSeg methodology outputs
# ============================================================

def acoustic_perturb(x):
    out = x.clone()
    b, c, h, w = out.shape

    speckle = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.02, 0.08)
    out = out * (1 + speckle * torch.randn_like(out))

    depth = torch.linspace(0, 1, h, device=x.device).view(1, 1, h, 1)
    alpha = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.03, 0.22)
    out = out * torch.exp(-alpha * depth)

    if random.random() < 0.4:
        out = F.avg_pool2d(out, 3, 1, 1)

    noise = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.002, 0.02)
    out = out + noise * torch.randn_like(out)
    return out


@torch.no_grad()
def methodology_inference(model, x, mc_samples=12):
    model.eval()

    prob = predict_tta(model, x)

    tta_probs = []
    p = torch.sigmoid(model(x)); tta_probs.append(p)
    xf = torch.flip(x, dims=[3]); p = torch.sigmoid(model(xf)); tta_probs.append(torch.flip(p, dims=[3]))
    xf = torch.flip(x, dims=[2]); p = torch.sigmoid(model(xf)); tta_probs.append(torch.flip(p, dims=[2]))
    xf = torch.flip(x, dims=[2, 3]); p = torch.sigmoid(model(xf)); tta_probs.append(torch.flip(p, dims=[2, 3]))

    tta_stack = torch.stack(tta_probs, dim=0)
    epistemic = tta_stack.var(dim=0)

    pert_probs = []
    for _ in range(mc_samples):
        xp = acoustic_perturb(x)
        pert_probs.append(torch.sigmoid(model(xp)))

    pert_stack = torch.stack(pert_probs, dim=0)
    aleatoric = pert_stack.var(dim=0)
    total_unc = minmax_torch(aleatoric + epistemic)

    gray = x.mean(dim=1, keepdim=True)
    acoustic_evidence = minmax_torch(gradient_2d(gray))
    boundary = minmax_torch(gradient_2d(prob))

    confidence = torch.abs(prob - 0.5) * 2.0
    perturb_disagreement = minmax_torch(torch.abs(prob - pert_stack.mean(dim=0)))

    ident = confidence * torch.exp(-3.0 * total_unc) * torch.exp(-2.0 * perturb_disagreement)
    ident = minmax_torch(ident * (0.5 + 0.5 * acoustic_evidence))

    reliability = ident * (1 - total_unc)
    return prob, aleatoric, epistemic, total_unc, ident, boundary, reliability


def denormalize_img(x):
    img = x.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def save_gray(path, arr):
    cv2.imwrite(path, (minmax_np(arr) * 255).astype(np.uint8))


def save_heatmap(path, arr):
    h = (minmax_np(arr) * 255).astype(np.uint8)
    h = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    cv2.imwrite(path, h)


def save_methodology_outputs(model, loader, device, output_dir, threshold, min_area, keep_components, mc_samples=12):
    os.makedirs(output_dir, exist_ok=True)
    maps_dir = os.path.join(output_dir, "case_maps")
    os.makedirs(maps_dir, exist_ok=True)

    rows = []
    threshold_values = np.arange(0.10, 0.91, 0.05)
    threshold_dice = {float(t): [] for t in threshold_values}

    for x, y, names in tqdm(loader, desc="Saving IU-PSeg methodology outputs"):
        x = x.to(device)
        y_np = y[0, 0].numpy()
        name = os.path.splitext(names[0])[0]

        prob, ale, epi, total_unc, ident, boundary, reliability = methodology_inference(model, x, mc_samples)

        prob_np = prob[0, 0].cpu().numpy()
        ale_np = ale[0, 0].cpu().numpy()
        epi_np = epi[0, 0].cpu().numpy()
        unc_np = total_unc[0, 0].cpu().numpy()
        id_np = ident[0, 0].cpu().numpy()
        bnd_np = boundary[0, 0].cpu().numpy()
        rel_np = reliability[0, 0].cpu().numpy()

        pred = (prob_np >= threshold).astype(np.uint8)
        pred = postprocess_binary(pred, min_area=min_area, keep_components=keep_components)

        dice, iou, precision, recall, acc, auc = metrics_np(
            prob_np, y_np,
            threshold=threshold,
            min_area=min_area,
            keep_components=keep_components
        )

        denom = prob_np.sum() + 1e-7
        r_case = (prob_np * id_np * (1 - unc_np)).sum() / denom
        u_case = (prob_np * unc_np).sum() / denom
        i_case = (prob_np * id_np).sum() / denom
        flag = 1 if r_case < 0.60 else 0

        rows.append([
            name, dice, iou, precision, recall, acc, auc,
            ale_np.mean(), epi_np.mean(), unc_np.mean(),
            id_np.mean(), rel_np.mean(), r_case, u_case, i_case, flag
        ])

        for t in threshold_values:
            d, _, _, _, _, _ = metrics_np(prob_np, y_np, float(t), min_area, keep_components)
            threshold_dice[float(t)].append(d)

        case_dir = os.path.join(maps_dir, name)
        os.makedirs(case_dir, exist_ok=True)

        img_rgb = denormalize_img(x[0])
        cv2.imwrite(os.path.join(case_dir, "input.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        save_gray(os.path.join(case_dir, "gt_mask.png"), y_np)
        save_gray(os.path.join(case_dir, "refined_segmentation_mask.png"), pred)
        save_heatmap(os.path.join(case_dir, "refined_probability_map.png"), prob_np)
        save_heatmap(os.path.join(case_dir, "aleatoric_uncertainty_map.png"), ale_np)
        save_heatmap(os.path.join(case_dir, "epistemic_uncertainty_map.png"), epi_np)
        save_heatmap(os.path.join(case_dir, "total_uncertainty_map.png"), unc_np)
        save_heatmap(os.path.join(case_dir, "identifiability_map.png"), id_np)
        save_heatmap(os.path.join(case_dir, "boundary_response_map.png"), bnd_np)
        save_heatmap(os.path.join(case_dir, "voxel_reliability_map.png"), rel_np)

    with open(os.path.join(output_dir, "per_case_IU_PSeg_methodology_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Case", "Dice", "IoU", "Precision", "Recall", "Accuracy", "AUC",
            "Aleatoric", "Epistemic", "Total_Uncertainty",
            "Identifiability", "Voxel_Reliability",
            "R_case", "U_case", "I_case", "Expert_Review_Flag"
        ])
        writer.writerows(rows)

    arr = np.array([r[1:] for r in rows], dtype=float)
    summary = [
        ("Dice", np.nanmean(arr[:, 0])),
        ("IoU", np.nanmean(arr[:, 1])),
        ("Precision", np.nanmean(arr[:, 2])),
        ("Recall", np.nanmean(arr[:, 3])),
        ("Accuracy", np.nanmean(arr[:, 4])),
        ("AUC", np.nanmean(arr[:, 5])),
        ("Aleatoric_Uncertainty", np.nanmean(arr[:, 6])),
        ("Epistemic_Uncertainty", np.nanmean(arr[:, 7])),
        ("Total_Uncertainty", np.nanmean(arr[:, 8])),
        ("Identifiability", np.nanmean(arr[:, 9])),
        ("Voxel_Reliability", np.nanmean(arr[:, 10])),
        ("Case_Reliability_Rcase", np.nanmean(arr[:, 11])),
        ("Mean_Ucase", np.nanmean(arr[:, 12])),
        ("Mean_Icase", np.nanmean(arr[:, 13])),
        ("Expert_Review_Flag_Rate", np.nanmean(arr[:, 14])),
        ("Best_Threshold", threshold),
    ]

    with open(os.path.join(output_dir, "summary_IU_PSeg_methodology_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerows(summary)

    with open(os.path.join(output_dir, "threshold_sensitivity_analysis.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Threshold", "Mean_Dice"])
        for t in threshold_values:
            writer.writerow([float(t), np.mean(threshold_dice[float(t)])])

    print("\nIU-PSeg methodology outputs saved in:", output_dir)
    for k, v in summary:
        print(f"{k}: {v:.4f}")


# ============================================================
# Training
# ============================================================

def train(args):
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = MMOTUDataset(args.data_root, "train", args.img_size, augment=True)
    val_ds = MMOTUDataset(args.data_root, "val", args.img_size, augment=False)
    test_ds = MMOTUDataset(args.data_root, "test", args.img_size, augment=False)

    pin = device.type == "cuda"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=pin)

    model = build_model(args.model, args.encoder).to(device)

    start_epoch = 1
    best_val = 0.0
    best_th = 0.5

    if args.resume and os.path.exists(args.resume):
        print("Loading checkpoint:", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val_dice", 0.0)
        best_th = ckpt.get("best_threshold", 0.5)
        print(f"Resumed from epoch {start_epoch}")
        print(f"Previous best Dice: {best_val:.4f}, threshold: {best_th:.2f}")

    ema = EMA(model, decay=args.ema_decay)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=40,
        T_mult=2,
        eta_min=args.lr * 0.01
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "best_IU_PSeg_SOTA.pth")

    patience = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(x)
                loss = combo_loss(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{np.mean(losses):.4f}"})

        scheduler.step(epoch)

        ema.apply_to(model)
        val_probs, val_masks, _ = collect_probs(model, val_loader, device, use_tta=False)
        th, val_dice = find_best_threshold(
            val_probs, val_masks,
            min_area=args.min_area,
            keep_components=args.keep_components
        )
        ema.restore(model)

        print(f"Epoch {epoch}: Loss={np.mean(losses):.4f}, Val Dice={val_dice:.4f}, Threshold={th:.2f}")

        if val_dice > best_val:
            best_val = val_dice
            best_th = th
            patience = 0

            ema.apply_to(model)
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_val_dice": best_val,
                "best_threshold": best_th,
                "args": vars(args)
            }, best_path)
            ema.restore(model)

            print("Saved best:", best_path)
        else:
            patience += 1

        if patience >= args.early_stop:
            print("Early stopping.")
            break

    print("\nLoading best model...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    best_th = ckpt["best_threshold"]

    print(f"Best Validation Dice: {ckpt['best_val_dice']:.4f}")
    print(f"Best Threshold: {best_th:.2f}")

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        threshold=best_th,
        use_tta=True,
        min_area=args.min_area,
        keep_components=args.keep_components
    )

    print("\nFinal Test Results with TTA + post-processing")
    print("---------------------------------------------")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    with open(os.path.join(args.output_dir, "SOTA_test_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in test_metrics.items():
            writer.writerow([k, v])
        writer.writerow(["Best_Threshold", best_th])
        writer.writerow(["Best_Val_Dice", ckpt["best_val_dice"]])

    methodology_dir = os.path.join(args.output_dir, "IU_PSeg_methodology_outputs")
    save_methodology_outputs(
        model,
        test_loader,
        device,
        methodology_dir,
        threshold=best_th,
        min_area=args.min_area,
        keep_components=args.keep_components,
        mc_samples=args.mc_samples
    )


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str,
                        default=r"C:\Users\M YAQUB\CSU2026\DATA\MMOTU\OTU_3d")
    parser.add_argument("--output_dir", type=str,
                        default=r"C:\Users\M YAQUB\CSU2026\IU-PSeg\SOTA_outputs")

    parser.add_argument("--model", type=str, default="unetplusplus",
                        choices=["unetplusplus", "deeplabv3plus", "fpn"])
    parser.add_argument("--encoder", type=str, default="resnet50")

    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--min_area", type=int, default=80)
    parser.add_argument("--keep_components", type=int, default=2)

    parser.add_argument("--mc_samples", type=int, default=12)
    parser.add_argument("--early_stop", type=int, default=80)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)