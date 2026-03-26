import os
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm

from IU_PSeg import build_model, OTU3DDataset, dice_score


def evaluate(model, loader, device):
    model.eval()
    dices = []
    with torch.no_grad():
        for img, mask in tqdm(loader, desc="Evaluating"):
            img, mask = img.to(device), mask.to(device)
            pred = torch.sigmoid(model(img))
            d = dice_score(pred, mask)
            dices.append(d.item())
    return np.array(dices)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load dataset
    dataset = OTU3DDataset(args.data_root, args.split_txt)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False
    )

    # Load retrained model
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    print(f"[OK] Loaded retrained model: {args.ckpt}")

    # Evaluate
    new_dice = evaluate(model, loader, device)

    # Load baseline metrics
    base_df = pd.read_csv(args.step1_metrics)
    base_dice = base_df["dice"].values

    # Safety check
    assert len(base_dice) == len(new_dice)

    # Compute gains
    delta = new_dice - base_dice

    report = pd.DataFrame({
        "case": base_df["stem"],
        "dice_before": base_dice,
        "dice_after": new_dice,
        "delta_dice": delta
    })

    os.makedirs(args.out_root, exist_ok=True)
    report_path = os.path.join(args.out_root, "step8_gain_analysis.csv")
    report.to_csv(report_path, index=False)

    print("✅ STEP-8 COMPLETED SUCCESSFULLY")
    print(f"[OK] Gain report saved: {report_path}")

    print("\n📊 Summary:")
    print(f"Mean Dice (Before): {base_dice.mean():.4f}")
    print(f"Mean Dice (After) : {new_dice.mean():.4f}")
    print(f"Mean Gain         : {delta.mean():+.4f}")
    print(f"Recovered Failures (<0.7 Dice): {( (base_dice < 0.7) & (new_dice >= 0.7) ).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_txt", required=True)
    parser.add_argument("--step1_metrics", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_root", required=True)

    args = parser.parse_args()
    main(args)
