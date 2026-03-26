import os
import argparse
import pandas as pd
import subprocess


def parse_args():
    parser = argparse.ArgumentParser("STEP-7: Hard-Sample Fine-Tuning")
    parser.add_argument("--hard_csv", required=True,
                        help="Top-K hard samples CSV from STEP-5")
    parser.add_argument("--data_root", required=True,
                        help="Dataset root (MMOTU/OTU_3d)")
    parser.add_argument("--out_root", required=True,
                        help="Output directory")
    parser.add_argument("--base_ckpt", required=True,
                        help="Base trained checkpoint (epoch_100.pth)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    # ---- Load hard samples
    df = pd.read_csv(args.hard_csv)
    assert "stem" in df.columns, "Missing 'stem' column in hard CSV"

    hard_list = os.path.join(args.out_root, "hard_train.txt")
    df["stem"].to_csv(hard_list, index=False, header=False)

    print(f"[OK] Hard-sample list saved: {hard_list}")
    print(f"[INFO] Hard samples: {len(df)}")

    # ---- IMPORTANT FIX ----
    # We pass base checkpoint via ENV variable (not CLI)
    os.environ["IU_PSEG_PRETRAINED"] = args.base_ckpt

    # ---- Run IU_PSeg normally
    cmd = [
        "python", "IU_PSeg.py",
        "--data_root", args.data_root,
        "--split_root", hard_list,
        "--out_root", args.out_root,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr)
    ]

    print("\n[INFO] Starting hard-sample fine-tuning (checkpoint injected internally)")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    print("\nSTEP-7 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
