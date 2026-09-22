"""
evaluate.py

Computes average PSNR / SSIM / LPIPS between RetinexNet's enhanced test
images and the LOL-v2-Real ground-truth normal-light images, and writes a
per-image CSV.

Usage:
    python evaluate.py --enhanced_dir results/retinexnet \
        --gt_dir dataset/test/high \
        --output_csv results/retinexnet_metrics.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from utils.metrics import compute_psnr, compute_ssim, compute_lpips


def load_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RetinexNet enhanced images against ground truth")
    parser.add_argument("--enhanced_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="results/retinexnet_metrics.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    enhanced_dir = Path(args.enhanced_dir)
    gt_dir = Path(args.gt_dir)

    enhanced_files = sorted([p for p in enhanced_dir.iterdir()
                              if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
    if len(enhanced_files) == 0:
        raise FileNotFoundError(f"No enhanced images found in {enhanced_dir}")

    rows = []
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for enh_path in tqdm(enhanced_files, desc="Evaluating"):
        gt_path = gt_dir / enh_path.name
        if not gt_path.exists():
            print(f"WARNING: no matching ground-truth file for {enh_path.name}, skipping.")
            continue

        pred = load_tensor(enh_path)
        target = load_tensor(gt_path)

        if pred.shape != target.shape:
            # Resize prediction to match ground truth resolution if they differ.
            target = torch.nn.functional.interpolate(
                target.unsqueeze(0), size=pred.shape[1:], mode="bilinear", align_corners=False
            ).squeeze(0)

        psnr = compute_psnr(pred, target)
        ssim = compute_ssim(pred, target)
        lpips_val = compute_lpips(pred, target, device=args.device)

        psnr_vals.append(psnr)
        ssim_vals.append(ssim)
        if lpips_val is not None:
            lpips_vals.append(lpips_val)

        rows.append({
            "filename": enh_path.name,
            "psnr": f"{psnr:.4f}",
            "ssim": f"{ssim:.4f}",
            "lpips": f"{lpips_val:.4f}" if lpips_val is not None else "N/A",
        })

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== Average Metrics =====")
    print(f"PSNR:  {np.mean(psnr_vals):.4f} dB")
    print(f"SSIM:  {np.mean(ssim_vals):.4f}")
    if lpips_vals:
        print(f"LPIPS: {np.mean(lpips_vals):.4f}")
    else:
        print("LPIPS: not computed (lpips package not installed)")
    print(f"\nPer-image metrics written to: {args.output_csv}")


if __name__ == "__main__":
    main()
