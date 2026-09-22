"""
test.py

Loads trained Decom-Net and Enhance-Net checkpoints, runs RetinexNet on a
folder of low-light test images, and saves enhanced outputs to
results/retinexnet/ under the ORIGINAL filenames.

Usage:
    python test.py --test_low_dir dataset/test/low \
        --decom_checkpoint checkpoints/decom_best.pth \
        --enhance_checkpoint checkpoints/enhance_best.pth \
        --output_dir results/retinexnet

Add --save_components to additionally save reflectance/illumination/
enhanced-illumination maps per image (off by default — only the final
enhanced image is saved by default, as requested).
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config, add_common_args, build_config_from_args
from datasets.lol_dataset import LOLDataset
from models.retinexnet import RetinexNet
from utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Run RetinexNet inference on test images")
    parser = add_common_args(parser)
    parser.add_argument("--decom_checkpoint", type=str, required=True)
    parser.add_argument("--enhance_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/retinexnet")
    parser.add_argument("--save_components", action="store_true",
                         help="Also save reflectance / illumination / enhanced illumination maps")
    args = parser.parse_args()

    cfg = build_config_from_args(args)
    print(f"Using device: {cfg.device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    components_dir = output_dir / "components"
    if args.save_components:
        components_dir.mkdir(parents=True, exist_ok=True)

    # Ground truth is optional for pure inference — LOLDataset supports high_dir=None.
    test_set = LOLDataset(
        low_dir=cfg.test_low_dir, high_dir=None,
        patch_size=None, image_size=cfg.image_size, augment=False,
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    print(f"Found {len(test_set)} test images in {cfg.test_low_dir}")

    model = RetinexNet().to(cfg.device)
    decom_ckpt = load_checkpoint(args.decom_checkpoint, device=cfg.device)
    enhance_ckpt = load_checkpoint(args.enhance_checkpoint, device=cfg.device)
    model.decom_net.load_state_dict(decom_ckpt["model_state"])
    model.enhance_net.load_state_dict(enhance_ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            low = batch["low"].to(cfg.device)
            filename = batch["filename"][0]

            out = model(low)
            enhanced = out["enhanced_image"][0].clamp(0, 1)

            save_image(enhanced, str(output_dir / filename))

            if args.save_components:
                stem = Path(filename).stem
                save_image(out["reflectance"][0].clamp(0, 1), str(components_dir / f"{stem}_reflectance.png"))
                save_image(out["illumination"][0].clamp(0, 1).repeat(3, 1, 1),
                           str(components_dir / f"{stem}_illumination.png"))
                save_image(out["enhanced_illumination"][0].clamp(0, 1).repeat(3, 1, 1),
                           str(components_dir / f"{stem}_enhanced_illumination.png"))

    print(f"\nDone. Enhanced images saved to: {output_dir}")


if __name__ == "__main__":
    main()
