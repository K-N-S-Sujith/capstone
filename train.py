"""
train.py

Two-stage RetinexNet training on LOL-v2-Real.

Stage 1 trains Decom-Net on (low, high) pairs using reconstruction +
reflectance-consistency + illumination-smoothness losses.

Stage 2 FREEZES Decom-Net and trains Enhance-Net to map low-light
illumination to high-light illumination.

Usage:
    python train.py --data_root dataset --batch_size 8 --epochs_decom 100 --epochs_enhance 100

All paths and hyperparameters are configurable via CLI flags (see config.py).
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, add_common_args, build_config_from_args
from datasets.lol_dataset import LOLDataset
from models.retinexnet import RetinexNet
from losses.retinex_losses import DecomNetLoss, EnhanceNetLoss
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.visualization import save_validation_panel
from utils.metrics import compute_psnr, compute_ssim, compute_lpips


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: Config):
    train_set = LOLDataset(
        low_dir=cfg.train_low_dir, high_dir=cfg.train_high_dir,
        patch_size=cfg.patch_size, image_size=None, augment=True,
    )
    val_set = LOLDataset(
        low_dir=cfg.val_low_dir, high_dir=cfg.val_high_dir,
        patch_size=None, image_size=cfg.image_size or 384, augment=False,
    )
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


def train_decom_stage(model: RetinexNet, train_loader, val_loader, cfg: Config) -> None:
    print("\n" + "=" * 60)
    print("STAGE 1: Training Decom-Net")
    print("=" * 60)

    model.decom_net.to(cfg.device).train()
    optimizer = torch.optim.Adam(model.decom_net.parameters(), lr=cfg.lr_decom)
    criterion = DecomNetLoss(
        w_recon=cfg.w_recon, w_reflectance=cfg.w_reflectance,
        w_illum_smooth=cfg.w_illum_smooth, w_cross=cfg.w_recon_low_to_high,
    )

    start_epoch = 0
    best_loss = float("inf")
    if cfg.resume_decom:
        ckpt = load_checkpoint(cfg.resume_decom, device=cfg.device)
        model.decom_net.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["best_metric"]
        print(f"Resumed Decom-Net from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs_decom):
        model.decom_net.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Decom] Epoch {epoch + 1}/{cfg.epochs_decom}")

        for batch in pbar:
            i_low = batch["low"].to(cfg.device)
            i_high = batch["high"].to(cfg.device)

            r_low, l_low = model.decom_net(i_low)
            r_high, l_high = model.decom_net(i_high)

            loss, logs = criterion(r_low, l_low, r_high, l_high, i_low, i_high)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += logs["total"]
            pbar.set_postfix(loss=logs["total"], recon=logs["recon"], refl=logs["reflectance"])

        avg_loss = epoch_loss / len(train_loader)
        print(f"[Decom] Epoch {epoch + 1} — avg total loss: {avg_loss:.4f}")

        if (epoch + 1) % cfg.val_every == 0:
            val_loss = validate_decom(model, val_loader, criterion, cfg, epoch)
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    str(Path(cfg.checkpoint_dir) / "decom_best.pth"),
                    model.decom_net.state_dict(), optimizer.state_dict(), epoch, best_loss,
                )
                print(f"  -> New best Decom-Net checkpoint (val loss {val_loss:.4f})")

        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(
                str(Path(cfg.checkpoint_dir) / f"decom_epoch{epoch + 1}.pth"),
                model.decom_net.state_dict(), optimizer.state_dict(), epoch, best_loss,
            )


def validate_decom(model, val_loader, criterion, cfg: Config, epoch: int) -> float:
    model.decom_net.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            i_low = batch["low"].to(cfg.device)
            i_high = batch["high"].to(cfg.device)
            r_low, l_low = model.decom_net(i_low)
            r_high, l_high = model.decom_net(i_high)
            _, logs = criterion(r_low, l_low, r_high, l_high, i_low, i_high)
            total_loss += logs["total"]
    avg = total_loss / len(val_loader)
    print(f"[Decom] Epoch {epoch + 1} — validation loss: {avg:.4f}")
    return avg


def train_enhance_stage(model: RetinexNet, train_loader, val_loader, cfg: Config) -> None:
    print("\n" + "=" * 60)
    print("STAGE 2: Training Enhance-Net (Decom-Net frozen)")
    print("=" * 60)

    model.freeze_decom()
    model.enhance_net.to(cfg.device).train()
    optimizer = torch.optim.Adam(model.enhance_net.parameters(), lr=cfg.lr_enhance)
    criterion = EnhanceNetLoss(w_enhance_smooth=cfg.w_enhance_smooth)

    start_epoch = 0
    best_metric = -float("inf")  # track best validation PSNR (higher is better)
    if cfg.resume_enhance:
        ckpt = load_checkpoint(cfg.resume_enhance, device=cfg.device)
        model.enhance_net.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt["best_metric"]
        print(f"Resumed Enhance-Net from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs_enhance):
        model.enhance_net.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Enhance] Epoch {epoch + 1}/{cfg.epochs_enhance}")

        for batch in pbar:
            i_low = batch["low"].to(cfg.device)
            i_high = batch["high"].to(cfg.device)

            with torch.no_grad():
                r_low, l_low = model.decom_net(i_low)
                _, l_high = model.decom_net(i_high)

            l_low_hat = model.enhance_net(r_low, l_low)
            loss, logs = criterion(l_low_hat, l_high, r_low)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += logs["total"]
            pbar.set_postfix(loss=logs["total"], illum=logs["illum_enhance"])

        avg_loss = epoch_loss / len(train_loader)
        print(f"[Enhance] Epoch {epoch + 1} — avg total loss: {avg_loss:.4f}")

        if (epoch + 1) % cfg.val_every == 0:
            psnr, ssim, lpips_val = validate_full_pipeline(model, val_loader, cfg, epoch)
            if psnr > best_metric:
                best_metric = psnr
                save_checkpoint(
                    str(Path(cfg.checkpoint_dir) / "enhance_best.pth"),
                    model.enhance_net.state_dict(), optimizer.state_dict(), epoch, best_metric,
                )
                print(f"  -> New best Enhance-Net checkpoint (val PSNR {psnr:.2f} dB)")

        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(
                str(Path(cfg.checkpoint_dir) / f"enhance_epoch{epoch + 1}.pth"),
                model.enhance_net.state_dict(), optimizer.state_dict(), epoch, best_metric,
            )


def validate_full_pipeline(model: RetinexNet, val_loader, cfg: Config, epoch: int):
    """Runs the FULL RetinexNet pipeline (Decom + Enhance) and reports PSNR/SSIM/LPIPS,
    plus saves a visual sample panel for the first validation image."""
    model.eval()
    psnr_list, ssim_list, lpips_list = [], [], []

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            i_low = batch["low"].to(cfg.device)
            i_high = batch["high"].to(cfg.device)

            out = model(i_low)

            pred = out["enhanced_image"][0]
            target = i_high[0]

            psnr_list.append(compute_psnr(pred, target))
            ssim_list.append(compute_ssim(pred, target))
            if cfg.compute_lpips:
                lp = compute_lpips(pred, target, device=cfg.device)
                if lp is not None:
                    lpips_list.append(lp)

            if idx == 0:
                sample_path = str(Path(cfg.sample_dir) / f"epoch_{epoch + 1}_{batch['filename'][0]}")
                save_validation_panel(
                    sample_path,
                    low=i_low[0], enhanced=pred, high=target,
                    reflectance=out["reflectance"][0], illumination=out["illumination"][0],
                    enhanced_illumination=out["enhanced_illumination"][0],
                )

    avg_psnr = float(np.mean(psnr_list))
    avg_ssim = float(np.mean(ssim_list))
    avg_lpips = float(np.mean(lpips_list)) if lpips_list else float("nan")
    print(f"[Full pipeline] Epoch {epoch + 1} — PSNR: {avg_psnr:.2f} dB | "
          f"SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}")
    return avg_psnr, avg_ssim, avg_lpips


def main():
    parser = argparse.ArgumentParser(description="Train RetinexNet on LOL-v2-Real")
    parser = add_common_args(parser)
    parser.add_argument("--epochs_decom", type=int, default=100)
    parser.add_argument("--epochs_enhance", type=int, default=100)
    parser.add_argument("--lr_decom", type=float, default=1e-4)
    parser.add_argument("--lr_enhance", type=float, default=1e-4)
    parser.add_argument("--w_recon", type=float, default=1.0)
    parser.add_argument("--w_reflectance", type=float, default=0.01)
    parser.add_argument("--w_illum_smooth", type=float, default=0.1)
    parser.add_argument("--w_enhance_smooth", type=float, default=3.0)
    parser.add_argument("--resume_decom", type=str, default=None)
    parser.add_argument("--resume_enhance", type=str, default=None)
    parser.add_argument("--sample_dir", type=str, default="results/val_samples")
    parser.add_argument("--no_lpips", action="store_true", help="Skip LPIPS computation during validation")
    args = parser.parse_args()

    cfg = build_config_from_args(args, compute_lpips=not args.no_lpips)
    set_seed(cfg.seed)

    print(f"Using device: {cfg.device}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, training will run on CPU (slow).")

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.sample_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train pairs: {len(train_loader.dataset)} | Val pairs: {len(val_loader.dataset)}")

    model = RetinexNet().to(cfg.device)

    t0 = time.time()
    train_decom_stage(model, train_loader, val_loader, cfg)
    train_enhance_stage(model, train_loader, val_loader, cfg)
    print(f"\nTotal training time: {(time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
