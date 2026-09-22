"""
config.py

Central configuration for the RetinexNet project.

Design choice: instead of scattering argparse flags across every script,
we define a single dataclass-based Config object here. train.py, test.py,
and evaluate.py all build a Config from command-line arguments, so paths,
hyperparameters, and loss weights are configurable in exactly one place.

Nothing here is hard-coded to a specific machine. All paths default to
relative folders under `dataset/` and can be overridden with CLI flags.
"""

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import torch


@dataclass
class Config:
    # ---------------- Dataset paths (override via CLI) ----------------
    data_root: str = "dataset"          # expects data_root/{train,val,test}/{low,high}
    train_low_dir: str = None           # auto-derived from data_root if None
    train_high_dir: str = None
    val_low_dir: str = None
    val_high_dir: str = None
    test_low_dir: str = None
    test_high_dir: str = None

    # ---------------- Image processing ----------------
    patch_size: int = 384               # random crop size used during training
    image_size: int = None              # if set, resize instead of crop (used for val/test)

    # ---------------- Training hyperparameters ----------------
    batch_size: int = 8
    num_workers: int = 4
    lr_decom: float = 1e-4
    lr_enhance: float = 1e-4
    epochs_decom: int = 100
    epochs_enhance: int = 100
    seed: int = 42

    # ---------------- Loss weights ----------------
    # L_total = w_recon * L_reconstruction
    #         + w_reflectance * L_reflectance
    #         + w_illum_smooth * L_illumination_smoothness (Decom-Net stage)
    #         + w_enhance_smooth * L_illumination_smoothness (Enhance-Net stage)
    w_recon: float = 1.0
    w_recon_low_to_high: float = 0.001  # weight for the cross reconstruction term (see losses doc)
    w_reflectance: float = 0.01
    w_illum_smooth: float = 0.1
    w_enhance_smooth: float = 3.0

    # ---------------- Checkpointing ----------------
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10                # periodic checkpoint frequency (epochs)
    resume_decom: str = None
    resume_enhance: str = None

    # ---------------- Validation / output ----------------
    val_every: int = 1
    sample_dir: str = "results/val_samples"
    compute_lpips: bool = True

    # ---------------- Device ----------------
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        root = Path(self.data_root)
        if self.train_low_dir is None:
            self.train_low_dir = str(root / "train" / "low")
        if self.train_high_dir is None:
            self.train_high_dir = str(root / "train" / "high")
        if self.val_low_dir is None:
            self.val_low_dir = str(root / "val" / "low")
        if self.val_high_dir is None:
            self.val_high_dir = str(root / "val" / "high")
        if self.test_low_dir is None:
            self.test_low_dir = str(root / "test" / "low")
        if self.test_high_dir is None:
            self.test_high_dir = str(root / "test" / "high")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Shared CLI arguments used by train.py, test.py, and evaluate.py."""
    parser.add_argument("--data_root", type=str, default="dataset",
                         help="Root folder containing train/val/test splits.")
    parser.add_argument("--train_low_dir", type=str, default=None)
    parser.add_argument("--train_high_dir", type=str, default=None)
    parser.add_argument("--val_low_dir", type=str, default=None)
    parser.add_argument("--val_high_dir", type=str, default=None)
    parser.add_argument("--test_low_dir", type=str, default=None)
    parser.add_argument("--test_high_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=384)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    return parser


def build_config_from_args(args: argparse.Namespace, **overrides) -> Config:
    """Merge parsed CLI args (and any extra overrides) into a Config object."""
    cfg = Config()
    for key, value in vars(args).items():
        if hasattr(cfg, key) and value is not None:
            setattr(cfg, key, value)
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    cfg.__post_init__()
    return cfg
