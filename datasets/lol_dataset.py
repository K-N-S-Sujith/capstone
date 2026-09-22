"""
datasets/lol_dataset.py

Paired low/high-light dataset loader for LOL-v2-Real.

LOL-v2-Real ships as two folders of images with matching filenames:
    train/low/xxx.png   <-> train/high/xxx.png
This loader pairs them by sorted filename order (falling back to matching
by stem if the extensions differ), applies synchronized augmentation
(the SAME random crop / flip must be applied to both the low and high
image, otherwise the pixel-wise Retinex losses become meaningless), and
returns normalized tensors in [0, 1].

We do NOT use ImageNet mean/std normalization here. RetinexNet's Retinex
decomposition (I = R * L) assumes pixel values are non-negative reflectance
and illumination products, which only makes sense in the [0, 1] range.
Standardizing to zero-mean would break the multiplicative decomposition.
"""

from pathlib import Path
from typing import Optional, Tuple, List
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def _list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}\n"
            f"Expected structure: <data_root>/<split>/<low|high>/*.png"
        )
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])
    if len(files) == 0:
        raise FileNotFoundError(f"No images with extensions {IMG_EXTENSIONS} found in {folder}")
    return files


class LOLDataset(Dataset):
    """
    Paired low-light / normal-light dataset.

    Args:
        low_dir: path to low-light images.
        high_dir: path to ground-truth normal-light images. If None, the
            dataset operates in "unpaired/inference" mode and returns only
            the low-light image (used by test.py when ground truth is
            unavailable).
        patch_size: if set, a random square crop of this size is taken
            (training mode). Mutually exclusive with image_size.
        image_size: if set, both images are resized to (image_size, image_size)
            instead of cropped (used for validation/testing where we want
            deterministic, full-image evaluation).
        augment: whether to apply random horizontal/vertical flips (training only).
    """

    def __init__(
        self,
        low_dir: str,
        high_dir: Optional[str] = None,
        patch_size: Optional[int] = 384,
        image_size: Optional[int] = None,
        augment: bool = False,
    ):
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir) if high_dir is not None else None
        self.patch_size = patch_size
        self.image_size = image_size
        self.augment = augment

        self.low_files = _list_images(self.low_dir)

        if self.high_dir is not None:
            self.high_files = _list_images(self.high_dir)
            if len(self.low_files) != len(self.high_files):
                raise ValueError(
                    f"Mismatched pair counts: {len(self.low_files)} low images in "
                    f"{self.low_dir} vs {len(self.high_files)} high images in {self.high_dir}. "
                    f"LOL-v2-Real requires one high-light ground truth per low-light image."
                )
        else:
            self.high_files = None

    def __len__(self) -> int:
        return len(self.low_files)

    def _load(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _sync_crop(self, low: Image.Image, high: Optional[Image.Image]):
        """Apply the SAME random crop to both images so pixels stay aligned."""
        w, h = low.size
        ps = self.patch_size
        if w < ps or h < ps:
            # Upscale small images so a full-size patch can be extracted.
            scale = ps / min(w, h)
            new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
            low = low.resize((new_w, new_h), Image.BICUBIC)
            if high is not None:
                high = high.resize((new_w, new_h), Image.BICUBIC)
            w, h = new_w, new_h

        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        low = low.crop((x, y, x + ps, y + ps))
        if high is not None:
            high = high.crop((x, y, x + ps, y + ps))
        return low, high

    def _sync_flip(self, low: Image.Image, high: Optional[Image.Image]):
        if random.random() < 0.5:
            low = TF.hflip(low)
            high = TF.hflip(high) if high is not None else None
        if random.random() < 0.5:
            low = TF.vflip(low)
            high = TF.vflip(high) if high is not None else None
        return low, high

    def __getitem__(self, idx: int):
        low_path = self.low_files[idx]
        low_img = self._load(low_path)
        high_img = self._load(self.high_files[idx]) if self.high_files is not None else None

        if self.patch_size is not None and self.image_size is None:
            low_img, high_img = self._sync_crop(low_img, high_img)
        elif self.image_size is not None:
            low_img = low_img.resize((self.image_size, self.image_size), Image.BICUBIC)
            if high_img is not None:
                high_img = high_img.resize((self.image_size, self.image_size), Image.BICUBIC)

        if self.augment:
            low_img, high_img = self._sync_flip(low_img, high_img)

        low_tensor = TF.to_tensor(low_img)  # [0, 1], shape (3, H, W)
        sample = {"low": low_tensor, "filename": low_path.name}

        if high_img is not None:
            sample["high"] = TF.to_tensor(high_img)

        return sample
