"""
utils/metrics.py

Image quality metrics used during validation and evaluation.

Which metrics need ground truth?
-----------------------------------
* PSNR  -- needs a ground-truth normal-light image (it's a pixel-wise
           error metric: higher = closer to ground truth in dB).
* SSIM  -- needs ground truth (compares local luminance/contrast/structure
           statistics between two images).
* LPIPS -- needs ground truth (a learned perceptual distance between two
           images, using a pretrained network's feature space; lower = more
           perceptually similar).

None of these three can be computed from the enhanced image alone — they
are all *full-reference* metrics. If you ever run RetinexNet on truly
unlabeled low-light images (no paired ground truth), you can only report
qualitative results or no-reference metrics (e.g. NIQE), which are NOT
implemented here since LOL-v2-Real provides paired ground truth.
"""

from typing import Optional
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

_lpips_model = None  # lazy-loaded singleton


def _to_numpy_hwc(x: torch.Tensor) -> np.ndarray:
    """(C, H, W) tensor in [0,1] -> (H, W, C) numpy array in [0,1]."""
    return x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR between two (3, H, W) tensors in [0, 1]. Requires ground truth."""
    pred_np = _to_numpy_hwc(pred)
    target_np = _to_numpy_hwc(target)
    return float(sk_psnr(target_np, pred_np, data_range=1.0))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM between two (3, H, W) tensors in [0, 1]. Requires ground truth."""
    pred_np = _to_numpy_hwc(pred)
    target_np = _to_numpy_hwc(target)
    return float(sk_ssim(target_np, pred_np, data_range=1.0, channel_axis=2))


def compute_lpips(pred: torch.Tensor, target: torch.Tensor, device: str = "cpu") -> Optional[float]:
    """
    LPIPS between two (3, H, W) tensors in [0, 1]. Requires ground truth and
    the `lpips` package. Returns None if the package is not installed, so
    callers can gracefully skip this metric ("compute LPIPS if practical").
    """
    global _lpips_model
    try:
        import lpips as lpips_pkg
    except ImportError:
        return None

    if _lpips_model is None:
        _lpips_model = lpips_pkg.LPIPS(net="alex").to(device)
        _lpips_model.eval()

    # lpips expects inputs normalized to [-1, 1].
    pred_n = (pred.unsqueeze(0).to(device) * 2.0) - 1.0
    target_n = (target.unsqueeze(0).to(device) * 2.0) - 1.0

    with torch.no_grad():
        dist = _lpips_model(pred_n, target_n)
    return float(dist.item())
