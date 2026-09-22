"""
utils/visualization.py

Saves side-by-side validation panels so you can visually inspect:
low-light input, enhanced output, ground truth, reflectance, illumination,
and enhanced illumination -- exactly the six panels requested for
validation sampling.
"""

from pathlib import Path
from typing import Optional, Dict
import torch
import torchvision.utils as vutils


def _prep(t: torch.Tensor) -> torch.Tensor:
    """Ensure a tensor is 3-channel and clamped to [0,1] for saving as an image."""
    t = t.detach().cpu().clamp(0, 1)
    if t.shape[0] == 1:  # grayscale illumination -> replicate to 3 channels for the grid
        t = t.repeat(3, 1, 1)
    return t


def save_validation_panel(
    save_path: str,
    low: torch.Tensor,
    enhanced: torch.Tensor,
    reflectance: torch.Tensor,
    illumination: torch.Tensor,
    enhanced_illumination: torch.Tensor,
    high: Optional[torch.Tensor] = None,
) -> None:
    """
    Saves a single row-grid image containing (in order):
    low-light | enhanced | ground truth (if available) | reflectance |
    illumination | enhanced illumination.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    panels = [_prep(low), _prep(enhanced)]
    if high is not None:
        panels.append(_prep(high))
    panels.extend([_prep(reflectance), _prep(illumination), _prep(enhanced_illumination)])

    grid = vutils.make_grid(panels, nrow=len(panels), padding=4)
    vutils.save_image(grid, save_path)
