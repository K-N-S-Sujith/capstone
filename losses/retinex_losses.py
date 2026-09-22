"""
losses/retinex_losses.py

Implements the loss functions from "Deep Retinex Decomposition for
Low-Light Enhancement" (Wei et al., BMVC 2018), used across the two
training stages (Decom-Net, then Enhance-Net).

Notation:
    I_low, I_high : low-light and high-light (ground truth) input images
    R_low, L_low  : reflectance / illumination decomposed from I_low
    R_high, L_high: reflectance / illumination decomposed from I_high
    L_low_hat     : enhanced illumination produced by Enhance-Net from L_low

-------------------------------------------------------------------------
1) Reconstruction loss  (L_reconstruction)   -- used in Stage 1 (Decom-Net)
-------------------------------------------------------------------------
Retinex theory says I = R * L. For a correct decomposition, multiplying
the predicted reflectance and illumination back together must reproduce
the original image. We enforce this on BOTH images of the pair, AND with
a small CROSS term that swaps illumination/reflectance between the pair.
The cross term encodes the assumption that reflectance is shared across
the pair (see loss 2 below): if R_low can pair with L_high to reconstruct
something image-like, that's consistent with a shared-reflectance world.

    L_recon = sum over {(R_low, L_low, I_low), (R_high, L_high, I_high)} of
                  || R * L - I ||_1
            + w_cross * ( || R_low * L_high - I_high ||_1
                        + || R_high * L_low - I_low ||_1 )

We use the L1 norm (mean absolute error), matching the original paper,
because L1 produces sharper reconstructions than L2/MSE (L2 tends to
average competing plausible solutions into a blurrier result).

-------------------------------------------------------------------------
2) Reflectance consistency loss  (L_reflectance)  -- Stage 1 (Decom-Net)
-------------------------------------------------------------------------
A low-light photo and its well-lit counterpart are photos of the SAME
scene: the true surface reflectance shouldn't change, only the amount of
light falling on it. So we penalize the difference between the two
predicted reflectance maps:

    L_reflectance = || R_low - R_high ||_1

This is what actually teaches Decom-Net to push illumination-dependent
variation into L and keep R illumination-invariant.

-------------------------------------------------------------------------
3) Illumination smoothness loss  (L_smoothness)  -- Stage 1 (Decom-Net)
-------------------------------------------------------------------------
Illumination is expected to vary smoothly over large regions (light
doesn't have hard edges except where reflectance/texture edges are).
We therefore penalize the gradient of L, but WEIGHT that penalty down at
locations with strong reflectance gradients (i.e., real texture/object
edges), so the network is allowed to keep L sharp exactly where the scene
itself has edges, and smooth everywhere else. This is an
"edge-aware total variation" loss:

    L_smoothness = sum_{x in {h,v}} || grad_x(L) * exp(-lambda_g * grad_x(R)) ||_1

where grad_x is the image gradient in the horizontal/vertical direction,
R here is first converted to grayscale (illumination should not depend on
color, only on structure), and lambda_g controls how strongly reflectance
edges suppress the smoothness penalty (we follow the paper's lambda_g = 10).
This loss is applied to BOTH L_low (against R_low) and L_high (against R_high).

-------------------------------------------------------------------------
4) Illumination enhancement loss  (L_illumination / Stage 2, Enhance-Net)
-------------------------------------------------------------------------
Once Decom-Net is frozen, Enhance-Net must learn to map L_low -> something
close to L_high (the ground-truth illumination extracted from the WELL-LIT
image). This is a direct supervised L1 regression:

    L_illum_enhance = || L_low_hat - L_high ||_1

In addition, the same edge-aware smoothness term from (3) is re-applied to
the ENHANCED illumination L_low_hat (again weighted by R_low's gradient),
so the brightened illumination doesn't introduce halo artifacts:

    L_enhance_smooth = smoothness(L_low_hat, R_low)

-------------------------------------------------------------------------
Combining everything
-------------------------------------------------------------------------
Stage 1 total (Decom-Net):
    L_decom = w_recon * L_reconstruction
            + w_reflectance * L_reflectance
            + w_illum_smooth * (smoothness(L_low, R_low) + smoothness(L_high, R_high))

Stage 2 total (Enhance-Net, Decom-Net frozen):
    L_enhance = L_illum_enhance + w_enhance_smooth * L_enhance_smooth

These weights are all configurable via config.py; the specific numeric
defaults (w_reflectance=0.01, w_illum_smooth=0.1, w_enhance_smooth=3.0,
w_cross=0.001) follow the values reported in the original paper's
official implementation. If you change datasets or resolutions, these
may need re-tuning — nothing about them is fundamentally tied to LOL-v2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """Standard luminance-weighted grayscale conversion, (B,3,H,W) -> (B,1,H,W)."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _gradient(x: torch.Tensor, direction: str) -> torch.Tensor:
    """Simple forward-difference image gradient along 'h' or 'v'."""
    if direction == "h":
        return torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    elif direction == "v":
        return torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    else:
        raise ValueError("direction must be 'h' or 'v'")


def smoothness_loss(illumination: torch.Tensor, reflectance: torch.Tensor, lambda_g: float = 10.0) -> torch.Tensor:
    """
    Edge-aware illumination smoothness loss (see doc header, item 3).

    Args:
        illumination: (B, 1, H, W)
        reflectance:  (B, 3, H, W) — used only to locate real edges.
        lambda_g: how strongly reflectance edges suppress the smoothness penalty.
    """
    gray_reflectance = _rgb_to_gray(reflectance)

    illum_grad_h = _gradient(illumination, "h")
    illum_grad_v = _gradient(illumination, "v")
    refl_grad_h = _gradient(gray_reflectance, "h")
    refl_grad_v = _gradient(gray_reflectance, "v")

    weight_h = torch.exp(-lambda_g * refl_grad_h)
    weight_v = torch.exp(-lambda_g * refl_grad_v)

    loss_h = torch.mean(illum_grad_h * weight_h)
    loss_v = torch.mean(illum_grad_v * weight_v)
    return loss_h + loss_v


def reconstruction_loss(
    r_low: torch.Tensor, l_low: torch.Tensor, i_low: torch.Tensor,
    r_high: torch.Tensor, l_high: torch.Tensor, i_high: torch.Tensor,
    w_cross: float = 0.001,
) -> torch.Tensor:
    """Self- and cross-reconstruction L1 loss (see doc header, item 1)."""
    l_low_3ch = l_low.repeat(1, 3, 1, 1)
    l_high_3ch = l_high.repeat(1, 3, 1, 1)

    recon_low = F.l1_loss(r_low * l_low_3ch, i_low)
    recon_high = F.l1_loss(r_high * l_high_3ch, i_high)

    cross_low = F.l1_loss(r_high * l_low_3ch, i_low)
    cross_high = F.l1_loss(r_low * l_high_3ch, i_high)

    return recon_low + recon_high + w_cross * (cross_low + cross_high)


def reflectance_consistency_loss(r_low: torch.Tensor, r_high: torch.Tensor) -> torch.Tensor:
    """L1 distance between the two predicted reflectance maps (item 2)."""
    return F.l1_loss(r_low, r_high)


class DecomNetLoss(nn.Module):
    """Stage 1 combined loss for training Decom-Net."""

    def __init__(self, w_recon: float = 1.0, w_reflectance: float = 0.01,
                 w_illum_smooth: float = 0.1, w_cross: float = 0.001):
        super().__init__()
        self.w_recon = w_recon
        self.w_reflectance = w_reflectance
        self.w_illum_smooth = w_illum_smooth
        self.w_cross = w_cross

    def forward(self, r_low, l_low, r_high, l_high, i_low, i_high):
        recon = reconstruction_loss(r_low, l_low, i_low, r_high, l_high, i_high, self.w_cross)
        reflect = reflectance_consistency_loss(r_low, r_high)
        smooth = smoothness_loss(l_low, r_low) + smoothness_loss(l_high, r_high)

        total = self.w_recon * recon + self.w_reflectance * reflect + self.w_illum_smooth * smooth
        return total, {
            "recon": recon.item(),
            "reflectance": reflect.item(),
            "smoothness": smooth.item(),
            "total": total.item(),
        }


class EnhanceNetLoss(nn.Module):
    """Stage 2 combined loss for training Enhance-Net (Decom-Net frozen)."""

    def __init__(self, w_enhance_smooth: float = 3.0):
        super().__init__()
        self.w_enhance_smooth = w_enhance_smooth

    def forward(self, l_low_hat: torch.Tensor, l_high: torch.Tensor, r_low: torch.Tensor):
        enhance = F.l1_loss(l_low_hat, l_high)
        smooth = smoothness_loss(l_low_hat, r_low)

        total = enhance + self.w_enhance_smooth * smooth
        return total, {
            "illum_enhance": enhance.item(),
            "enhance_smoothness": smooth.item(),
            "total": total.item(),
        }
