"""
models/decom_net.py

Decom-Net: decomposes an RGB image I into reflectance R (3 channels) and
illumination L (1 channel), following the architecture described in
"Deep Retinex Decomposition for Low-Light Enhancement" (Wei et al., BMVC 2018).

Why 3 channels for reflectance and 1 channel for illumination?
----------------------------------------------------------------
Retinex theory models a color image as I = R * L, where the multiplication
is element-wise. Two design choices follow directly from the physics:

1. Reflectance (R) captures the intrinsic color/texture of a scene's
   surfaces (what a "well-lit" image looks like) and is therefore a
   3-channel RGB quantity — one value per color channel, just like the
   input image.

2. Illumination (L) captures the light intensity falling on the scene.
   The original paper treats illumination as a single achromatic
   (grayscale) map shared across all three color channels: the same L
   multiplies R, G, and B. This matches the physical intuition that a
   light source's brightness doesn't usually change hue per-channel,
   only intensity. So L has exactly 1 channel, and is broadcast
   (repeated) across the 3 reflectance channels when reconstructing I.

Architecture (following the paper as closely as practical)
------------------------------------------------------------
The original Decom-Net:
  - Concatenates the RGB input with its max-channel (per-pixel max across
    R,G,B) as an extra input channel. This max-channel acts as an initial
    illumination estimate/prior that helps the network converge faster.
  - Uses a first large-kernel conv (9x9) to give the shallow network a
    large receptive field without adding many layers.
  - Stacks several 3x3 conv + ReLU layers.
  - Ends with a conv producing 4 output channels (3 for R, 1 for L),
    followed by sigmoid to keep values in [0, 1].

Practical modification (clearly stated, not attributed to the original paper):
We use 5 middle conv layers (the paper uses 5 as well in most public
reference implementations), but we add optional BatchNorm-free design
(the original RetinexNet authors do NOT use batch norm in Decom-Net,
and neither do we, since batch statistics computed on small low-light
patches can shift the illumination scale in ways that hurt the Retinex
decomposition). This is consistent with the original design, not a
deviation.
"""

import torch
import torch.nn as nn


class DecomNet(nn.Module):
    def __init__(self, num_layers: int = 5, num_channels: int = 64):
        """
        Args:
            num_layers: number of middle 3x3 conv+ReLU blocks (paper uses 5).
            num_channels: width of the middle feature maps (paper uses 64).
        """
        super().__init__()

        # Input: 4 channels = 3 (RGB) + 1 (per-pixel max across RGB, used as
        # an illumination prior, exactly as in the original paper's public
        # reference implementation).
        self.first_conv = nn.Conv2d(4, num_channels, kernel_size=9, padding=4)

        middle_layers = []
        for _ in range(num_layers):
            middle_layers.append(nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1))
            middle_layers.append(nn.ReLU(inplace=True))
        self.middle_convs = nn.Sequential(*middle_layers)

        # Output: 3 (reflectance) + 1 (illumination) = 4 channels.
        self.final_conv = nn.Conv2d(num_channels, 4, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _max_channel(x: torch.Tensor) -> torch.Tensor:
        """Per-pixel max across the RGB channels -> (B, 1, H, W)."""
        return torch.max(x, dim=1, keepdim=True)[0]

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input RGB image, tensor of shape (B, 3, H, W), values in [0, 1].

        Returns:
            reflectance: (B, 3, H, W), values in [0, 1]
            illumination: (B, 1, H, W), values in [0, 1]
        """
        illum_prior = self._max_channel(x)
        net_input = torch.cat([x, illum_prior], dim=1)  # (B, 4, H, W)

        feat = self.relu(self.first_conv(net_input))
        feat = self.middle_convs(feat)
        out = self.final_conv(feat)
        out = self.sigmoid(out)

        reflectance = out[:, 0:3, :, :]
        illumination = out[:, 3:4, :, :]
        return reflectance, illumination
