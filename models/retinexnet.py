"""
models/retinexnet.py

Full RetinexNet: wires Decom-Net and Enhance-Net together and implements
the Retinex reconstruction: Enhanced Image = Reflectance * Enhanced Illumination.

Training strategy (matches the original paper)
--------------------------------------------------
The original RetinexNet is trained in TWO STAGES, not end-to-end jointly:

  Stage 1 (Decom-Net): trained on BOTH low-light and high-light images from
  the same pair, using reconstruction + reflectance-consistency +
  illumination-smoothness losses (see losses/retinex_losses.py). This
  teaches Decom-Net that a low/high image PAIR should share (nearly) the
  same reflectance, differing mainly in illumination.

  Stage 2 (Enhance-Net): Decom-Net's weights are FROZEN. Enhance-Net is
  trained to map the low-light illumination to the high-light illumination,
  using an enhancement loss + a smoothness loss on the enhanced map.

This two-stage design is why `train.py` exposes `epochs_decom` and
`epochs_enhance` separately, and why this module provides both a joint
forward pass (used at test/inference time) and access to each sub-network
(used to freeze Decom-Net during Stage 2).
"""

import torch
import torch.nn as nn

from models.decom_net import DecomNet
from models.enhance_net import EnhanceNet


class RetinexNet(nn.Module):
    def __init__(self, decom_layers: int = 5, decom_channels: int = 64, enhance_channels: int = 64):
        super().__init__()
        self.decom_net = DecomNet(num_layers=decom_layers, num_channels=decom_channels)
        self.enhance_net = EnhanceNet(num_channels=enhance_channels)

    def decompose(self, x: torch.Tensor):
        """Run only Decom-Net. Returns (reflectance, illumination)."""
        return self.decom_net(x)

    def forward(self, low_img: torch.Tensor):
        """
        Full inference pipeline used at test time.

        Args:
            low_img: (B, 3, H, W) low-light input in [0, 1].

        Returns:
            dict with reflectance, illumination, enhanced_illumination,
            and the final enhanced_image = reflectance * enhanced_illumination.
        """
        reflectance, illumination = self.decom_net(low_img)
        enhanced_illum = self.enhance_net(reflectance, illumination)

        # Broadcast the 1-channel illumination across the 3 reflectance
        # channels for the element-wise Retinex reconstruction.
        enhanced_image = reflectance * enhanced_illum.repeat(1, 3, 1, 1)
        enhanced_image = torch.clamp(enhanced_image, 0.0, 1.0)

        return {
            "reflectance": reflectance,
            "illumination": illumination,
            "enhanced_illumination": enhanced_illum,
            "enhanced_image": enhanced_image,
        }

    def freeze_decom(self):
        for p in self.decom_net.parameters():
            p.requires_grad = False
        self.decom_net.eval()

    def unfreeze_decom(self):
        for p in self.decom_net.parameters():
            p.requires_grad = True
        self.decom_net.train()
