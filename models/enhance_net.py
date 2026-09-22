"""
models/enhance_net.py

Enhance-Net: brightens the illumination map L predicted by Decom-Net.

Original paper design
----------------------
RetinexNet's Enhance-Net is a small encoder-decoder ("U-Net-like") network
with multi-scale feature fusion:
  - It takes the concatenation of (reflectance, illumination) as input —
    NOT illumination alone. The paper conditions enhancement on reflectance
    too, because noise in the reflectance (which is amplified when
    illumination is later divided/multiplied back in) needs to be
    accounted for while brightening.
  - It downsamples with strided convs, processes at multiple scales, then
    upsamples and fuses multi-scale features via concatenation, followed
    by 1x1 convs, similar in spirit to a shallow U-Net / FPN.
  - It ends with a single conv producing the enhanced illumination (1
    channel).

Practical modification (explicitly stated)
--------------------------------------------
The paper's official implementation resizes intermediate feature maps to a
common resolution before concatenating multi-scale features. We implement
this with `F.interpolate` using bilinear upsampling, which is the standard
modern-PyTorch equivalent of the original TensorFlow `tf.image.resize`
calls used in the authors' released code. This is a direct, faithful port
— not a simplification — but we call it out because the original code was
written in TensorFlow 1.x and the resizing op names differ.

We do NOT add BatchNorm (again matching the original, for the same reason
given in decom_net.py: batch statistics can distort absolute illumination
scale).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhanceNet(nn.Module):
    def __init__(self, num_channels: int = 64):
        super().__init__()

        # Input: reflectance (3) + illumination (1) = 4 channels.
        self.conv0 = nn.Conv2d(4, num_channels, kernel_size=3, padding=1)

        # Encoder (downsampling path).
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=1)

        # Decoder (upsampling + skip fusion path).
        self.deconv1 = nn.Conv2d(num_channels * 2, num_channels, kernel_size=3, padding=1)
        self.deconv2 = nn.Conv2d(num_channels * 2, num_channels, kernel_size=3, padding=1)
        self.deconv3 = nn.Conv2d(num_channels * 2, num_channels, kernel_size=3, padding=1)

        # Multi-scale feature fusion: resize each decoder stage's output to
        # the input resolution and concatenate before the final prediction.
        self.fusion = nn.Conv2d(num_channels * 3, num_channels, kernel_size=1)

        self.final_conv = nn.Conv2d(num_channels, 1, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, reflectance: torch.Tensor, illumination: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reflectance: (B, 3, H, W)
            illumination: (B, 1, H, W)

        Returns:
            enhanced_illumination: (B, 1, H, W), values in [0, 1]
        """
        x = torch.cat([reflectance, illumination], dim=1)  # (B, 4, H, W)
        target_size = x.shape[2:]

        f0 = self.relu(self.conv0(x))          # full resolution
        f1 = self.relu(self.conv1(f0))         # /2
        f2 = self.relu(self.conv2(f1))         # /4
        f3 = self.relu(self.conv3(f2))         # /8

        # Upsample + skip-connect back up (U-Net style).
        up3 = F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.relu(self.deconv3(torch.cat([up3, f2], dim=1)))

        up2 = F.interpolate(d3, size=f1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.relu(self.deconv2(torch.cat([up2, f1], dim=1)))

        up1 = F.interpolate(d2, size=f0.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.relu(self.deconv1(torch.cat([up1, f0], dim=1)))

        # Multi-scale fusion: bring d1, d2, d3 to the same (input) resolution.
        d3_up = F.interpolate(d3, size=target_size, mode="bilinear", align_corners=False)
        d2_up = F.interpolate(d2, size=target_size, mode="bilinear", align_corners=False)
        d1_up = F.interpolate(d1, size=target_size, mode="bilinear", align_corners=False)

        fused = self.fusion(torch.cat([d1_up, d2_up, d3_up], dim=1))
        out = self.sigmoid(self.final_conv(fused))
        return out
