import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import ssl
import os

# Create unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

class TransformerColorizer(nn.Module):
    def __init__(self, upscale_factor=2):
        super(TransformerColorizer, self).__init__()

        # Use pretrained MobileNetV3 as encoder with newer weights parameter
        self.encoder = models.mobilenet_v3_small(weights='DEFAULT').features

        # Decoder for colorization (outputs 2 channels: U and V in YUV)
        self.color_head = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, kernel_size=3, padding=1),
            nn.Tanh()  # U and V values between -1 and 1
        )

        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.Conv2d(1 + 2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64 * upscale_factor**2, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, kernel_size=1)
        )

    def forward(self, x_gray):
        # x_gray: [B, 1, H, W] grayscale
        # Ensure input is 3 channels by repeating the grayscale channel
        if x_gray.size(1) == 1:
            x_3ch = x_gray.repeat(1, 3, 1, 1)  # Convert to 3 channel
        else:
            x_3ch = x_gray[:, :3, :, :]  # Take first 3 channels if more are provided
        
        features = self.encoder(x_3ch)
        uv = self.color_head(features)  # Predict UV
        uv_upsampled = F.interpolate(
            uv, size=x_gray.shape[2:], mode='bilinear', align_corners=False)
        color_input = torch.cat([x_gray[:, :1, :, :], uv_upsampled], dim=1)  # Ensure we only use 1 channel from input
        out = self.upsample(color_input)  # Upscale and produce RGB
        return out
