"""
ARCHITECTURE 5: New Innovation -- Adversarial Segmentation Refinement (GAN)
The Generator reuses Architecture 3B (best-performing: ResNet34 encoder,
SE blocks, Attention Gates) for segmentation + classification. A deep
PatchGAN-style Discriminator (6 conv layers) is added, trained to tell
apart real ground-truth masks from generator-predicted masks -- pushing
the generator to produce sharper, more realistic segmentation boundaries
than pixel-wise loss alone can achieve.
"""
import torch
import torch.nn as nn
from model_3b_resnet_attention import JointSegClsNet as Generator


class Discriminator(nn.Module):
    """PatchGAN-style discriminator (as used in pix2pix / SegAN).
    Takes the image concatenated with a segmentation mask (5 channels:
    3 RGB + 2 lesion masks) and outputs a grid of real/fake scores
    (one per image patch, not a single scalar) -- this is deliberately
    deep (6 conv layers) to judge realism at multiple receptive-field
    scales."""
    def __init__(self, in_channels=5):
        super().__init__()
        def block(in_ch, out_ch, stride=2, use_bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, use_bn=False),   # 512 -> 256
            *block(64, 128),                          # 256 -> 128
            *block(128, 256),                         # 128 -> 64
            *block(256, 512),                         # 64 -> 32
            *block(512, 512),                         # 32 -> 16
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # 16 -> 15, patch scores
        )

    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)
        return self.model(x)
