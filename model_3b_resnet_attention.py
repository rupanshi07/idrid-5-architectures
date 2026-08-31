"""
ARCHITECTURE 3B: Pretrained ResNet34 Encoder + SE Blocks + Attention Gates
Adds SE channel attention after each ResNet stage output, and Attention
Gates on every decoder skip connection -- tests whether attention adds
value ON TOP OF pretraining, not just instead of it.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)
        s = self.relu(self.fc1(s))
        s = self.sigmoid(self.fc2(s))
        return x * s


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.theta_x = nn.Conv2d(skip_channels, inter_channels, kernel_size=1)
        self.phi_g = nn.Conv2d(gate_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)
        f = self.relu(theta_x + phi_g)
        psi = self.sigmoid(self.psi(f))
        return x * psi


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.attn = AttentionGate(gate_channels=out_ch, skip_channels=skip_ch,
                                   inter_channels=max(out_ch // 2, 8))
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:])
        attn_skip = self.attn(g=x, x=skip)
        x = torch.cat([x, attn_skip], dim=1)
        return self.conv(x)


class JointSegClsNet(nn.Module):
    """Architecture 3B: pretrained ResNet34 encoder, SE blocks after each
    stage, Attention Gates on every decoder skip connection."""
    def __init__(self, num_lesion_classes=2, num_dr_classes=5, pretrained=True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.se_stem = SEBlock(64)
        self.pool = resnet.maxpool
        self.enc1 = resnet.layer1; self.se1 = SEBlock(64)
        self.enc2 = resnet.layer2; self.se2 = SEBlock(128)
        self.enc3 = resnet.layer3; self.se3 = SEBlock(256)
        self.enc4 = resnet.layer4; self.se4 = SEBlock(512)

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 32)
        self.seg_head = nn.Conv2d(32, num_lesion_classes, 1)

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_dr_classes)
        )

    def forward(self, x):
        s0 = self.se_stem(self.stem(x))
        p0 = self.pool(s0)
        e1 = self.se1(self.enc1(p0))
        e2 = self.se2(self.enc2(e1))
        e3 = self.se3(self.enc3(e2))
        e4 = self.se4(self.enc4(e3))

        d4 = self.up4(e4, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d1 = self.up1(d2, s0)

        seg_out = self.seg_head(d1)
        seg_out = nn.functional.interpolate(seg_out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cls_out = self.cls_head(e4)
        return seg_out, cls_out
