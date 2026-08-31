"""
ARCHITECTURE 3A: Simple Encoder + SE Blocks + Attention Gates
Same from-scratch encoder as Architecture 1, now with Squeeze-and-
Excitation channel attention on every stage and Attention Gates on
every decoder skip connection.
"""
import torch
import torch.nn as nn


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
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
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
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, dropout=0.1)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:])
        attn_skip = self.attn(g=x, x=skip)
        x = torch.cat([x, attn_skip], dim=1)
        return self.conv(x)


class SimpleAttentionEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.stage0 = ConvBlock(in_channels, 32, dropout=0.05)
        self.se0 = SEBlock(32)
        self.pool0 = nn.MaxPool2d(2)

        self.stage1 = ConvBlock(32, 64, dropout=0.10)
        self.se1 = SEBlock(64)
        self.pool1 = nn.MaxPool2d(2)

        self.stage2 = ConvBlock(64, 128, dropout=0.15)
        self.se2 = SEBlock(128)
        self.pool2 = nn.MaxPool2d(2)

        self.stage3 = ConvBlock(128, 256, dropout=0.20)
        self.se3 = SEBlock(256)
        self.pool3 = nn.MaxPool2d(2)

        self.stage4 = ConvBlock(256, 512, dropout=0.20)
        self.se4 = SEBlock(512)

    def forward(self, x):
        s0 = self.se0(self.stage0(x)); p0 = self.pool0(s0)
        s1 = self.se1(self.stage1(p0)); p1 = self.pool1(s1)
        s2 = self.se2(self.stage2(p1)); p2 = self.pool2(s2)
        s3 = self.se3(self.stage3(p2)); p3 = self.pool3(s3)
        s4 = self.se4(self.stage4(p3))
        return s0, s1, s2, s3, s4


class JointSegClsNet(nn.Module):
    """Architecture 3A: from-scratch encoder + SE blocks + Attention Gates."""
    def __init__(self, num_lesion_classes=2, num_dr_classes=5):
        super().__init__()
        self.encoder = SimpleAttentionEncoder(in_channels=3)

        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.up0 = UpBlock(64, 32, 32)
        self.seg_head = nn.Conv2d(32, num_lesion_classes, 1)

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, num_dr_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        s0, s1, s2, s3, s4 = self.encoder(x)
        d3 = self.up3(s4, s3); d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1); d0 = self.up0(d1, s0)
        seg_out = self.seg_head(d0)
        seg_out = nn.functional.interpolate(seg_out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cls_out = self.cls_head(s4)
        return seg_out, cls_out
