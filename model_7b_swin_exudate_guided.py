"""
ARCHITECTURE 7B (faculty's "3B", exudate-guided classification):
Swin Transformer + SE Blocks + Attention Gates + segmentation-guided
Transformer classifier. FIXED: added num_lesion_classes parameter
(was hardcoded to 1 channel, incompatible with our 2-class EX+SE task).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        s = self.fc1(s); s = self.relu(s); s = self.fc2(s); s = self.sigmoid(s)
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
        if theta_x.shape[-2:] != phi_g.shape[-2:]:
            phi_g = F.interpolate(phi_g, size=theta_x.shape[-2:], mode="bilinear", align_corners=False)
        f = self.relu(theta_x + phi_g)
        psi = self.sigmoid(self.psi(f))
        return x * psi


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attn = AttentionGate(gate_channels=out_ch, skip_channels=skip_ch, inter_channels=max(out_ch // 2, 8))
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attn_skip = self.attn(g=x, x=skip)
        x = torch.cat([x, attn_skip], dim=1)
        return self.conv(x)


class SwinEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        swin = models.swin_t(weights=weights)
        self.features = swin.features

    def forward(self, x):
        x = self.features[0](x)
        s0 = x.permute(0, 3, 1, 2).contiguous()
        x = self.features[1](x)
        e1 = x.permute(0, 3, 1, 2).contiguous()
        x = self.features[2](x)
        x = self.features[3](x)
        e2 = x.permute(0, 3, 1, 2).contiguous()
        x = self.features[4](x)
        x = self.features[5](x)
        e3 = x.permute(0, 3, 1, 2).contiguous()
        x = self.features[6](x)
        x = self.features[7](x)
        e4 = x.permute(0, 3, 1, 2).contiguous()
        return s0, e1, e2, e3, e4


class ExudateTransformerClassifier(nn.Module):
    """Uses the predicted lesion segmentation to guide DR classification.
    FIXED: exudate_embedding now takes num_lesion_classes input channels
    (was hardcoded to 1)."""
    def __init__(self, feature_dim=768, num_classes=5, num_heads=8, num_layers=2, num_lesion_classes=2):
        super().__init__()
        self.exudate_embedding = nn.Sequential(
            nn.Conv2d(num_lesion_classes, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_dim, kernel_size=1)
        )
        self.feature_projection = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads, dim_feedforward=feature_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, features, lesion_mask):
        lesion_mask = F.interpolate(lesion_mask, size=features.shape[-2:], mode="bilinear", align_corners=False)
        lesion_features = self.exudate_embedding(lesion_mask)
        image_features = self.feature_projection(features)
        guided_features = image_features * (1.0 + lesion_features)

        B, C, H, W = guided_features.shape
        tokens = guided_features.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        pooled = tokens.mean(dim=1)
        cls_out = self.classifier(pooled)
        return cls_out


class JointSegClsNet(nn.Module):
    """Architecture 7B: Swin Transformer + SE + Attention Gates +
    segmentation-guided Transformer classification. FIXED: added
    num_lesion_classes parameter, wired into seg_head and the
    exudate-guided classifier's mask embedding."""
    def __init__(self, num_lesion_classes=2, num_dr_classes=5, pretrained=True):
        super().__init__()
        self.encoder = SwinEncoder(pretrained=pretrained)

        self.se_stem = SEBlock(96)
        self.se1 = SEBlock(96)
        self.se2 = SEBlock(192)
        self.se3 = SEBlock(384)
        self.se4 = SEBlock(768)

        self.up4 = UpBlock(in_ch=768, skip_ch=384, out_ch=384)
        self.up3 = UpBlock(in_ch=384, skip_ch=192, out_ch=192)
        self.up2 = UpBlock(in_ch=192, skip_ch=96, out_ch=96)
        self.up1 = UpBlock(in_ch=96, skip_ch=96, out_ch=48)

        self.seg_head = nn.Conv2d(48, num_lesion_classes, kernel_size=1)

        self.cls_head = ExudateTransformerClassifier(
            feature_dim=768, num_classes=num_dr_classes, num_heads=8, num_layers=2,
            num_lesion_classes=num_lesion_classes,
        )

    def forward(self, x):
        s0, e1, e2, e3, e4 = self.encoder(x)
        s0 = self.se_stem(s0); e1 = self.se1(e1); e2 = self.se2(e2)
        e3 = self.se3(e3); e4 = self.se4(e4)

        d4 = self.up4(e4, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d1 = self.up1(d2, s0)

        seg_out = self.seg_head(d1)
        seg_out = F.interpolate(seg_out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        lesion_prob = torch.sigmoid(seg_out)
        cls_out = self.cls_head(e4, lesion_prob)

        return seg_out, cls_out
