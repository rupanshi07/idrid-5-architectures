"""
ARCHITECTURE 6 v5: Bottleneck-Split Design (per faculty's diagram)
Key change from v1-v4: segmentation and classification now branch
SEPARATELY from the raw encoder bottleneck, rather than both
depending on transformer-processed features.
  - Classification path: raw bottleneck -> Transformer -> CLS token -> classifier
  - Segmentation path: raw bottleneck -> attention-gated U-Net decoder (pure CNN, skips the transformer entirely)
Uses the enhanced (ResBlock + SE) encoder from v4, our best-performing
custom encoder so far.
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


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_ch)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        return self.relu(out)


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


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.attn = AttentionGate(gate_channels=out_ch, skip_channels=skip_ch,
                                   inter_channels=max(out_ch // 2, 8))
        self.conv = ResBlock(out_ch + skip_ch, out_ch, dropout=0.1)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:])
        attn_skip = self.attn(g=x, x=skip)
        x = torch.cat([x, attn_skip], dim=1)
        return self.conv(x)


class ClassificationTransformer(nn.Module):
    """Takes the RAW bottleneck feature map, adds a CLS token, runs
    through Transformer encoder layers. Used ONLY for classification --
    segmentation never sees this output, per the faculty's design."""
    def __init__(self, channels, num_layers=4, num_heads=8, mlp_ratio=2, max_tokens=1024, dropout=0.15):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, channels))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens + 1, channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=num_heads, dim_feedforward=channels * mlp_ratio,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        patch_tokens = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0]  # only the CLS token is needed -- segmentation doesn't use this path


class EnhancedEncoder(nn.Module):
    """4-stage ResBlock+SE encoder (our best-performing custom encoder
    from v4). Outputs the raw bottleneck (256ch, 32x32) UNMODIFIED by
    any transformer -- both downstream heads branch from here."""
    def __init__(self, in_channels=3):
        super().__init__()
        self.stage0 = ResBlock(in_channels, 32, dropout=0.05)
        self.pool0 = nn.MaxPool2d(2)
        self.stage1 = ResBlock(32, 64, dropout=0.10)
        self.pool1 = nn.MaxPool2d(2)
        self.stage2 = ResBlock(64, 128, dropout=0.15)
        self.pool2 = nn.MaxPool2d(2)
        self.stage3 = ResBlock(128, 256, dropout=0.20)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ResBlock(256, 256, dropout=0.20)  # raw bottleneck, 256ch @ 32x32

    def forward(self, x):
        s0 = self.stage0(x); p0 = self.pool0(s0)
        s1 = self.stage1(p0); p1 = self.pool1(s1)
        s2 = self.stage2(p1); p2 = self.pool2(s2)
        s3 = self.stage3(p2); p3 = self.pool3(s3)
        bottleneck_raw = self.bottleneck(p3)
        return s0, s1, s2, s3, bottleneck_raw


class JointSegClsNet(nn.Module):
    """Architecture 6 v5: bottleneck-split design.
    - bottleneck_raw -> Transformer -> classification
    - bottleneck_raw -> attention-gated CNN decoder -> segmentation
    (the two paths never interfere with each other)."""
    def __init__(self, num_lesion_classes=2, num_dr_classes=5):
        super().__init__()
        self.encoder = EnhancedEncoder(in_channels=3)

        self.cls_transformer = ClassificationTransformer(
            channels=256, num_layers=4, num_heads=8, mlp_ratio=2, max_tokens=32 * 32,
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_dr_classes)
        )

        # Segmentation decoder projects the raw 256ch bottleneck to 512ch,
        # then upsamples through 4 attention-gated blocks -- pure CNN,
        # completely bypassing the transformer.
        self.bottleneck_proj = ResBlock(256, 512, dropout=0.2)
        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.up0 = UpBlock(64, 32, 32)
        self.seg_head = nn.Conv2d(32, num_lesion_classes, 1)

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
        s0, s1, s2, s3, bottleneck_raw = self.encoder(x)

        # Classification path: raw bottleneck -> transformer -> CLS token
        cls_token = self.cls_transformer(bottleneck_raw)
        cls_out = self.cls_head(cls_token)

        # Segmentation path: raw bottleneck -> CNN decoder (transformer bypassed entirely)
        s4 = self.bottleneck_proj(bottleneck_raw)
        d3 = self.up3(s4, s3); d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1); d0 = self.up0(d1, s0)
        seg_out = self.seg_head(d0)
        seg_out = nn.functional.interpolate(seg_out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return seg_out, cls_out
