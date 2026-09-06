"""
ARCHITECTURE 6 (v3): Dual-Pathway Classification
Same custom encoder + transformer bottleneck + attention-gated decoder
as before, but classification now combines TWO signals:
  1. The CLS token (transformer's learned global summary)
  2. Global-average-pooled spatial bottleneck features (CNN-style,
     the same mechanism that worked well in Architectures 2/3B/5)
These are concatenated before the final classification layers, giving
the network a CNN-based "fallback" signal instead of relying entirely
on the transformer pathway, which was repeatedly collapsing to
coarse 2-3 class groupings.
"""
import torch
import torch.nn as nn


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
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, dropout=0.1)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:])
        attn_skip = self.attn(g=x, x=skip)
        x = torch.cat([x, attn_skip], dim=1)
        return self.conv(x)


class TransformerBottleneckWithCLS(nn.Module):
    def __init__(self, channels, num_layers=4, num_heads=8, mlp_ratio=2, max_tokens=1024, dropout=0.15):
        super().__init__()
        self.channels = channels
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

        num_tokens = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :num_tokens, :]
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        cls_out = tokens[:, 0]
        patch_out = tokens[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        return patch_out, cls_out


class CustomEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.stage0 = ConvBlock(in_channels, 32, dropout=0.05)
        self.pool0 = nn.MaxPool2d(2)
        self.stage1 = ConvBlock(32, 64, dropout=0.10)
        self.pool1 = nn.MaxPool2d(2)
        self.stage2 = ConvBlock(64, 128, dropout=0.15)
        self.pool2 = nn.MaxPool2d(2)
        self.stage3 = ConvBlock(128, 256, dropout=0.20)
        self.pool3 = nn.MaxPool2d(2)

        self.transformer_bottleneck = TransformerBottleneckWithCLS(
            channels=256, num_layers=4, num_heads=8, mlp_ratio=2, max_tokens=32 * 32,
        )
        self.bottleneck_proj = ConvBlock(256, 512, dropout=0.2)

    def forward(self, x):
        s0 = self.stage0(x); p0 = self.pool0(s0)
        s1 = self.stage1(p0); p1 = self.pool1(s1)
        s2 = self.stage2(p1); p2 = self.pool2(s2)
        s3 = self.stage3(p2); p3 = self.pool3(s3)

        patch_out, cls_token = self.transformer_bottleneck(p3)
        s4 = self.bottleneck_proj(patch_out)
        return s0, s1, s2, s3, s4, cls_token


class JointSegClsNet(nn.Module):
    """Architecture 6 v3: dual-pathway classification -- combines the
    transformer's CLS token with CNN-style pooled spatial features."""
    def __init__(self, num_lesion_classes=2, num_dr_classes=5):
        super().__init__()
        self.encoder = CustomEncoder(in_channels=3)

        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.up0 = UpBlock(64, 32, 32)
        self.seg_head = nn.Conv2d(32, num_lesion_classes, 1)

        # Pathway A: CNN-style global average pooling of spatial bottleneck (512-dim)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_proj = nn.Sequential(nn.Linear(512, 256), nn.GELU())

        # Pathway B: transformer CLS token (256-dim) -- projected to match
        self.cls_proj = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.GELU())

        # Combined classification head: concatenate both 256-dim pathways
        self.cls_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3),
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
        s0, s1, s2, s3, s4, cls_token = self.encoder(x)
        d3 = self.up3(s4, s3); d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1); d0 = self.up0(d1, s0)
        seg_out = self.seg_head(d0)
        seg_out = nn.functional.interpolate(seg_out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        spatial_feat = self.spatial_proj(self.spatial_pool(s4).flatten(1))  # (B, 256)
        cls_feat = self.cls_proj(cls_token)                                  # (B, 256)
        combined = torch.cat([spatial_feat, cls_feat], dim=1)                # (B, 512)
        cls_out = self.cls_head(combined)

        return seg_out, cls_out
