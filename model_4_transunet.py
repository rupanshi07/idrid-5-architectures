"""
ARCHITECTURE 4: TransUNet (hybrid CNN + Vision Transformer)
First 3 stages are convolutional (cheap local feature extraction).
The bottleneck replaces a conv block with a lightweight Transformer
Encoder, giving the network global self-attention across the whole
feature map before decoding -- the core idea of the TransUNet paper
(Chen et al., 2021), scaled down to fit a 4GB GPU.
"""
import torch
import torch.nn as nn


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
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TransformerBottleneck(nn.Module):
    """Flattens a (B, C, H, W) feature map into a sequence of H*W tokens,
    adds learned positional embeddings, passes through a small stack of
    standard Transformer encoder layers (multi-head self-attention +
    feed-forward), then reshapes back to (B, C, H, W)."""
    def __init__(self, channels, num_layers=4, num_heads=8, mlp_ratio=2, max_tokens=1024, dropout=0.1):
        super().__init__()
        self.channels = channels
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=num_heads, dim_feedforward=channels * mlp_ratio,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)          # (B, H*W, C)
        num_tokens = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :num_tokens, :]
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        out = tokens.transpose(1, 2).reshape(B, C, H, W)
        return out


class HybridEncoder(nn.Module):
    """Stages 0-2 are convolutional (as in Architecture 1). Stage 3 and
    the bottleneck are replaced by the Transformer bottleneck operating
    at 32x32 resolution, 256 channels -- kept deliberately small
    (256-dim tokens, 4 layers, 8 heads) to fit a 4GB GPU."""
    def __init__(self, in_channels=3):
        super().__init__()
        self.stage0 = ConvBlock(in_channels, 32)
        self.pool0 = nn.MaxPool2d(2)
        self.stage1 = ConvBlock(32, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.stage2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.stage3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.transformer_bottleneck = TransformerBottleneck(
            channels=256, num_layers=4, num_heads=8, mlp_ratio=2, max_tokens=32 * 32,
        )
        self.bottleneck_proj = ConvBlock(256, 512)

    def forward(self, x):
        s0 = self.stage0(x); p0 = self.pool0(s0)
        s1 = self.stage1(p0); p1 = self.pool1(s1)
        s2 = self.stage2(p1); p2 = self.pool2(s2)
        s3 = self.stage3(p2); p3 = self.pool3(s3)

        transformed = self.transformer_bottleneck(p3)
        s4 = self.bottleneck_proj(transformed)
        return s0, s1, s2, s3, s4


class JointSegClsNet(nn.Module):
    """Architecture 4: TransUNet-style hybrid CNN + Transformer bottleneck."""
    def __init__(self, num_lesion_classes=2, num_dr_classes=5):
        super().__init__()
        self.encoder = HybridEncoder(in_channels=3)

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
