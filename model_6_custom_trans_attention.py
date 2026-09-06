"""
ARCHITECTURE 6: Custom CNN Encoder + TransUNet Bottleneck + Attention-Gated
Skip Connections, with a dedicated CLS token for classification.

- Encoder: fully custom convolutional stages (no pretrained backbone)
- Bottleneck: Transformer encoder layers processing patch tokens PLUS
  a learnable CLS token (ViT-style) that specializes in whole-image
  understanding for classification
- Decoder: U-Net style, with Attention Gates filtering every skip
  connection from the custom encoder
- Classification head: reads ONLY the CLS token (not spatial features)
- Segmentation head: reads the decoder's final spatial output
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
    """Like Architecture 4's transformer bottleneck, but adds ONE extra
    learnable 'CLS' token to the sequence (standard ViT technique). This
    token has no fixed spatial position -- it's free to learn to
    aggregate whatever global information is most useful for
    classification, separately from the patch tokens that get reshaped
    back into a spatial map for segmentation."""
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
        patch_tokens = x.flatten(2).transpose(1, 2)          # (B, H*W, C)
        cls_tokens = self.cls_token.expand(B, -1, -1)          # (B, 1, C)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)  # (B, 1+H*W, C)

        num_tokens = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :num_tokens, :]
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        cls_out = tokens[:, 0]                                 # (B, C) -- for classification
        patch_out = tokens[:, 1:].transpose(1, 2).reshape(B, C, H, W)  # for segmentation
        return patch_out, cls_out


class CustomEncoder(nn.Module):
    """Fully custom convolutional encoder -- no pretrained backbone,
    per faculty requirement."""
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

        patch_out, cls_out = self.transformer_bottleneck(p3)
        s4 = self.bottleneck_proj(patch_out)
        return s0, s1, s2, s3, s4, cls_out


class JointSegClsNet(nn.Module):
    """Architecture 6: custom conv encoder -> transformer bottleneck
    (with CLS token) -> attention-gated U-Net decoder. Classification
    reads the dedicated CLS token; segmentation reads the fully
    decoded spatial output."""
    def __init__(self, num_lesion_classes=2, num_dr_classes=5):
        super().__init__()
        self.encoder = CustomEncoder(in_channels=3)

        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.up0 = UpBlock(64, 32, 32)
        self.seg_head = nn.Conv2d(32, num_lesion_classes, 1)

        # Classification head reads the 256-dim CLS token directly --
        # a richer, dedicated representation rather than pooling
        # spatial features that also have to serve segmentation.
        self.cls_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_dr_classes)
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

        cls_out = self.cls_head(cls_token)
        return seg_out, cls_out
