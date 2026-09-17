"""
ARCHITECTURE 7H

Pretrained Swin-Tiny Encoder
        ↓
SE-enhanced multi-scale encoder features
        ↓
Bottleneck projection
        ↓
Transformer bottleneck
        ↓
Segmentation decoder
        ↓
EX + SE segmentation head
        ↓
Segmentation-guided feature fusion
        ↓
Transformer classification
        ↓
5-class DR grading

Main idea:
The pretrained encoder extracts retinal features.
The bottleneck is processed by a Transformer.
The Transformer bottleneck is decoded for segmentation.
The predicted EX/SE segmentation is then fed back into
the classification branch together with Transformer features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ============================================================
# SE BLOCK
# ============================================================

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()

        reduced = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):

        scale = self.pool(x)
        scale = self.fc(scale)

        return x * scale


# ============================================================
# ATTENTION GATE
# ============================================================

class AttentionGate(nn.Module):

    def __init__(self, gate_channels, skip_channels, inter_channels):

        super().__init__()

        self.theta = nn.Conv2d(
            skip_channels,
            inter_channels,
            kernel_size=1,
            bias=False
        )

        self.phi = nn.Conv2d(
            gate_channels,
            inter_channels,
            kernel_size=1,
            bias=False
        )

        self.psi = nn.Sequential(
            nn.Conv2d(
                inter_channels,
                1,
                kernel_size=1
            ),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        theta_x = self.theta(x)
        phi_g = self.phi(g)

        if theta_x.shape[-2:] != phi_g.shape[-2:]:

            phi_g = F.interpolate(
                phi_g,
                size=theta_x.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        attention = self.relu(theta_x + phi_g)
        attention = self.psi(attention)

        return x * attention


# ============================================================
# RESIDUAL CONV BLOCK
# ============================================================

class ResidualConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_ch)

        self.act = nn.GELU()

        if in_ch != out_ch:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_ch)
            )

        else:

            self.shortcut = nn.Identity()

    def forward(self, x):

        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + residual

        return self.act(x)


# ============================================================
# SWIN ENCODER
# ============================================================

class SwinEncoder(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        weights = (
            models.Swin_T_Weights.DEFAULT
            if pretrained
            else None
        )

        swin = models.swin_t(weights=weights)

        self.features = swin.features

    def forward(self, x):

        # Patch embedding
        x = self.features[0](x)

        s0 = x.permute(
            0, 3, 1, 2
        ).contiguous()

        # Stage 1
        x = self.features[1](x)

        e1 = x.permute(
            0, 3, 1, 2
        ).contiguous()

        # Stage 2
        x = self.features[2](x)
        x = self.features[3](x)

        e2 = x.permute(
            0, 3, 1, 2
        ).contiguous()

        # Stage 3
        x = self.features[4](x)
        x = self.features[5](x)

        e3 = x.permute(
            0, 3, 1, 2
        ).contiguous()

        # Stage 4
        x = self.features[6](x)
        x = self.features[7](x)

        e4 = x.permute(
            0, 3, 1, 2
        ).contiguous()

        return s0, e1, e2, e3, e4


# ============================================================
# TRANSFORMER BOTTLENECK
# ============================================================

class BottleneckTransformer(nn.Module):

    def __init__(
        self,
        feature_dim=768,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    ):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):

        B, C, H, W = x.shape

        tokens = x.flatten(
            2
        ).transpose(
            1, 2
        )

        tokens = self.transformer(tokens)

        tokens = self.norm(tokens)

        x = tokens.transpose(
            1, 2
        ).reshape(
            B,
            C,
            H,
            W
        )

        return x


# ============================================================
# UPSAMPLING BLOCK
# ============================================================

class UpBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch
    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=2,
            stride=2
        )

        self.attention = AttentionGate(
            gate_channels=out_ch,
            skip_channels=skip_ch,
            inter_channels=max(
                out_ch // 2,
                8
            )
        )

        self.conv = ResidualConvBlock(
            out_ch + skip_ch,
            out_ch
        )

    def forward(self, x, skip):

        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:

            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        skip = self.attention(
            x,
            skip
        )

        x = torch.cat(
            [x, skip],
            dim=1
        )

        x = self.conv(x)

        return x


# ============================================================
# SEGMENTATION-GUIDED CLASSIFIER
# ============================================================

class SegmentationGuidedTransformerClassifier(nn.Module):

    def __init__(
        self,
        feature_dim=768,
        num_classes=5,
        num_lesion_classes=2,
        num_heads=8,
        num_layers=2
    ):

        super().__init__()

        # Convert EX/SE mask into feature representation
        self.mask_encoder = nn.Sequential(

            nn.Conv2d(
                num_lesion_classes,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(
                128,
                feature_dim,
                kernel_size=1
            )
        )

        # Learnable strength of segmentation guidance
        self.guidance_scale = nn.Parameter(
            torch.tensor(0.5)
        )

        # Fuse image + segmentation features
        self.fusion = nn.Sequential(

            nn.Conv2d(
                feature_dim * 2,
                feature_dim,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(
            feature_dim
        )

        # Attention pooling
        self.attention_pool = nn.Sequential(

            nn.Linear(
                feature_dim,
                128
            ),

            nn.Tanh(),

            nn.Linear(
                128,
                1
            )
        )

        # Final classifier
        self.classifier = nn.Sequential(

            nn.LayerNorm(feature_dim),

            nn.Linear(
                feature_dim,
                256
            ),

            nn.GELU(),

            nn.Dropout(0.25),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(
        self,
        bottleneck,
        segmentation_logits
    ):

        # Convert logits to probabilities
        lesion_prob = torch.sigmoid(
            segmentation_logits
        )

        # Resize mask to bottleneck resolution
        lesion_prob = F.interpolate(
            lesion_prob,
            size=bottleneck.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Encode segmentation
        mask_features = self.mask_encoder(
            lesion_prob
        )

        # Normalize guidance strength
        scale = torch.sigmoid(
            self.guidance_scale
        )

        mask_features = (
            mask_features * scale
        )

        # Fuse original Transformer bottleneck
        # with segmentation information
        fused = torch.cat(
            [
                bottleneck,
                mask_features
            ],
            dim=1
        )

        fused = self.fusion(
            fused
        )

        # Transformer tokens
        B, C, H, W = fused.shape

        tokens = fused.flatten(
            2
        ).transpose(
            1,
            2
        )

        tokens = self.transformer(
            tokens
        )

        tokens = self.norm(
            tokens
        )

        # Attention pooling
        scores = self.attention_pool(
            tokens
        )

        weights = torch.softmax(
            scores,
            dim=1
        )

        pooled = (
            tokens * weights
        ).sum(dim=1)

        output = self.classifier(
            pooled
        )

        return output


# ============================================================
# COMPLETE JOINT MODEL
# ============================================================

class JointSegClsNet(nn.Module):

    def __init__(
        self,
        num_lesion_classes=2,
        num_dr_classes=5,
        pretrained=True
    ):

        super().__init__()

        # ----------------------------------------------------
        # 1. PRETRAINED ENCODER
        # ----------------------------------------------------

        self.encoder = SwinEncoder(
            pretrained=pretrained
        )

        # ----------------------------------------------------
        # 2. SE BLOCKS
        # ----------------------------------------------------

        self.se_stem = SEBlock(96)

        self.se1 = SEBlock(96)

        self.se2 = SEBlock(192)

        self.se3 = SEBlock(384)

        self.se4 = SEBlock(768)

        # ----------------------------------------------------
        # 3. TRANSFORMER BOTTLENECK
        # ----------------------------------------------------

        self.bottleneck_transformer = (
            BottleneckTransformer(
                feature_dim=768,
                num_heads=8,
                num_layers=2
            )
        )

        # ----------------------------------------------------
        # 4. SEGMENTATION DECODER
        # ----------------------------------------------------

        self.up4 = UpBlock(
            768,
            384,
            384
        )

        self.up3 = UpBlock(
            384,
            192,
            192
        )

        self.up2 = UpBlock(
            192,
            96,
            96
        )

        self.up1 = UpBlock(
            96,
            96,
            48
        )

        # ----------------------------------------------------
        # 5. SEGMENTATION HEAD
        # ----------------------------------------------------

        self.seg_head = nn.Conv2d(
            48,
            num_lesion_classes,
            kernel_size=1
        )

        # ----------------------------------------------------
        # 6. CLASSIFICATION
        # ----------------------------------------------------

        self.cls_head = (
            SegmentationGuidedTransformerClassifier(
                feature_dim=768,
                num_classes=num_dr_classes,
                num_lesion_classes=num_lesion_classes,
                num_heads=8,
                num_layers=2
            )
        )

    def forward(self, x):

        # ====================================================
        # ENCODER
        # ====================================================

        s0, e1, e2, e3, e4 = self.encoder(x)

        # SE enhancement
        s0 = self.se_stem(s0)

        e1 = self.se1(e1)

        e2 = self.se2(e2)

        e3 = self.se3(e3)

        e4 = self.se4(e4)

        # ====================================================
        # TRANSFORMER BOTTLENECK
        # ====================================================

        bottleneck = self.bottleneck_transformer(
            e4
        )

        # ====================================================
        # SEGMENTATION DECODER
        # ====================================================

        d4 = self.up4(
            bottleneck,
            e3
        )

        d3 = self.up3(
            d4,
            e2
        )

        d2 = self.up2(
            d3,
            e1
        )

        d1 = self.up1(
            d2,
            s0
        )

        # ====================================================
        # SEGMENTATION HEAD
        # ====================================================

        seg_out = self.seg_head(
            d1
        )

        seg_out = F.interpolate(
            seg_out,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        cls_out = self.cls_head(
            bottleneck,
            seg_out
        )

        return seg_out, cls_out, bottleneck