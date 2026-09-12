"""
ARCHITECTURE 7C

Swin Transformer Tiny
        +
SE Blocks
        +
Attention U-Net Decoder
        +
EX-guided bounded attention
        +
Transformer classifier
        +
Global + EX-focused feature pooling
        +
Ordinal DR classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ============================================================
# SE BLOCK
# ============================================================

class SEBlock(nn.Module):

    def __init__(
        self,
        channels,
        reduction=8
    ):

        super().__init__()

        reduced = max(
            channels // reduction,
            4
        )

        self.pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc1 = nn.Conv2d(
            channels,
            reduced,
            kernel_size=1
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.fc2 = nn.Conv2d(
            reduced,
            channels,
            kernel_size=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        s = self.pool(x)

        s = self.fc1(s)

        s = self.relu(s)

        s = self.fc2(s)

        s = self.sigmoid(s)

        return x * s


# ============================================================
# ATTENTION GATE
# ============================================================

class AttentionGate(nn.Module):

    def __init__(
        self,
        gate_channels,
        skip_channels,
        inter_channels
    ):

        super().__init__()

        self.theta_x = nn.Conv2d(
            skip_channels,
            inter_channels,
            kernel_size=1
        )

        self.phi_g = nn.Conv2d(
            gate_channels,
            inter_channels,
            kernel_size=1
        )

        self.psi = nn.Conv2d(
            inter_channels,
            1,
            kernel_size=1
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        g,
        x
    ):

        theta_x = self.theta_x(x)

        phi_g = self.phi_g(g)

        if theta_x.shape[-2:] != phi_g.shape[-2:]:

            phi_g = F.interpolate(
                phi_g,
                size=theta_x.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        f = self.relu(
            theta_x + phi_g
        )

        psi = self.sigmoid(
            self.psi(f)
        )

        return x * psi


# ============================================================
# RESIDUAL CONV BLOCK
# ============================================================

class ResidualConvBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            out_ch
        )

        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            out_ch
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        if in_ch != out_ch:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(
                    out_ch
                )
            )

        else:

            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out = out + identity

        out = self.relu(out)

        return out


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

        self.attn = AttentionGate(
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

    def forward(
        self,
        x,
        skip
    ):

        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:

            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        attn_skip = self.attn(
            g=x,
            x=skip
        )

        x = torch.cat(
            [x, attn_skip],
            dim=1
        )

        return self.conv(x)


# ============================================================
# SWIN ENCODER
# ============================================================

class SwinEncoder(nn.Module):

    def __init__(
        self,
        pretrained=True
    ):

        super().__init__()

        weights = (
            models.Swin_T_Weights.DEFAULT
            if pretrained
            else None
        )

        swin = models.swin_t(
            weights=weights
        )

        self.features = swin.features

    def forward(self, x):

        # ----------------------------------------------------
        # Stage 0
        # ----------------------------------------------------

        x = self.features[0](x)

        s0 = x.permute(
            0,
            3,
            1,
            2
        ).contiguous()

        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------

        x = self.features[1](x)

        e1 = x.permute(
            0,
            3,
            1,
            2
        ).contiguous()

        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------

        x = self.features[2](x)

        x = self.features[3](x)

        e2 = x.permute(
            0,
            3,
            1,
            2
        ).contiguous()

        # ----------------------------------------------------
        # Stage 3
        # ----------------------------------------------------

        x = self.features[4](x)

        x = self.features[5](x)

        e3 = x.permute(
            0,
            3,
            1,
            2
        ).contiguous()

        # ----------------------------------------------------
        # Stage 4
        # ----------------------------------------------------

        x = self.features[6](x)

        x = self.features[7](x)

        e4 = x.permute(
            0,
            3,
            1,
            2
        ).contiguous()

        return (
            s0,
            e1,
            e2,
            e3,
            e4
        )


# ============================================================
# EX-GUIDED TRANSFORMER
# ============================================================

class ExudateGuidedTransformer(nn.Module):

    def __init__(
        self,
        feature_dim=768,
        num_heads=8,
        num_layers=2,
        num_classes=4
    ):

        super().__init__()

        # ----------------------------------------------------
        # EX mask embedding
        # ----------------------------------------------------

        self.ex_embedding = nn.Sequential(

            nn.Conv2d(
                1,
                128,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                128,
                feature_dim,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                feature_dim
            ),

            nn.ReLU(
                inplace=True
            )
        )

        # ----------------------------------------------------
        # BOUNDED EX GATE
        # ----------------------------------------------------

        self.ex_gate = nn.Sequential(

            nn.Conv2d(
                1,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                64,
                feature_dim,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

        # ----------------------------------------------------
        # Image feature projection
        # ----------------------------------------------------

        self.feature_projection = nn.Sequential(

            nn.Conv2d(
                feature_dim,
                feature_dim,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                feature_dim
            ),

            nn.ReLU(
                inplace=True
            )
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        encoder_layer = (
            nn.TransformerEncoderLayer(

                d_model=feature_dim,

                nhead=num_heads,

                dim_feedforward=feature_dim * 4,

                dropout=0.1,

                activation="gelu",

                batch_first=True,

                norm_first=True
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
        )

        # ----------------------------------------------------
        # Global + EX fusion
        # ----------------------------------------------------

        self.fusion = nn.Sequential(

            nn.Linear(
                feature_dim * 2,
                512
            ),

            nn.LayerNorm(
                512
            ),

            nn.GELU(),

            nn.Dropout(
                0.3
            )
        )

        # ----------------------------------------------------
        # Ordinal classifier
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            512,
            num_classes
        )

    def forward(
        self,
        features,
        ex_probability
    ):

        # ----------------------------------------------------
        # Resize EX mask
        # ----------------------------------------------------

        ex_probability = F.interpolate(
            ex_probability,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # ----------------------------------------------------
        # Image features
        # ----------------------------------------------------

        image_features = (
            self.feature_projection(
                features
            )
        )

        # ----------------------------------------------------
        # EX embedding
        # ----------------------------------------------------

        ex_features = (
            self.ex_embedding(
                ex_probability
            )
        )

        # ----------------------------------------------------
        # Bounded EX gate
        # ----------------------------------------------------

        gate = self.ex_gate(
            ex_probability
        )

        # ----------------------------------------------------
        # Guided features
        # ----------------------------------------------------

        guided_features = (
            image_features
            *
            (1.0 + gate)
            +
            ex_features
        )

        # ----------------------------------------------------
        # Flatten into tokens
        # ----------------------------------------------------

        B, C, H, W = (
            guided_features.shape
        )

        tokens = (
            guided_features
            .flatten(2)
            .transpose(1, 2)
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        tokens = self.transformer(
            tokens
        )

        # ----------------------------------------------------
        # GLOBAL POOLING
        # ----------------------------------------------------

        global_feature = (
            tokens.mean(dim=1)
        )

        # ----------------------------------------------------
        # EX-FOCUSED POOLING
        # ----------------------------------------------------

        ex_weights = (
            ex_probability.flatten(1)
        )

        ex_weights = (
            ex_weights
            /
            (
                ex_weights.sum(
                    dim=1,
                    keepdim=True
                )
                +
                1e-6
            )
        )

        ex_feature = torch.bmm(
            ex_weights.unsqueeze(1),
            tokens
        ).squeeze(1)

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        fused = torch.cat(
            [
                global_feature,
                ex_feature
            ],
            dim=1
        )

        fused = self.fusion(
            fused
        )

        # ----------------------------------------------------
        # ORDINAL OUTPUT
        # ----------------------------------------------------

        cls_out = self.classifier(
            fused
        )

        return cls_out


# ============================================================
# JOINT MODEL
# ============================================================

class JointSegClsNet(nn.Module):

    def __init__(
        self,
        num_lesion_classes=2,
        num_dr_classes=4,
        pretrained=True
    ):

        super().__init__()

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder = SwinEncoder(
            pretrained=pretrained
        )

        # ----------------------------------------------------
        # SE blocks
        # ----------------------------------------------------

        self.se_stem = SEBlock(96)

        self.se1 = SEBlock(96)

        self.se2 = SEBlock(192)

        self.se3 = SEBlock(384)

        self.se4 = SEBlock(768)

        # ----------------------------------------------------
        # Decoder
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
        # Segmentation head
        # ----------------------------------------------------

        self.seg_head = nn.Sequential(

            nn.Conv2d(
                48,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                32,
                num_lesion_classes,
                kernel_size=1
            )
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        self.cls_head = (
            ExudateGuidedTransformer(
                feature_dim=768,
                num_heads=8,
                num_layers=2,
                num_classes=num_dr_classes
            )
        )

    def forward(self, x):

        # ====================================================
        # ENCODER
        # ====================================================

        s0, e1, e2, e3, e4 = (
            self.encoder(x)
        )

        s0 = self.se_stem(s0)

        e1 = self.se1(e1)

        e2 = self.se2(e2)

        e3 = self.se3(e3)

        e4 = self.se4(e4)

        # ====================================================
        # DECODER
        # ====================================================

        d4 = self.up4(
            e4,
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
        # SEGMENTATION
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
        # PREDICTED MASK
        # ====================================================

        lesion_prob = torch.sigmoid(
            seg_out
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Only EX guides classification.
        #
        # Channel 0 = EX
        # Channel 1 = SE
        # ----------------------------------------------------

        ex_probability = (
            lesion_prob[:, 0:1]
        )

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        cls_out = self.cls_head(
            e4,
            ex_probability
        )

        return (
            seg_out,
            cls_out
        )