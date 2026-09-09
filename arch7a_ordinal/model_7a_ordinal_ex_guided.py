"""
ARCHITECTURE 7A - Improved 3B-Transformer

Pretrained Swin Transformer-Tiny Encoder
        +
SE Blocks
        +
Attention U-Net Decoder
        +
EX-only Segmentation
        +
Predicted EX Mask
        +
EX-Guided Transformer
        +
Ordinal DR Classification

Input:
    [B, 3, 512, 512]

Segmentation output:
    [B, 1, 512, 512]

Classification output:
    [B, 4]

The four classification logits represent:

    P(DR > 0)
    P(DR > 1)
    P(DR > 2)
    P(DR > 3)
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

        self.pool = nn.AdaptiveAvgPool2d(1)

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

        s = self.relu(
            self.fc1(s)
        )

        s = self.sigmoid(
            self.fc2(s)
        )

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
            kernel_size=1,
            bias=False
        )

        self.phi_g = nn.Conv2d(
            gate_channels,
            inter_channels,
            kernel_size=1,
            bias=False
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

        if (
            theta_x.shape[-2:]
            !=
            phi_g.shape[-2:]
        ):

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

        # ----------------------------------------------------
        # Residual projection
        # ----------------------------------------------------

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

        if (
            x.shape[-2:]
            !=
            skip.shape[-2:]
        ):

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
            [
                x,
                attn_skip
            ],
            dim=1
        )

        return self.conv(x)


# ============================================================
# SWIN TRANSFORMER ENCODER
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

    def forward(
        self,
        x
    ):

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
# EX-GUIDED TRANSFORMER CLASSIFIER
# ============================================================

class ExudateGuidedTransformer(
    nn.Module
):

    def __init__(
        self,
        feature_dim=768,
        num_heads=8,
        num_layers=2,
        num_classes=4
    ):

        super().__init__()

        self.feature_dim = feature_dim

        # ----------------------------------------------------
        # Image feature projection
        # ----------------------------------------------------

        self.image_projection = nn.Sequential(

            nn.Conv2d(
                feature_dim,
                feature_dim,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                feature_dim
            ),

            nn.GELU()
        )

        # ----------------------------------------------------
        # EX mask embedding
        # ----------------------------------------------------

        self.exudate_embedding = nn.Sequential(

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

            nn.GELU(),

            nn.Conv2d(
                128,
                feature_dim,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                feature_dim
            ),

            nn.GELU()
        )

        # ----------------------------------------------------
        # Bounded EX attention gate
        # ----------------------------------------------------

        self.exudate_gate = nn.Sequential(

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

            nn.GELU(),

            nn.Conv2d(
                64,
                feature_dim,
                kernel_size=1
            ),

            nn.Sigmoid()
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
        # Classification head
        #
        # 4 outputs because DR is ordinal:
        #
        # >0
        # >1
        # >2
        # >3
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.LayerNorm(
                feature_dim
            ),

            nn.Linear(
                feature_dim,
                256
            ),

            nn.GELU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(
        self,
        image_features,
        exudate_prob
    ):

        # ----------------------------------------------------
        # Resize EX mask to deep feature resolution
        # ----------------------------------------------------

        exudate_prob = F.interpolate(
            exudate_prob,
            size=image_features.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # ----------------------------------------------------
        # Project image features
        # ----------------------------------------------------

        image_features = self.image_projection(
            image_features
        )

        # ----------------------------------------------------
        # Encode EX mask
        # ----------------------------------------------------

        exudate_features = self.exudate_embedding(
            exudate_prob
        )

        # ----------------------------------------------------
        # Bounded EX attention
        # ----------------------------------------------------

        gate = self.exudate_gate(
            exudate_prob
        )

        # ----------------------------------------------------
        # EX-guided image representation
        #
        # First:
        #     image × (1 + gate)
        #
        # Then:
        #     add EX semantic features
        # ----------------------------------------------------

        guided_features = (
            image_features
            *
            (1.0 + gate)
        )

        guided_features = (
            guided_features
            +
            exudate_features
        )

        # ----------------------------------------------------
        # Convert feature map into tokens
        #
        # [B,C,H,W]
        #
        # ->
        #
        # [B,H*W,C]
        # ----------------------------------------------------

        B, C, H, W = (
            guided_features.shape
        )

        tokens = (
            guided_features
            .flatten(2)
            .transpose(1, 2)
            .contiguous()
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        tokens = self.transformer(
            tokens
        )

        # ----------------------------------------------------
        # Global token pooling
        # ----------------------------------------------------

        pooled = tokens.mean(
            dim=1
        )

        # ----------------------------------------------------
        # Ordinal classification
        # ----------------------------------------------------

        logits = self.classifier(
            pooled
        )

        return logits


# ============================================================
# COMPLETE JOINT MODEL
# ============================================================

class JointSegClsNet(nn.Module):

    def __init__(
        self,
        num_lesion_classes=1,
        num_dr_classes=4,
        pretrained=True
    ):

        super().__init__()

        # ====================================================
        # ENCODER
        # ====================================================

        self.encoder = SwinEncoder(
            pretrained=pretrained
        )

        # ====================================================
        # SE BLOCKS
        # ====================================================

        self.se_stem = SEBlock(
            96
        )

        self.se1 = SEBlock(
            96
        )

        self.se2 = SEBlock(
            192
        )

        self.se3 = SEBlock(
            384
        )

        self.se4 = SEBlock(
            768
        )

        # ====================================================
        # DECODER
        # ====================================================

        self.up4 = UpBlock(
            in_ch=768,
            skip_ch=384,
            out_ch=384
        )

        self.up3 = UpBlock(
            in_ch=384,
            skip_ch=192,
            out_ch=192
        )

        self.up2 = UpBlock(
            in_ch=192,
            skip_ch=96,
            out_ch=96
        )

        self.up1 = UpBlock(
            in_ch=96,
            skip_ch=96,
            out_ch=48
        )

        # ====================================================
        # EX SEGMENTATION HEAD
        # ====================================================

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

        # ====================================================
        # EX-GUIDED TRANSFORMER
        # ====================================================

        self.cls_head = (
            ExudateGuidedTransformer(
                feature_dim=768,
                num_heads=8,
                num_layers=2,
                num_classes=num_dr_classes
            )
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x
    ):

        # ====================================================
        # ENCODER
        # ====================================================

        (
            s0,
            e1,
            e2,
            e3,
            e4
        ) = self.encoder(x)

        # ====================================================
        # SE ATTENTION
        # ====================================================

        s0 = self.se_stem(
            s0
        )

        e1 = self.se1(
            e1
        )

        e2 = self.se2(
            e2
        )

        e3 = self.se3(
            e3
        )

        e4 = self.se4(
            e4
        )

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
        # EX SEGMENTATION
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
        # PREDICTED EX MASK
        # ====================================================

        exudate_prob = torch.sigmoid(
            seg_out
        )

        # ====================================================
        # EX-GUIDED TRANSFORMER CLASSIFICATION
        # ====================================================

        cls_out = self.cls_head(
            e4,
            exudate_prob
        )

        # ====================================================
        # RETURN
        # ====================================================

        return (
            seg_out,
            cls_out
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = JointSegClsNet(
        num_lesion_classes=1,
        num_dr_classes=4,
        pretrained=False
    ).to(device)

    x = torch.randn(
        2,
        3,
        512,
        512
    ).to(device)

    with torch.no_grad():

        seg_out, cls_out = model(x)

    print(
        "Input:",
        x.shape
    )

    print(
        "Segmentation:",
        seg_out.shape
    )

    print(
        "Classification:",
        cls_out.shape
    )