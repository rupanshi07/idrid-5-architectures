"""
ARCHITECTURE 7G -- Experiment 1 per faculty's ordered plan.
Everything else held IDENTICAL to 7E (same loss, same freeze schedule,
same sampler, same Swin-T encoder) -- the ONLY change is:
  - Bottleneck Transformer gets proper 2D positional embeddings
    (not flat 1D) -- so it knows WHERE each token came from
  - EX and SE probabilities get SEPARATE feature extractors (not
    combined into one "exudate" signal)
  - Classification uses CONCATENATED fusion (Transformer + Global +
    EX + SE features), never multiplicative gating that can destroy
    useful image features if segmentation is wrong
  - Attention pooling (learned) replaces mean pooling
  - Soft (sigmoid) probabilities only -- never hard-thresholded
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
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
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


class BottleneckTransformer(nn.Module):
    """Uses a genuine 2D positional embedding (separate learnable
    embeddings for row and column, summed per token) instead of a flat
    1D sequence embedding -- so the Transformer knows each token's
    actual (row, col) location in the retina, not just its position
    in a flattened list."""
    def __init__(self, channels=768, grid_size=16, num_layers=4, num_heads=8, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.grid_size = grid_size
        self.row_embed = nn.Parameter(torch.zeros(grid_size, channels // 2))
        self.col_embed = nn.Parameter(torch.zeros(grid_size, channels // 2))
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=num_heads, dim_feedforward=channels * mlp_ratio,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # Build the 2D positional grid: (H, W, C) via broadcasting row/col embeddings
        row_pe = self.row_embed[:H].unsqueeze(1).expand(H, W, -1)   # (H, W, C/2)
        col_pe = self.col_embed[:W].unsqueeze(0).expand(H, W, -1)   # (H, W, C/2)
        pos_2d = torch.cat([row_pe, col_pe], dim=-1)                 # (H, W, C)
        pos_2d = pos_2d.reshape(1, H * W, C)

        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        tokens = tokens + pos_2d
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class LesionFeatureExtractor(nn.Module):
    """Separate feature extractor for ONE lesion type's soft probability
    map. EX and SE each get their own instance -- they are NOT combined
    into a single 'exudate' signal, per faculty's point #7."""
    def __init__(self, out_channels=768, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, prob_map):
        return self.net(prob_map)  # (B, out_channels)


class AttentionPoolingHead(nn.Module):
    """Learned attention pooling over Transformer tokens, replacing
    simple mean pooling."""
    def __init__(self, channels=768):
        super().__init__()
        self.attn_score = nn.Sequential(nn.Linear(channels, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, tokens):  # tokens: (B, N, C)
        scores = self.attn_score(tokens)
        weights = torch.softmax(scores, dim=1)
        return (tokens * weights).sum(dim=1)


class FusionClassifier(nn.Module):
    """CONCATENATES: Transformer-pooled feature + EX feature + SE feature
    (Global feature is the same as the Transformer-pooled feature here,
    since the bottleneck transformer already summarizes the whole image --
    kept as a separate named term for clarity per faculty's diagram).
    Never multiplies image features by a gate -- concatenation means a
    bad segmentation prediction can't erase useful image information,
    it just contributes a less-informative extra input."""
    def __init__(self, channels=768, num_classes=5):
        super().__init__()
        fusion_dim = channels * 3  # transformer-pooled + EX + SE
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, transformer_feat, ex_feat, se_feat):
        fused = torch.cat([transformer_feat, ex_feat, se_feat], dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)


class JointSegClsNet(nn.Module):
    def __init__(self, num_lesion_classes=2, num_dr_classes=5, pretrained=True):
        super().__init__()
        self.encoder = SwinEncoder(pretrained=pretrained)

        self.se_stem = SEBlock(96)
        self.se1 = SEBlock(96)
        self.se2 = SEBlock(192)
        self.se3 = SEBlock(384)
        self.se4 = SEBlock(768)

        self.bottleneck_transformer = BottleneckTransformer(channels=768, grid_size=16, num_layers=4, num_heads=8)
        self.attn_pool = AttentionPoolingHead(channels=768)

        self.up4 = UpBlock(in_ch=768, skip_ch=384, out_ch=384)
        self.up3 = UpBlock(in_ch=384, skip_ch=192, out_ch=192)
        self.up2 = UpBlock(in_ch=192, skip_ch=96, out_ch=96)
        self.up1 = UpBlock(in_ch=96, skip_ch=96, out_ch=48)

        self.seg_head = nn.Conv2d(48, num_lesion_classes, kernel_size=1)
        self.aux_seg_head_4 = nn.Conv2d(384, num_lesion_classes, kernel_size=1)
        self.aux_seg_head_3 = nn.Conv2d(192, num_lesion_classes, kernel_size=1)

        self.ex_extractor = LesionFeatureExtractor(out_channels=768)
        self.se_extractor = LesionFeatureExtractor(out_channels=768)
        self.cls_head = FusionClassifier(channels=768, num_classes=num_dr_classes)

    def set_encoder_trainable(self, trainable: bool):
        for p in self.encoder.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        s0, e1, e2, e3, e4 = self.encoder(x)
        s0 = self.se_stem(s0); e1 = self.se1(e1); e2 = self.se2(e2)
        e3 = self.se3(e3); e4 = self.se4(e4)

        e4_context = self.bottleneck_transformer(e4)

        d4 = self.up4(e4_context, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d1 = self.up1(d2, s0)

        seg_out = self.seg_head(d1)
        seg_out = F.interpolate(seg_out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        aux_out_4 = F.interpolate(self.aux_seg_head_4(d4), size=x.shape[-2:], mode="bilinear", align_corners=False)
        aux_out_3 = F.interpolate(self.aux_seg_head_3(d3), size=x.shape[-2:], mode="bilinear", align_corners=False)

        # Soft probabilities only -- never thresholded
        seg_prob = torch.sigmoid(seg_out)
        ex_prob = seg_prob[:, 0:1]
        se_prob = seg_prob[:, 1:2]

        B, C, H, W = e4_context.shape
        tokens = e4_context.flatten(2).transpose(1, 2)
        transformer_feat = self.attn_pool(tokens)  # (B, 768) -- "Transformer feature" AND "Global feature"

        ex_feat = self.ex_extractor(ex_prob)
        se_feat = self.se_extractor(se_prob)

        cls_out = self.cls_head(transformer_feat, ex_feat, se_feat)

        return seg_out, cls_out, [aux_out_4, aux_out_3]
