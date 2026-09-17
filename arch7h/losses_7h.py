import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
    logits,
    target,
    eps=1e-6
):

    prob = torch.sigmoid(logits)

    dims = (
        0,
        2,
        3
    )

    intersection = (
        prob * target
    ).sum(dims)

    denominator = (
        prob.sum(dims)
        +
        target.sum(dims)
    )

    dice = (
        2.0 * intersection + eps
    ) / (
        denominator + eps
    )

    return 1.0 - dice.mean()


# ============================================================
# FOCAL LOSS
# ============================================================

def focal_loss(
    logits,
    target,
    alpha=None,
    gamma=2.0
):

    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none"
    )

    prob = torch.sigmoid(
        logits
    )

    pt = (
        prob * target
        +
        (1.0 - prob) * (1.0 - target)
    )

    focal = (
        1.0 - pt
    ).pow(gamma) * bce

    if alpha is not None:

        alpha = alpha.view(
            1,
            -1,
            1,
            1
        ).to(logits.device)

        alpha_factor = (
            alpha * target
            +
            (1.0 - alpha) * (1.0 - target)
        )

        focal = focal * alpha_factor

    return focal.mean()


# ============================================================
# SEGMENTATION LOSS
# ============================================================

def segmentation_loss(
    logits,
    target,
    alpha=None
):

    d_loss = dice_loss(
        logits,
        target
    )

    f_loss = focal_loss(
        logits,
        target,
        alpha=alpha,
        gamma=2.0
    )

    return (
        0.65 * d_loss
        +
        0.35 * f_loss
    )


# ============================================================
# CLASSIFICATION FOCAL LOSS
# ============================================================

def classification_focal_loss(
    logits,
    target,
    class_weights=None,
    gamma=1.5
):

    ce = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        reduction="none"
    )

    pt = torch.exp(
        -ce
    )

    loss = (
        1.0 - pt
    ).pow(gamma) * ce

    return loss.mean()


# ============================================================
# LABEL-SMOOTHED CE
# ============================================================

def label_smoothing_loss(
    logits,
    target,
    class_weights=None,
    smoothing=0.05
):

    return F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        label_smoothing=smoothing
    )


# ============================================================
# HYBRID CLASSIFICATION LOSS
# ============================================================

def hybrid_classification_loss(
    logits,
    target,
    class_weights=None
):

    focal = classification_focal_loss(
        logits,
        target,
        class_weights=class_weights,
        gamma=1.5
    )

    smooth_ce = label_smoothing_loss(
        logits,
        target,
        class_weights=class_weights,
        smoothing=0.05
    )

    return (
        0.65 * focal
        +
        0.35 * smooth_ce
    )