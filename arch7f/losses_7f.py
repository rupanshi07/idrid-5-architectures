import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, eps=1e-6):
    """Per-image, per-channel Dice loss."""
    pred = torch.sigmoid(pred)
    dims = (2, 3)
    intersection = (pred * target).sum(dim=dims)
    denominator = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return (1.0 - dice).mean()


def focal_tversky_loss(pred, target, alpha=0.3, beta=0.7, gamma=1.33, eps=1e-6):
    """Focal-Tversky loss. beta > alpha specifically emphasizes False
    Negatives over False Positives -- important for small lesions
    (missing a lesion entirely is worse than a slightly-oversized
    prediction). gamma > 1 additionally focuses learning on harder,
    lower-Tversky-score cases, similar in spirit to focal loss."""
    pred_s = torch.sigmoid(pred)
    dims = (2, 3)
    tp = (pred_s * target).sum(dim=dims)
    fp = (pred_s * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - pred_s) * target).sum(dim=dims)
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    loss = (1.0 - tversky).pow(1.0 / gamma)
    return loss.mean()


def segmentation_loss(pred, target, dice_weight=1.0, tversky_weight=1.0):
    return dice_weight * dice_loss(pred, target) + tversky_weight * focal_tversky_loss(pred, target)


def hybrid_classification_loss(logits, targets, class_weights=None, ce_weight=0.8, ordinal_weight=0.2):
    """CE (respects standard 5-way classification) + a small ordinal
    regression auxiliary term (respects that DR grades are ORDERED,
    so predicting grade 0 instead of grade 4 should be penalized more
    than predicting grade 0 instead of grade 1). The ordinal term
    computes the softmax distribution's EXPECTED grade value and
    penalizes its squared distance from the true grade -- this doesn't
    require a separate output head, just regularizes the existing
    5-way softmax to respect ordinal structure."""
    ce = F.cross_entropy(logits, targets, weight=class_weights)

    probs = torch.softmax(logits, dim=1)
    grade_values = torch.arange(logits.shape[1], device=logits.device, dtype=torch.float32)
    expected_grade = (probs * grade_values.unsqueeze(0)).sum(dim=1)
    ordinal_loss = F.mse_loss(expected_grade, targets.float())

    return ce_weight * ce + ordinal_weight * ordinal_loss
