import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    dims = (2, 3)
    intersection = (pred * target).sum(dim=dims)
    denominator = pred.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps))


def sigmoid_focal_loss_per_channel(logits, targets, alpha_per_channel, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_c = alpha_per_channel.view(1, -1, 1, 1)
    alpha_t = alpha_c * targets + (1.0 - alpha_c) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    return loss.mean(dim=(2, 3))


def weighted_segmentation_loss(seg_pred, seg_target, seg_focal_alpha, lesion_weights):
    """UNCHANGED from 7G-v2 -- SE weighted higher than EX."""
    dsc_per_channel = dice_loss(seg_pred, seg_target)
    focal_per_channel = sigmoid_focal_loss_per_channel(seg_pred, seg_target, seg_focal_alpha)
    weights = torch.tensor(lesion_weights, device=seg_pred.device, dtype=torch.float32)
    weights = weights / weights.sum() * len(lesion_weights)
    dsc_weighted = (dsc_per_channel.mean(dim=0) * weights).sum() / len(lesion_weights)
    focal_weighted = (focal_per_channel.mean(dim=0) * weights).sum() / len(lesion_weights)
    return dsc_weighted + focal_weighted


def build_classification_criterion(class_weights, label_smoothing=0.05):
    """Improvement 1 + 2 from faculty's feedback: mild class weighting
    (Grade 1 and Grade 4 upweighted slightly, since they're the weakest
    classes) combined with label smoothing (softens targets since
    adjacent DR grades visually overlap). Plain CE, NOT focal -- per
    faculty's exact recommended code."""
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
