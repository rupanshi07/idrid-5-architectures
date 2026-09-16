import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    dims = (0, 2, 3)
    intersection = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def sigmoid_focal_loss(logits, targets, alpha_per_channel, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_c = alpha_per_channel.view(1, -1, 1, 1)
    alpha_t = alpha_c * targets + (1 - alpha_c) * (1 - targets)
    loss = alpha_t * (1 - p_t).pow(gamma) * bce
    loss_per_channel = loss.mean(dim=(2, 3))
    loss_per_sample = loss_per_channel.mean(dim=1)
    return loss_per_sample


def multiclass_focal_loss(logits, targets, gamma=2.0, weight=None):
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    loss = (1 - pt).pow(gamma) * ce
    return loss


class JointLoss(nn.Module):
    def __init__(self, seg_weight=2.5, cls_weight=1.3,
                 dice_weight=1.0, focal_weight=1.0,
                 seg_focal_alpha=None, seg_focal_gamma=2.0,
                 cls_focal_gamma=2.0, class_weights=None):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.seg_focal_alpha = seg_focal_alpha
        self.seg_focal_gamma = seg_focal_gamma
        self.cls_focal_gamma = cls_focal_gamma
        self.class_weights = class_weights

    def forward(self, seg_pred, seg_target, seg_valid, cls_pred, cls_target, cls_valid):
        if seg_valid.sum() > 0:
            valid_idx = seg_valid.bool()
            focal = sigmoid_focal_loss(seg_pred, seg_target, self.seg_focal_alpha, gamma=self.seg_focal_gamma)
            focal = (focal * seg_valid).sum() / seg_valid.sum()
            dsc = dice_loss(seg_pred[valid_idx], seg_target[valid_idx])
            seg_loss = self.dice_weight * dsc + self.focal_weight * focal
        else:
            seg_loss = torch.tensor(0.0, device=seg_pred.device)

        if cls_valid.sum() > 0:
            safe_target = cls_target.clone()
            safe_target[safe_target < 0] = 0
            focal_cls = multiclass_focal_loss(cls_pred, safe_target, gamma=self.cls_focal_gamma, weight=self.class_weights)
            cls_loss = (focal_cls * cls_valid).sum() / cls_valid.sum()
        else:
            cls_loss = torch.tensor(0.0, device=cls_pred.device)

        total = self.seg_weight * seg_loss + self.cls_weight * cls_loss
        return total, seg_loss.detach(), cls_loss.detach()
