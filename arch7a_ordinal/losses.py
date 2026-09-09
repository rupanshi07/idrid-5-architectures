import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    dims = (1, 2, 3)
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return (1.0 - dice).mean()


def sigmoid_focal_loss(logits, targets, alpha_per_channel, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_c = alpha_per_channel.view(1, -1, 1, 1)
    alpha_t = alpha_c * targets + (1.0 - alpha_c) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    loss_per_channel = loss.mean(dim=(2, 3))
    return loss_per_channel.mean(dim=1)


def ordinal_targets(grades):
    return torch.stack([(grades > 0).float(), (grades > 1).float(),
                         (grades > 2).float(), (grades > 3).float()], dim=1)


def ordinal_focal_loss(logits, grades, gamma=2.0, pos_weight=None):
    targets = ordinal_targets(grades)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    probabilities = torch.sigmoid(logits)
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    focal = (1.0 - p_t).pow(gamma) * bce
    return focal.mean()


def ordinal_to_grade(logits, threshold=0.5):
    probabilities = torch.sigmoid(logits)
    return (probabilities > threshold).sum(dim=1).long()


class JointLoss(nn.Module):
    def __init__(self, seg_weight=2.5, cls_weight=1.5, dice_weight=1.0, focal_weight=1.0,
                 seg_focal_alpha=None, seg_focal_gamma=2.0, cls_focal_gamma=2.0, ordinal_pos_weight=None):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.seg_focal_alpha = seg_focal_alpha
        self.seg_focal_gamma = seg_focal_gamma
        self.cls_focal_gamma = cls_focal_gamma
        self.ordinal_pos_weight = ordinal_pos_weight

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
            valid_cls_idx = cls_valid.bool()
            valid_targets = cls_target[valid_cls_idx]
            valid_predictions = cls_pred[valid_cls_idx]
            cls_loss = ordinal_focal_loss(valid_predictions, valid_targets,
                                           gamma=self.cls_focal_gamma, pos_weight=self.ordinal_pos_weight)
        else:
            cls_loss = torch.tensor(0.0, device=cls_pred.device)

        total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss
        return total_loss, seg_loss.detach(), cls_loss.detach()
