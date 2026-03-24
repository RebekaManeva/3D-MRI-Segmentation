import numpy as np
import torch
from medpy.metric.binary import hd95
from torch import nn

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)

        dice_per_class = []
        for c in range(pred.shape[1]):
            pred_c = pred[:, c].contiguous().view(-1)
            target_c = target[:, c].contiguous().view(-1)

            intersection = (pred_c * target_c).sum()
            dice = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
            dice_per_class.append(1 - dice)

        return torch.stack(dice_per_class).mean()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    dice_per_class = []
    for c in range(pred.shape[1]):
        pred_c = pred[:, c]
        target_c = target[:, c]
        intersection = (pred_c * target_c).sum()
        dice = (2 * intersection + eps) / (pred_c.sum() + target_c.sum() + eps)
        dice_per_class.append(dice)
    return torch.stack(dice_per_class).mean()

def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    iou_per_class = []
    for c in range(pred.shape[1]):
        pred_c = pred[:, c]
        target_c = target[:, c]
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        iou = (intersection + eps) / (union + eps)
        iou_per_class.append(iou)
    return torch.stack(iou_per_class).mean()


def precision_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    prec_per_class = []
    for c in range(pred.shape[1]):
        pred_c = pred[:, c]
        target_c = target[:, c]
        tp = (pred_c * target_c).sum()
        fp = (pred_c * (1 - target_c)).sum()
        prec = (tp + eps) / (tp + fp + eps)
        prec_per_class.append(prec)
    return torch.stack(prec_per_class).mean()


def recall_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    rec_per_class = []
    for c in range(pred.shape[1]):
        pred_c = pred[:, c]
        target_c = target[:, c]
        tp = (pred_c * target_c).sum()
        fn = ((1 - pred_c) * target_c).sum()
        rec = (tp + eps) / (tp + fn + eps)
        rec_per_class.append(rec)
    return torch.stack(rec_per_class).mean()


def f1_score(pred, target, eps=1e-6):
    p = precision_score(pred, target, eps)
    r = recall_score(pred, target, eps)
    return (2 * p * r + eps) / (p + r + eps)


def hd95_score(pred, target):
    pred = (pred > 0.5).cpu().numpy().astype(np.uint8)
    target = target.cpu().numpy().astype(np.uint8)

    if pred.ndim == 5:
        pred = pred[0, :].max(axis=0)
        target = target[0, :].max(axis=0)
    elif pred.ndim == 4:
        pred = pred[:].max(axis=0)
        target = target[:].max(axis=0)

    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    elif pred.sum() == 0 or target.sum() == 0:
        return np.nan

    try:
        return hd95(pred, target)
    except Exception as e:
        return np.nan

