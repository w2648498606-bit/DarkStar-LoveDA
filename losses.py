import torch
from torch import nn
from torch.nn import functional as F


class MulticlassDiceLoss(nn.Module):
    """Soft Dice loss for multi-class semantic segmentation with ignore pixels."""

    def __init__(self, num_classes=7, ignore_index=255, smooth=1.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.smooth = float(smooth)

    def forward(self, logits, target):
        probs = torch.softmax(logits.float(), dim=1)
        valid = target != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0

        losses = []
        valid_f = valid.float()
        for c in range(self.num_classes):
            p = probs[:, c] * valid_f
            t = ((target == c) & valid).float()
            intersection = torch.sum(p * t)
            denominator = torch.sum(p) + torch.sum(t)
            dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
            losses.append(1.0 - dice)
        return torch.stack(losses).mean()


class CEDiceLoss(nn.Module):
    """Weighted CE + Dice. dice_weight=0.3 means 0.7*CE + 0.3*Dice."""

    def __init__(self, num_classes=7, ignore_index=255, dice_weight=0.3):
        super().__init__()
        if not (0.0 <= dice_weight <= 1.0):
            raise ValueError("dice_weight must be in [0, 1]")
        self.dice_weight = float(dice_weight)
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = MulticlassDiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        dice = self.dice(logits, target)
        return (1.0 - self.dice_weight) * ce + self.dice_weight * dice



def _lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension with respect to sorted errors."""
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-12)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[:-1]
    return jaccard


def _flatten_probas(probas, labels, ignore_index=255):
    # [B,C,H,W] -> [P,C], [B,H,W] -> [P], dropping ignore pixels.
    if probas.dim() != 4:
        raise ValueError(f"Expected [B,C,H,W] probabilities, got {tuple(probas.shape)}")
    c = probas.size(1)
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, c)
    labels = labels.contiguous().view(-1)
    valid = labels != int(ignore_index)
    return probas[valid], labels[valid]


def lovasz_softmax_flat(probas, labels, classes="present"):
    """Multi-class Lovasz-Softmax loss on flattened probabilities."""
    if probas.numel() == 0:
        return probas.sum() * 0.0
    c = probas.size(1)
    losses = []
    class_ids = range(c) if classes in {"all", "present"} else classes
    for cls in class_ids:
        fg = (labels == int(cls)).float()
        if classes == "present" and fg.sum() == 0:
            continue
        class_pred = probas[:, int(cls)]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


class CELovaszLoss(nn.Module):
    """Cross-entropy + lambda * Lovasz-Softmax (IoU-oriented surrogate)."""

    def __init__(self, num_classes=7, ignore_index=255, lovasz_weight=0.5):
        super().__init__()
        if lovasz_weight < 0:
            raise ValueError("lovasz_weight must be >= 0")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.lovasz_weight = float(lovasz_weight)
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        probs = torch.softmax(logits.float(), dim=1)
        probs_flat, labels_flat = _flatten_probas(probs, target, self.ignore_index)
        lovasz = lovasz_softmax_flat(probs_flat, labels_flat, classes="present")
        return ce + self.lovasz_weight * lovasz

def build_loss(name="ce", num_classes=7, ignore_index=255, dice_weight=0.3, lovasz_weight=0.5):
    name = str(name).lower()
    if name == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    if name == "ce_dice":
        return CEDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            dice_weight=dice_weight,
        )
    if name == "ce_lovasz":
        return CELovaszLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            lovasz_weight=lovasz_weight,
        )
    raise ValueError(f"Unknown loss: {name}")
