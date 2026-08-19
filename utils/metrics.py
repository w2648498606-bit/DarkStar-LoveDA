import numpy as np
import torch


class SegmentationMetrics:
    def __init__(self, num_classes=7, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.detach().to('cpu').long().reshape(-1)
        target = target.detach().to('cpu').long().reshape(-1)
        valid = target != self.ignore_index
        pred, target = pred[valid], target[valid]
        valid = (target >= 0) & (target < self.num_classes) & (pred >= 0) & (pred < self.num_classes)
        pred, target = pred[valid], target[valid]
        if target.numel() == 0:
            return
        idx = target * self.num_classes + pred
        hist = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion += hist.reshape(self.num_classes, self.num_classes)

    def compute(self):
        cm = self.confusion.double()
        tp = torch.diag(cm)
        gt = cm.sum(dim=1)
        pd = cm.sum(dim=0)
        union = gt + pd - tp
        iou = torch.where(union > 0, tp / union, torch.nan)
        miou = torch.nanmean(iou).item()
        total = cm.sum()
        pixel_acc = (tp.sum() / total).item() if total > 0 else float('nan')
        return {
            'iou': iou.numpy(),
            'miou': miou,
            'pixel_acc': pixel_acc,
            'confusion': self.confusion.numpy(),
        }
