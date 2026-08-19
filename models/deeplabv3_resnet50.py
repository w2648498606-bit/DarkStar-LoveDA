import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabV3ResNet50(nn.Module):
    """Torchvision DeepLabV3 with an ImageNet-pretrained ResNet50 backbone.

    For small-GPU training with batch_size=1, BatchNorm is frozen in eval mode.
    This avoids the ASPP global-pooling branch failing on a [1,C,1,1] tensor and
    is also a common choice when segmentation batch sizes are very small.
    """

    def __init__(self, num_classes=7, pretrained=True, freeze_bn=True):
        super().__init__()
        backbone_weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = deeplabv3_resnet50(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
            aux_loss=False,
        )
        self.freeze_bn = freeze_bn
        if self.freeze_bn:
            self._freeze_bn_stats()

    def _freeze_bn_stats(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_bn:
            self._freeze_bn_stats()
        return self

    def forward(self, x):
        h, w = x.shape[-2:]
        out = self.net(x)["out"]
        if out.shape[-2:] != (h, w):
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out
