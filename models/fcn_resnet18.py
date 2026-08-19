import torch
from torch import nn
from torch.nn import functional as F

try:
    from torchvision.models import resnet18, ResNet18_Weights
except ImportError:
    from torchvision.models import resnet18
    ResNet18_Weights = None


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    center = factor - 1 if kernel_size % 2 == 1 else factor - 0.5
    og0 = torch.arange(kernel_size).reshape(-1, 1)
    og1 = torch.arange(kernel_size).reshape(1, -1)
    filt = (1 - torch.abs(og0 - center) / factor) * (1 - torch.abs(og1 - center) / factor)
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    for i in range(min(in_channels, out_channels)):
        weight[i, i] = filt
    return weight


class FCNResNet18(nn.Module):
    """D2L-style FCN-32s: ResNet18 feature extractor + 1x1 conv + x32 transposed conv."""
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        if ResNet18_Weights is not None:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
        else:
            base = resnet18(pretrained=pretrained)

        # Remove avgpool and fc; output stride is 32, channels=512.
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)
        self.upsample = nn.ConvTranspose2d(
            num_classes, num_classes,
            kernel_size=64, stride=32, padding=16,
            bias=False,
        )
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
        with torch.no_grad():
            self.upsample.weight.copy_(bilinear_kernel(num_classes, num_classes, 64))

    def forward(self, x):
        h, w = x.shape[-2:]
        feat = self.backbone(x)          # [B,512,H/32,W/32] for divisible sizes
        score = self.classifier(feat)    # [B,7,H/32,W/32]
        out = self.upsample(score)       # [B,7,H,W] for H,W divisible by 32
        if out.shape[-2:] != (h, w):
            out = F.interpolate(out, size=(h,w), mode='bilinear', align_corners=False)
        return out
