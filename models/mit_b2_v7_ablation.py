"""v7 strict ablation models built as exact-identity extensions of E009 UPerNet.

Every variant preserves the E009 module names:
    backbone, decoder, head
so the complete E009 checkpoint can be loaded without remapping.

Each new adapter is attached as ``adapter`` and is initialized to contribute exactly
zero. Therefore, immediately after loading E009, the variant must produce numerically
identical logits to the E009 baseline in eval mode. This property is checked by
``v7_verify_init.py`` before any training sweep.

These are controlled mechanism probes, not claims of faithful reproduction of a
specific paper implementation.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .mit_b2_structures import MiTB2UPerNet


def _gn(channels: int):
    # GroupNorm avoids small-batch BatchNorm state drift in newly-added adapters.
    groups = 32
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


def _zero_conv(in_ch: int, out_ch: int, kernel_size=1, padding=0):
    layer = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=True)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class _StrictAblationBase(MiTB2UPerNet):
    ablation_base_model = "mit_b2_upernet"
    ablation_new_prefixes = ("adapter.",)

    def ablation_new_modules(self):
        return (self.adapter,)

    def _base_feature(self, x):
        return self.decoder(self.backbone(x))

    def _finish(self, feat, x):
        logits = self.head(feat)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


class ResidualRFAdapter(nn.Module):
    """Multi-receptive-field residual probe with exact-zero initial contribution.

    It tests whether a learned mixture of local receptive fields helps beyond E009.
    It is intentionally described as an ablation adapter, not as FADC reproduction.
    """

    def __init__(self, channels=256, branch_dim=64, dilations=(1, 2, 4, 6)):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(channels, branch_dim, 1, bias=False),
            _gn(branch_dim),
            nn.GELU(),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(branch_dim, branch_dim, 3, padding=d, dilation=d, groups=branch_dim, bias=False),
                _gn(branch_dim),
                nn.GELU(),
            )
            for d in dilations
        ])
        self.gate = nn.Conv2d(branch_dim * len(dilations), len(dilations), 1)
        self.mix = nn.Sequential(
            nn.Conv2d(branch_dim, channels, 1, bias=False),
            _gn(channels),
            nn.GELU(),
        )
        self.out = _zero_conv(channels, channels, 1)

    def forward(self, x):
        z = self.pre(x)
        branches = [b(z) for b in self.branches]
        gate_input = torch.cat(branches, dim=1)
        weights = F.softmax(self.gate(gate_input), dim=1)
        mixed = torch.zeros_like(branches[0])
        for i, branch in enumerate(branches):
            mixed = mixed + weights[:, i:i+1] * branch
        delta = self.out(self.mix(mixed))
        return x + delta


class ResidualFrequencyAdapter(nn.Module):
    """High/low-frequency residual probe with spatially varying fusion.

    Low frequency is an anti-aliased local mean; high frequency is the residual.
    A learned spatial gate chooses the contribution before a zero-initialized output
    projection, so the complete model is an exact E009 identity at initialization.
    """

    def __init__(self, channels=256):
        super().__init__()
        self.low_path = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            _gn(channels),
            nn.GELU(),
        )
        self.high_path = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            _gn(channels),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1, bias=True),
        )
        self.out = _zero_conv(channels, channels, 1)

    def forward(self, x):
        low0 = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        high0 = x - low0
        low = self.low_path(low0)
        high = self.high_path(high0)
        g = torch.sigmoid(self.gate(torch.cat([low, high], dim=1)))
        delta = self.out(g * high + (1.0 - g) * low)
        return x + delta


class ResidualLocalGlobalAdapter(nn.Module):
    """Adaptive local/global residual probe.

    The local path preserves fine structures; the global path pools a wider context.
    The gate is spatially varying. The final projection is zero-initialized to guarantee
    exact parity with E009 before fine-tuning.
    """

    def __init__(self, channels=256):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            _gn(channels),
            nn.GELU(),
        )
        self.global_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            _gn(channels),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1, bias=True),
        )
        self.out = _zero_conv(channels, channels, 1)

    def forward(self, x):
        local = self.local(x)
        # 4x coarser contextual view, then return to UPerNet 1/4 scale.
        h, w = x.shape[-2:]
        ph, pw = max(h // 4, 1), max(w // 4, 1)
        global_feat = F.adaptive_avg_pool2d(x, (ph, pw))
        global_feat = self.global_proj(global_feat)
        global_feat = F.interpolate(global_feat, size=(h, w), mode="bilinear", align_corners=False)
        g = torch.sigmoid(self.gate(torch.cat([local, global_feat], dim=1)))
        delta = self.out(g * local + (1.0 - g) * global_feat)
        return x + delta


class MiTB2UPerNetV7RF(_StrictAblationBase):
    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained, fpn_dim=fpn_dim)
        self.adapter = ResidualRFAdapter(fpn_dim)

    def forward(self, x):
        return self._finish(self.adapter(self._base_feature(x)), x)


class MiTB2UPerNetV7Frequency(_StrictAblationBase):
    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained, fpn_dim=fpn_dim)
        self.adapter = ResidualFrequencyAdapter(fpn_dim)

    def forward(self, x):
        return self._finish(self.adapter(self._base_feature(x)), x)


class MiTB2UPerNetV7LocalGlobal(_StrictAblationBase):
    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained, fpn_dim=fpn_dim)
        self.adapter = ResidualLocalGlobalAdapter(fpn_dim)

    def forward(self, x):
        return self._finish(self.adapter(self._base_feature(x)), x)
