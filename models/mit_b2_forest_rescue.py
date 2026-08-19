"""
E041: MiT-B2 + UPerNet + Forest/Background Rescue Head.

Only one lightweight residual head is added on top of the normal
UPerNet fused feature.

The rescue head predicts a bounded correction delta:

    bg_logit     <- bg_logit     - scale * tanh(delta)
    forest_logit <- forest_logit + scale * tanh(delta)

All other class logits remain unchanged.

LoveDA train IDs:
    0 background
    1 building
    2 road
    3 water
    4 barren
    5 forest
    6 agriculture
"""

import torch
from torch import nn
from torch.nn import functional as F

from .mit_b2_structures import MiTB2UPerNet


class MiTB2ForestRescueUPerNet(MiTB2UPerNet):
    def __init__(
        self,
        num_classes=7,
        pretrained=True,
        fpn_dim=256,
        rescue_scale=1.0,
    ):
        if int(num_classes) != 7:
            raise ValueError(
                "Forest Rescue currently expects LoveDA's 7 classes."
            )

        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            fpn_dim=fpn_dim,
        )

        self.num_classes = int(num_classes)
        self.rescue_scale = float(rescue_scale)

        # Find the existing final 7-class classifier so that the
        # rescue head automatically uses the same fused feature channels.
        final_classifier = None

        if isinstance(self.head, nn.Conv2d):
            if self.head.out_channels == self.num_classes:
                final_classifier = self.head

        if final_classifier is None:
            for module in self.head.modules():
                if (
                    isinstance(module, nn.Conv2d)
                    and module.out_channels == self.num_classes
                ):
                    final_classifier = module

        if final_classifier is None:
            raise RuntimeError(
                "Could not locate the final 7-class Conv2d "
                "inside the UPerNet head."
            )

        feature_channels = int(final_classifier.in_channels)

        # Extremely small targeted structural branch.
        self.forest_rescue = nn.Conv2d(
            feature_channels,
            1,
            kernel_size=1,
            bias=True,
        )

        # Critical:
        # zero initialization makes E041 initially identical to the
        # baseline classifier. The rescue branch learns only if the
        # segmentation objective finds a useful correction.
        nn.init.zeros_(self.forest_rescue.weight)
        nn.init.zeros_(self.forest_rescue.bias)

    def forward(self, x):
        # Same backbone and decoder as E037.
        features = self.backbone(x)
        fused = self.decoder(features)

        # Original 7-class logits.
        logits = self.head(fused)

        # Targeted forest/background correction.
        delta = self.forest_rescue(fused)
        delta = torch.tanh(delta) * self.rescue_scale

        # Avoid risky in-place manipulation of logits.
        bg = logits[:, 0:1] - delta
        middle = logits[:, 1:5]
        forest = logits[:, 5:6] + delta
        agriculture = logits[:, 6:7]

        logits = torch.cat(
            [
                bg,
                middle,
                forest,
                agriculture,
            ],
            dim=1,
        )

        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return logits
