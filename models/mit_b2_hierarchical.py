"""
E040: Hierarchical BG/FG factorization on top of MiT-B2 + UPerNet.

The backbone, decoder and original 7-channel semantic head are unchanged.

Only the interpretation of the seven raw scores changes:

    raw[:, 0]   -> background gate
    raw[:, 1:7] -> conditional foreground class scores

Joint probabilities:

    P(bg) = sigmoid(bg_gate)

    P(class=c) =
        (1 - P(bg)) * softmax(fg_scores)[c-1]

for c = 1,...,6.

The forward method returns normalized log-probabilities [B,7,H,W].
They are directly compatible with the project's existing
CrossEntropyLoss and Lovasz-Softmax implementation.
"""

import math

import torch
from torch.nn import functional as F

from .mit_b2_structures import MiTB2UPerNet


class MiTB2HierUPerNet(MiTB2UPerNet):
    """
    MiT-B2 + UPerNet with hierarchical BG/FG -> foreground classification.

    No extra trainable parameters relative to MiTB2UPerNet.
    """

    def __init__(
        self,
        num_classes=7,
        pretrained=True,
        fpn_dim=256,
    ):
        if int(num_classes) != 7:
            raise ValueError(
                "E040 hierarchical head currently expects "
                "LoveDA's 7 training classes."
            )

        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            fpn_dim=fpn_dim,
        )

        self.num_classes = int(num_classes)

        # --------------------------------------------------------
        # Prior-neutral initialization.
        #
        # If all foreground conditional scores are initially equal,
        # we want:
        #
        #     P(bg) = 1/7
        #     P(each foreground class) = 1/7
        #
        # For sigmoid(g)=1/7:
        #
        #     g = log((1/7)/(6/7)) = -log(6)
        #
        # We therefore adjust only the background-gate bias if the
        # original final semantic Conv2d exposes a bias.
        # --------------------------------------------------------
        final_conv = None

        for module in self.head.modules():
            if (
                isinstance(module, torch.nn.Conv2d)
                and module.out_channels == self.num_classes
            ):
                final_conv = module

        if (
            final_conv is not None
            and final_conv.bias is not None
        ):
            with torch.no_grad():
                final_conv.bias[0] = -math.log(
                    self.num_classes - 1
                )

    @staticmethod
    def hierarchical_log_probs(raw):
        """
        raw: [B,7,H,W]

        channel 0:
            background gate logit

        channels 1..6:
            conditional foreground logits

        returns:
            normalized log P(class), shape [B,7,H,W]
        """

        if raw.ndim != 4 or raw.shape[1] != 7:
            raise ValueError(
                f"Expected raw [B,7,H,W], got {tuple(raw.shape)}"
            )

        # Work in FP32 for stability under AMP.
        x = raw.float()

        bg_gate = x[:, 0:1]
        fg_scores = x[:, 1:]

        # log P(background)
        log_p_bg = F.logsigmoid(bg_gate)

        # log P(foreground)
        log_p_fg = F.logsigmoid(-bg_gate)

        # log P(class | foreground)
        log_p_cond = F.log_softmax(
            fg_scores,
            dim=1,
        )

        # log P(foreground class)
        log_p_fg_classes = (
            log_p_fg
            + log_p_cond
        )

        return torch.cat(
            [
                log_p_bg,
                log_p_fg_classes,
            ],
            dim=1,
        )

    def forward(self, x):
        # Exactly the same backbone + UPerNet decoder as baseline.
        feat = self.decoder(
            self.backbone(x)
        )

        # Exactly the same original semantic head.
        raw = self.head(feat)

        # Match normal UPerNet output size first.
        if raw.shape[-2:] != x.shape[-2:]:
            raw = F.interpolate(
                raw,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Only this step differs from E037.
        return self.hierarchical_log_probs(raw)
