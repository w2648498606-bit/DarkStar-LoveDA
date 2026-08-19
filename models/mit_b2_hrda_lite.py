"""
E042: supervised HRDA-inspired MiT-B2 + UPerNet.

Core ideas retained from HRDA:
  - shared network for LR context and HR detail
  - larger-FOV low-resolution context
  - high-resolution detail
  - class-wise learned scale attention
  - attention predicted from context
  - detail auxiliary supervision

This is HRDA-lite, NOT a full reproduction of HRDA UDA.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .mit_b2_structures import MiTB2UPerNet


class MiTB2HRDALiteUPerNet(MiTB2UPerNet):

    uses_context_view = True
    uses_context_box = True

    # HRDA paper uses lambda_d = 0.1.
    hrda_detail_loss_weight = 0.1

    def __init__(
        self,
        num_classes=7,
        pretrained=True,
        fpn_dim=256,
    ):
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            fpn_dim=fpn_dim,
        )

        # Class-wise scale attention.
        #
        # 0 -> trust LR context
        # 1 -> trust HR detail
        self.scale_attention = nn.Conv2d(
            fpn_dim,
            num_classes,
            kernel_size=1,
            bias=True,
        )

        # Neutral initialization:
        # sigmoid(0) = 0.5
        nn.init.zeros_(
            self.scale_attention.weight
        )
        nn.init.zeros_(
            self.scale_attention.bias
        )

    def _branch(self, x):
        feat = self.decoder(
            self.backbone(x)
        )

        logits = self.head(feat)

        logits = self._upsample_logits(
            logits,
            x,
        )

        return feat, logits

    @staticmethod
    def _align_context(
        tensor,
        boxes,
        output_hw,
    ):
        """
        Crop the detail-aligned region from the LR context
        tensor and resize it to the HR detail resolution.

        boxes are expressed in context-input pixel coordinates.
        """

        if boxes is None:
            return F.interpolate(
                tensor,
                size=output_hw,
                mode="bilinear",
                align_corners=False,
            )

        b, c, h, w = tensor.shape

        box_list = (
            boxes.detach()
            .cpu()
            .tolist()
        )

        aligned = []

        for i in range(b):
            x1, y1, x2, y2 = box_list[i]

            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))

            x2 = max(
                x1 + 1,
                min(w, x2),
            )

            y2 = max(
                y1 + 1,
                min(h, y2),
            )

            roi = tensor[
                i:i + 1,
                :,
                y1:y2,
                x1:x2,
            ]

            roi = F.interpolate(
                roi,
                size=output_hw,
                mode="bilinear",
                align_corners=False,
            )

            aligned.append(roi)

        return torch.cat(
            aligned,
            dim=0,
        )

    def forward(
        self,
        detail_image,
        context_image=None,
        context_box=None,
    ):
        # Compatibility fallback.
        if context_image is None:
            return super().forward(
                detail_image
            )

        # During training both tensors are normally 512x512.
        # Concatenate them so the shared backbone/decoder sees both
        # resolutions in the same batch.
        if (
            detail_image.shape[-2:]
            == context_image.shape[-2:]
        ):
            b = detail_image.shape[0]

            both = torch.cat(
                [
                    context_image,
                    detail_image,
                ],
                dim=0,
            )

            feat = self.decoder(
                self.backbone(both)
            )

            logits = self.head(feat)

            context_feat = feat[:b]
            detail_feat = feat[b:]

            context_logits = logits[:b]
            detail_logits = logits[b:]

            context_logits = F.interpolate(
                context_logits,
                size=context_image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            detail_logits = F.interpolate(
                detail_logits,
                size=detail_image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        else:
            # Validation: full-resolution detail + half-resolution
            # context generally have different tensor shapes.
            context_feat, context_logits = self._branch(
                context_image
            )

            detail_feat, detail_logits = self._branch(
                detail_image
            )

        # Attention is predicted from the large-FOV context,
        # as in HRDA.
        attention = torch.sigmoid(
            self.scale_attention(
                context_feat
            )
        )

        attention = F.interpolate(
            attention,
            size=context_image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        context_logits_aligned = (
            self._align_context(
                context_logits,
                context_box,
                detail_image.shape[-2:],
            )
        )

        attention_aligned = (
            self._align_context(
                attention,
                context_box,
                detail_image.shape[-2:],
            )
        )

        # HRDA-style fusion:
        #
        # attention=0 -> context
        # attention=1 -> detail
        fused_logits = (
            (1.0 - attention_aligned)
            * context_logits_aligned
            +
            attention_aligned
            * detail_logits
        )

        if self.training:
            return {
                "logits": fused_logits,
                "detail_logits": detail_logits,
            }

        return fused_logits
