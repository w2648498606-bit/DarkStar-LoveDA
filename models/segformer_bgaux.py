import torch
from torch import nn
from torch.nn import functional as F

from .segformer import SegFormerB2


def _group_count(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class SegFormerB2BGAux(SegFormerB2):
    """SegFormer-B2 with an auxiliary foreground/background branch.

    The 7-class SegFormer path is unchanged. During training, a lightweight
    auxiliary head receives the fused SegFormer decoder feature immediately
    before the original dropout/classifier and predicts foreground vs.
    background. At eval/inference time the model returns only the original
    7-class logits, so existing evaluation/prediction code remains compatible.

    This targets LoveDA's dominant error mode where true foreground pixels are
    absorbed into the background class, while preserving the ADE20K-pretrained
    SegFormer-B2 encoder and decoder.
    """

    def __init__(self, num_classes=7, pretrained=True):
        super().__init__(num_classes=num_classes, pretrained=pretrained)

        decode_head = self.net.decode_head
        if not hasattr(decode_head, "dropout"):
            raise RuntimeError("Expected Hugging Face SegFormer decode_head.dropout; incompatible transformers version.")

        hidden = int(getattr(self.net.config, "decoder_hidden_size", 256))
        mid = max(hidden // 2, 64)
        self.bg_head = nn.Sequential(
            nn.Conv2d(hidden, mid, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(mid), mid),
            nn.GELU(),
            nn.Dropout2d(0.10),
            nn.Conv2d(mid, 1, kernel_size=1),
        )

        self._decoder_feature = None

        def _capture_decoder_feature(module, inputs):
            # SegformerDecodeHead sends the fused decoder feature into dropout
            # immediately before the final semantic classifier.
            self._decoder_feature = inputs[0]

        self._bgaux_hook = decode_head.dropout.register_forward_pre_hook(_capture_decoder_feature)

    def forward(self, x):
        h, w = x.shape[-2:]
        self._decoder_feature = None
        logits = self.net(pixel_values=x).logits
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

        # Keep the public inference/evaluation contract identical to other models.
        if not self.training:
            self._decoder_feature = None
            return logits

        feat = self._decoder_feature
        if feat is None:
            raise RuntimeError("Could not capture SegFormer decoder feature for BG/FG auxiliary branch.")
        fg_logits = self.bg_head(feat)
        if fg_logits.shape[-2:] != (h, w):
            fg_logits = F.interpolate(fg_logits, size=(h, w), mode="bilinear", align_corners=False)

        return {"logits": logits, "fg_logits": fg_logits}
