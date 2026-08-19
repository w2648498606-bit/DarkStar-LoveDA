"""MiT-B2 structure experiments for LoveDA.

Three decoder variants are included while keeping the same SegFormer MiT-B2 encoder:

1) MiT-B2 + UPerNet-style PPM/FPN decoder.
2) MiT-B2 + Semantic-FPN + OCR-style class-context refinement.
3) MiT-B2 + UPerNet-style decoder + OCR-style class-context refinement.

The implementation intentionally keeps the encoder common so the experiments isolate
how multi-scale fusion and class-aware context affect LoveDA performance.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


_B2_CHECKPOINT = "nvidia/segformer-b2-finetuned-ade-512-512"
_B2_REVISION = "refs/pr/2"


def _conv_bn_relu(in_ch, out_ch, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MiTB2Backbone(nn.Module):
    """Hugging Face MiT-B2 encoder with ADE20K segmentation-pretrained weights."""

    def __init__(self, pretrained=True):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerModel, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError(
                "MiT-B2 structure models require `transformers` and `safetensors`. "
                "Run: python -m pip install -r requirements.txt"
            ) from exc

        config = SegformerConfig.from_pretrained(_B2_CHECKPOINT, revision=_B2_REVISION)
        self.config = config
        self.segformer = SegformerModel(config)
        self.out_channels = tuple(int(x) for x in config.hidden_sizes)

        if pretrained:
            source = SegformerForSemanticSegmentation.from_pretrained(
                _B2_CHECKPOINT,
                revision=_B2_REVISION,
                use_safetensors=True,
            )
            self.segformer.load_state_dict(source.segformer.state_dict(), strict=True)
            del source

    def forward(self, x):
        out = self.segformer(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        feats = tuple(out.hidden_states)
        if len(feats) != 4:
            raise RuntimeError(f"Expected 4 MiT stages, got {len(feats)}")
        # Hugging Face SegFormer exposes each encoder stage as BCHW feature maps.
        for i, f in enumerate(feats):
            if f.ndim != 4:
                raise RuntimeError(f"MiT stage {i} is not BCHW: shape={tuple(f.shape)}")
        return feats


class PyramidPoolingModule(nn.Module):
    """PSP/UPerNet-style pyramid pooling on the deepest feature map."""

    def __init__(self, in_channels, out_channels=256, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for scale in pool_scales
        ])
        self.bottleneck = _conv_bn_relu(
            in_channels + len(pool_scales) * out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        parts = [x]
        for branch in self.branches:
            y = branch(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            parts.append(y)
        return self.bottleneck(torch.cat(parts, dim=1))


class SemanticFPNDecoder(nn.Module):
    """Top-down FPN with cross-layer fusion into the 1/4-resolution stage."""

    def __init__(self, in_channels=(64, 128, 320, 512), fpn_dim=256, use_ppm=False):
        super().__init__()
        self.use_ppm = bool(use_ppm)
        self.ppm = PyramidPoolingModule(in_channels[-1], fpn_dim) if use_ppm else None

        self.lateral = nn.ModuleList([
            nn.Conv2d(c, fpn_dim, kernel_size=1, bias=False)
            for c in in_channels[:3]
        ])
        if use_ppm:
            self.c4_lateral = nn.Identity()
        else:
            self.c4_lateral = nn.Conv2d(in_channels[-1], fpn_dim, kernel_size=1, bias=False)

        self.fpn_convs = nn.ModuleList([
            _conv_bn_relu(fpn_dim, fpn_dim, 3, 1) for _ in range(4)
        ])
        self.fuse = _conv_bn_relu(fpn_dim * 4, fpn_dim, 3, 1)

    def forward(self, feats):
        c1, c2, c3, c4 = feats
        p4 = self.ppm(c4) if self.use_ppm else self.c4_lateral(c4)
        p3 = self.lateral[2](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral[1](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral[0](c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        p1 = self.fpn_convs[0](p1)
        p2 = self.fpn_convs[1](p2)
        p3 = self.fpn_convs[2](p3)
        p4 = self.fpn_convs[3](p4)

        target = p1.shape[-2:]
        merged = torch.cat([
            p1,
            F.interpolate(p2, size=target, mode="bilinear", align_corners=False),
            F.interpolate(p3, size=target, mode="bilinear", align_corners=False),
            F.interpolate(p4, size=target, mode="bilinear", align_corners=False),
        ], dim=1)
        return self.fuse(merged)


class OCRClassContext(nn.Module):
    """OCR-style class-context refinement.

    A coarse segmentation map is used to gather one context prototype per class.
    Each pixel then attends to the class prototypes and receives class-level context.
    The coarse logits are returned as an auxiliary supervised output during training.
    """

    def __init__(self, channels=256, num_classes=7, key_channels=128):
        super().__init__()
        self.num_classes = int(num_classes)
        self.coarse_head = nn.Sequential(
            _conv_bn_relu(channels, channels, 3, 1),
            nn.Dropout2d(0.10),
            nn.Conv2d(channels, num_classes, kernel_size=1),
        )
        self.query = nn.Conv2d(channels, key_channels, kernel_size=1, bias=False)
        self.key = nn.Conv1d(channels, key_channels, kernel_size=1, bias=False)
        self.value = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.refine = nn.Sequential(
            _conv_bn_relu(channels * 2, channels, 3, 1),
            nn.Dropout2d(0.10),
        )
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.scale = key_channels ** -0.5

    def forward(self, feat):
        b, c, h, w = feat.shape
        n = h * w
        coarse = self.coarse_head(feat)

        # Gather class prototypes from the coarse semantic probabilities.
        probs = coarse.reshape(b, self.num_classes, n)
        probs = F.softmax(probs, dim=2)  # each class pools over spatial positions
        feat_flat = feat.reshape(b, c, n)
        prototypes = torch.bmm(feat_flat, probs.transpose(1, 2))  # B,C,K

        # Pixel-to-class relation: N pixels attend to K class prototypes.
        q = self.query(feat).reshape(b, -1, n).transpose(1, 2)      # B,N,D
        k = self.key(prototypes)                                    # B,D,K
        attn = torch.bmm(q, k) * self.scale                         # B,N,K
        attn = F.softmax(attn, dim=-1)
        v = self.value(prototypes).transpose(1, 2)                  # B,K,C
        ctx = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)

        refined = self.refine(torch.cat([feat, ctx], dim=1))
        logits = self.classifier(refined)
        return logits, coarse


class _MiTB2StructureBase(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        self.num_classes = int(num_classes)
        self.backbone = MiTB2Backbone(pretrained=pretrained)

    def _upsample_logits(self, logits, x):
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def load_loveda_segformer_encoder(self, state_dict):
        """Load only MiT-B2 encoder weights from an E007 SegFormer-B2 checkpoint."""
        prefix = "net.segformer."
        mapped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if not mapped:
            raise RuntimeError(
                "No `net.segformer.*` keys found. Expected a checkpoint from model=segformer_b2 (E007)."
            )
        incompatible = self.backbone.segformer.load_state_dict(mapped, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Encoder transfer mismatch: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        return len(mapped)


class MiTB2UPerNet(_MiTB2StructureBase):
    """MiT-B2 + UPerNet-style PPM/FPN multi-scale decoder."""

    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.decoder = SemanticFPNDecoder(self.backbone.out_channels, fpn_dim=fpn_dim, use_ppm=True)
        self.head = nn.Sequential(
            nn.Dropout2d(0.10),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1),
        )

    def forward(self, x):
        feat = self.decoder(self.backbone(x))
        return self._upsample_logits(self.head(feat), x)


class MiTB2OCRFPN(_MiTB2StructureBase):
    """MiT-B2 + Semantic-FPN + OCR-style class-aware context."""

    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.decoder = SemanticFPNDecoder(self.backbone.out_channels, fpn_dim=fpn_dim, use_ppm=False)
        self.ocr = OCRClassContext(fpn_dim, num_classes=num_classes, key_channels=128)

    def forward(self, x):
        feat = self.decoder(self.backbone(x))
        logits, aux = self.ocr(feat)
        logits = self._upsample_logits(logits, x)
        aux = self._upsample_logits(aux, x)
        if self.training:
            return {"logits": logits, "aux_logits": aux}
        return logits


class MiTB2UPerOCR(_MiTB2StructureBase):
    """MiT-B2 + UPerNet-style multi-scale fusion + OCR class-context refinement."""

    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.decoder = SemanticFPNDecoder(self.backbone.out_channels, fpn_dim=fpn_dim, use_ppm=True)
        self.ocr = OCRClassContext(fpn_dim, num_classes=num_classes, key_channels=128)

    def forward(self, x):
        feat = self.decoder(self.backbone(x))
        logits, aux = self.ocr(feat)
        logits = self._upsample_logits(logits, x)
        aux = self._upsample_logits(aux, x)
        if self.training:
            return {"logits": logits, "aux_logits": aux}
        return logits

class MultiDilatedBlock(nn.Module):
    """Lightweight multi-dilated context block inspired by AerialFormer's MD-CNN idea.

    This is an adaptation for a MiT-B2 encoder, not a reproduction of AerialFormer.
    Parallel dilated 3x3 branches capture local-to-broader context and are fused with
    a residual projection.
    """

    def __init__(self, in_channels, out_channels=256, dilations=(1, 3, 5)):
        super().__init__()
        self.pre = _conv_bn_relu(in_channels, out_channels, 1, 0)
        branch_ch = out_channels // len(dilations)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, branch_ch, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        ])
        fused_ch = branch_ch * len(dilations)
        self.fuse = _conv_bn_relu(fused_ch, out_channels, 1, 0)
        self.refine = _conv_bn_relu(out_channels, out_channels, 3, 1)

    def forward(self, x):
        base = self.pre(x)
        y = self.fuse(torch.cat([b(base) for b in self.branches], dim=1))
        return self.refine(y + base)


class HierarchicalMDCDecoder(nn.Module):
    """Top-down multi-resolution decoder with multi-dilated convolution at each scale."""

    def __init__(self, in_channels=(64, 128, 320, 512), dim=256):
        super().__init__()
        self.c4 = MultiDilatedBlock(in_channels[3], dim)
        self.c3 = MultiDilatedBlock(in_channels[2] + dim, dim)
        self.c2 = MultiDilatedBlock(in_channels[1] + dim, dim)
        self.c1 = MultiDilatedBlock(in_channels[0] + dim, dim)
        self.final = _conv_bn_relu(dim, dim, 3, 1)

    def forward(self, feats):
        c1, c2, c3, c4 = feats
        x = self.c4(c4)
        x = F.interpolate(x, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.c3(torch.cat([c3, x], dim=1))
        x = F.interpolate(x, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.c2(torch.cat([c2, x], dim=1))
        x = F.interpolate(x, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.c1(torch.cat([c1, x], dim=1))
        return self.final(x)


class MiTB2MDC(_MiTB2StructureBase):
    """MiT-B2 + hierarchical multi-dilated CNN decoder (AerialFormer-inspired adaptation)."""

    def __init__(self, num_classes=7, pretrained=True, dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.decoder = HierarchicalMDCDecoder(self.backbone.out_channels, dim=dim)
        self.head = nn.Sequential(nn.Dropout2d(0.10), nn.Conv2d(dim, num_classes, kernel_size=1))

    def forward(self, x):
        feat = self.decoder(self.backbone(x))
        return self._upsample_logits(self.head(feat), x)
