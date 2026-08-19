"""CNN + MiT-B2 hybrid structure experiments for LoveDA.

These are controlled adaptations built around the already-validated MiT-B2 + UPerNet
baseline. They are inspired by established local/global hybrid design ideas (CvT,
Conformer, LEFormer, UNetFormer) but are NOT verbatim reproductions of those models.

All models keep the Hugging Face MiT-B2 encoder and support loading the LoveDA-adapted
E007 SegFormer-B2 encoder via the inherited load_loveda_segformer_encoder() method.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .mit_b2_structures import _MiTB2StructureBase, SemanticFPNDecoder


def _cbr(in_ch, out_ch, k=3, s=1, p=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = _cbr(channels, channels, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            _cbr(in_ch, in_ch, 3, stride, 1, groups=in_ch),
            _cbr(in_ch, out_ch, 1, 1, 0),
            ResidualConvBlock(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class GatedLocalFusion(nn.Module):
    """Inject a local CNN feature into a Transformer feature with conservative gating.

    Gate bias starts negative, so the network begins close to the pretrained MiT path
    instead of letting a random CNN branch overwrite it at initialization.
    """

    def __init__(self, global_ch, local_ch):
        super().__init__()
        self.local_proj = nn.Conv2d(local_ch, global_ch, 1, bias=False)
        self.gate = nn.Conv2d(global_ch * 2, global_ch, 1, bias=True)
        self.refine = nn.Sequential(
            nn.Conv2d(global_ch, global_ch, 3, padding=1, groups=global_ch, bias=False),
            nn.BatchNorm2d(global_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(global_ch, global_ch, 1, bias=False),
            nn.BatchNorm2d(global_ch),
        )
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, global_feat, local_feat):
        if local_feat.shape[-2:] != global_feat.shape[-2:]:
            local_feat = F.interpolate(local_feat, size=global_feat.shape[-2:], mode="bilinear", align_corners=False)
        local = self.local_proj(local_feat)
        g = torch.sigmoid(self.gate(torch.cat([global_feat, local], dim=1)))
        out = global_feat + g * local
        return F.relu(out + self.refine(out), inplace=True)


class InputCNNPreprocessor(nn.Module):
    """Small residual CNN placed literally before MiT-B2.

    It learns local edge/texture correction in normalized RGB space while preserving the
    original 3-channel input and pretrained MiT patch embedding. Alpha is initialized
    small to make the initial network almost identical to the baseline.
    """

    def __init__(self, width=32):
        super().__init__()
        self.body = nn.Sequential(
            _cbr(3, width, 3, 1, 1),
            ResidualConvBlock(width),
            ResidualConvBlock(width),
            nn.Conv2d(width, 3, 3, padding=1, bias=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        delta = torch.tanh(self.body(x))
        return x + self.alpha * delta


class CNNStem(nn.Module):
    """Shallow high-resolution CNN stem; returns 1/4 resolution local features."""

    def __init__(self, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            _cbr(3, 32, 3, 2, 1),       # 1/2
            _cbr(32, out_ch, 3, 2, 1),  # 1/4
            ResidualConvBlock(out_ch),
            ResidualConvBlock(out_ch),
        )

    def forward(self, x):
        return self.net(x)


class CNNPyramid(nn.Module):
    """Efficient local CNN pyramid matching MiT stage resolutions 1/4..1/32."""

    def __init__(self, channels=(64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = CNNStem(c1)
        self.stage2 = DepthwiseSeparableBlock(c1, c2, stride=2)
        self.stage3 = DepthwiseSeparableBlock(c2, c3, stride=2)
        self.stage4 = DepthwiseSeparableBlock(c3, c4, stride=2)

    def forward(self, x):
        l1 = self.stem(x)
        l2 = self.stage2(l1)
        l3 = self.stage3(l2)
        l4 = self.stage4(l3)
        return (l1, l2, l3, l4)


class LocalConvEnhance(nn.Module):
    """Depthwise local refinement for one MiT stage (CvT/local-bias inspired)."""

    def __init__(self, channels):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x):
        return x + self.scale * self.local(x)


class _HybridUPerBase(_MiTB2StructureBase):
    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.decoder = SemanticFPNDecoder(self.backbone.out_channels, fpn_dim=fpn_dim, use_ppm=True)
        self.head = nn.Sequential(
            nn.Dropout2d(0.10),
            nn.Conv2d(fpn_dim, num_classes, 1),
        )

    def _decode(self, feats, x):
        feat = self.decoder(feats)
        return self._upsample_logits(self.head(feat), x)


class CNNPreMiTB2UPerNet(_HybridUPerBase):
    """E013: residual CNN preprocessor -> MiT-B2 -> UPerNet.

    This is the literal 'CNN first, Transformer second' test while retaining all MiT
    pretrained weights and its original 3-channel patch embedding.
    """

    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained, fpn_dim=fpn_dim)
        self.cnn_pre = InputCNNPreprocessor(width=32)

    def forward(self, x):
        x_enh = self.cnn_pre(x)
        return self._decode(self.backbone(x_enh), x)


class MiTB2CNNStemUPerNet(_HybridUPerBase):
    """E014: MiT-B2 plus an early 1/4-resolution CNN stem fused into C1."""

    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained, fpn_dim=fpn_dim)
        c1 = self.backbone.out_channels[0]
        self.local_stem = CNNStem(out_ch=64)
        self.fuse_c1 = GatedLocalFusion(c1, 64)

    def forward(self, x):
        c1, c2, c3, c4 = self.backbone(x)
        l1 = self.local_stem(x)
        c1 = self.fuse_c1(c1, l1)
        return self._decode((c1, c2, c3, c4), x)


class MiTB2CNNPyramidUPerNet(_HybridUPerBase):
    """E015: parallel CNN local pyramid + MiT global pyramid + gated stage-wise fusion.

    Conformer/LEFormer-inspired controlled adaptation for semantic segmentation.
    """

    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained, fpn_dim=fpn_dim)
        local_channels = (64, 128, 256, 512)
        self.local_pyramid = CNNPyramid(local_channels)
        self.fusions = nn.ModuleList([
            GatedLocalFusion(g, l)
            for g, l in zip(self.backbone.out_channels, local_channels)
        ])

    def forward(self, x):
        globals_ = self.backbone(x)
        locals_ = self.local_pyramid(x)
        fused = tuple(f(g, l) for f, g, l in zip(self.fusions, globals_, locals_))
        return self._decode(fused, x)


class MiTB2DetailUPerNet(_MiTB2StructureBase):
    """E016: UPerNet global-semantic path + late high-resolution CNN detail path.

    The Transformer pyramid is left untouched. A CNN branch retains local edges/textures
    at 1/4 resolution and is fused only after UPerNet, reducing interference with the
    pretrained global representation.
    """

    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256, detail_dim=96):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.decoder = SemanticFPNDecoder(self.backbone.out_channels, fpn_dim=fpn_dim, use_ppm=True)
        self.detail = nn.Sequential(
            _cbr(3, 32, 3, 2, 1),
            ResidualConvBlock(32),
            _cbr(32, detail_dim, 3, 2, 1),
            ResidualConvBlock(detail_dim),
            ResidualConvBlock(detail_dim),
        )
        self.fuse = nn.Sequential(
            _cbr(fpn_dim + detail_dim, fpn_dim, 3, 1, 1),
            ResidualConvBlock(fpn_dim),
            nn.Dropout2d(0.10),
        )
        self.head = nn.Conv2d(fpn_dim, num_classes, 1)

    def forward(self, x):
        global_feat = self.decoder(self.backbone(x))
        detail = self.detail(x)
        if detail.shape[-2:] != global_feat.shape[-2:]:
            detail = F.interpolate(detail, size=global_feat.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.fuse(torch.cat([global_feat, detail], dim=1))
        return self._upsample_logits(self.head(feat), x)


class MiTB2LocalConvUPerNet(_HybridUPerBase):
    """E017: MiT-B2 stages + depthwise convolutional local refinement + UPerNet.

    This tests whether adding explicit convolutional locality to each Transformer stage
    helps, without building a second encoder branch.
    """

    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained, fpn_dim=fpn_dim)
        self.local_refine = nn.ModuleList([LocalConvEnhance(c) for c in self.backbone.out_channels])

    def forward(self, x):
        feats = self.backbone(x)
        feats = tuple(m(f) for m, f in zip(self.local_refine, feats))
        return self._decode(feats, x)
