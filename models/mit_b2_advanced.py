"""Advanced MiT-B2 + UPerNet exploration models for LoveDA.

These are controlled adaptations around the current winning MiT-B2 + UPerNet baseline.
They are inspired by several published directions but are intentionally lightweight,
compatible variants for ablation on the existing project rather than verbatim reimplementations:

- Adaptive receptive field / frequency-aware refinement (FADC, AFENet/LSK ideas)
- Selective pyramid fusion (PyramidMamba motivation, without requiring mamba-ssm)
- Dynamic class dictionary refinement (D2LS motivation)
- Class-query mask prediction (Mask2Former motivation)
- Wider scene-context conditioning (WiCoNet motivation)

All B2-based models inherit load_loveda_segformer_encoder(), so they can initialize the
MiT-B2 encoder from E007 while newly added modules start fresh.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .mit_b2_structures import (
    _MiTB2StructureBase,
    SemanticFPNDecoder,
    PyramidPoolingModule,
)


def _cbr(in_ch, out_ch, k=3, p=1, d=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            padding=p,
            dilation=d,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )


class _UPerFeatureBase(_MiTB2StructureBase):
    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.fpn_dim = int(fpn_dim)
        self.decoder = SemanticFPNDecoder(
            self.backbone.out_channels,
            fpn_dim=self.fpn_dim,
            use_ppm=True,
        )

    def _features(self, x):
        return self.decoder(self.backbone(x))

    def _finish(self, logits, x):
        return self._upsample_logits(logits, x)


class AdaptiveReceptiveFieldBlock(nn.Module):
    """Spatially gated multi-dilation refinement.

    Each pixel learns how much to use four receptive-field sizes. This is an
    inexpensive FADC/large-selective-kernel inspired test, not the exact FADC operator.
    """

    def __init__(self, channels=256, dilations=(1, 2, 3, 5)):
        super().__init__()
        self.dilations = tuple(int(d) for d in dilations)
        self.branches = nn.ModuleList()
        for d in self.dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        3,
                        padding=d,
                        dilation=d,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 1, bias=False),
                    nn.BatchNorm2d(channels),
                )
            )
        hidden = max(channels // 4, 32)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, len(self.dilations), 1, bias=True),
        )
        self.out = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x):
        weights = F.softmax(self.gate(x), dim=1)
        ys = [branch(x) for branch in self.branches]
        fused = torch.zeros_like(ys[0])
        for i, y in enumerate(ys):
            fused = fused + weights[:, i : i + 1] * y
        return x + self.scale * self.out(fused)


class MiTB2AdaptiveRFUPerNet(_UPerFeatureBase):
    """E018: MiT-B2 + UPerNet + adaptive receptive-field refinement."""

    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained, fpn_dim=fpn_dim)
        self.refine = AdaptiveReceptiveFieldBlock(fpn_dim)
        self.head = nn.Sequential(nn.Dropout2d(0.10), nn.Conv2d(fpn_dim, num_classes, 1))

    def forward(self, x):
        feat = self.refine(self._features(x))
        return self._finish(self.head(feat), x)


class FrequencySpatialInteraction(nn.Module):
    """Adaptive high-/low-frequency decomposition and fusion.

    Low frequency is represented by a local smoothing path; high frequency is the
    residual x-low. A content-conditioned spatial gate decides which component should
    dominate at each location. This is AFENet-inspired, not a verbatim AFSIM copy.
    """

    def __init__(self, channels=256):
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.high_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        hidden = max(channels // 4, 32)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 2, 1, bias=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        high = x - low
        low = self.low_proj(low)
        high = self.high_proj(high)
        w = F.softmax(self.gate(x), dim=1)
        fused = w[:, 0:1] * low + w[:, 1:2] * high
        return x + self.scale * self.fuse(fused)


class MiTB2FrequencyUPerNet(_UPerFeatureBase):
    """E019: MiT-B2 + UPerNet + adaptive frequency/spatial refinement."""

    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained, fpn_dim=fpn_dim)
        self.freq = FrequencySpatialInteraction(fpn_dim)
        self.head = nn.Sequential(nn.Dropout2d(0.10), nn.Conv2d(fpn_dim, num_classes, 1))

    def forward(self, x):
        feat = self.freq(self._features(x))
        return self._finish(self.head(feat), x)


class SelectivePyramidDecoder(nn.Module):
    """UPer-style top-down pyramid with learned scale selection instead of concat.

    PyramidMamba motivates selective removal of redundant pyramid semantics. This
    implementation uses spatial softmax gates rather than a state-space operator so it
    remains dependency-free and isolates the selective-fusion hypothesis.
    """

    def __init__(self, in_channels=(64, 128, 320, 512), fpn_dim=256):
        super().__init__()
        self.ppm = PyramidPoolingModule(in_channels[-1], fpn_dim)
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, fpn_dim, 1, bias=False) for c in in_channels[:3]
        ])
        self.smooth = nn.ModuleList([_cbr(fpn_dim, fpn_dim, 3, 1) for _ in range(4)])
        hidden = max(fpn_dim // 4, 32)
        self.scale_gate = nn.Sequential(
            nn.Conv2d(fpn_dim * 4, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 4, 1, bias=True),
        )
        self.refine = nn.Sequential(
            _cbr(fpn_dim, fpn_dim, 3, 1),
            _cbr(fpn_dim, fpn_dim, 3, 1),
        )

    def forward(self, feats):
        c1, c2, c3, c4 = feats
        p4 = self.ppm(c4)
        p3 = self.lateral[2](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral[1](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral[0](c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        ps = [m(p) for m, p in zip(self.smooth, (p1, p2, p3, p4))]
        target = ps[0].shape[-2:]
        ups = [ps[0]] + [F.interpolate(p, size=target, mode="bilinear", align_corners=False) for p in ps[1:]]
        stacked = torch.cat(ups, dim=1)
        weights = F.softmax(self.scale_gate(stacked), dim=1)
        fused = torch.zeros_like(ups[0])
        for i, p in enumerate(ups):
            fused = fused + weights[:, i : i + 1] * p
        return self.refine(fused)


class MiTB2SelectivePyramid(_MiTB2StructureBase):
    """E020: MiT-B2 + selectively gated UPer-style pyramid fusion."""

    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        self.decoder = SelectivePyramidDecoder(self.backbone.out_channels, fpn_dim=fpn_dim)
        self.head = nn.Sequential(nn.Dropout2d(0.10), nn.Conv2d(fpn_dim, num_classes, 1))

    def forward(self, x):
        feat = self.decoder(self.backbone(x))
        return self._upsample_logits(self.head(feat), x)


class DynamicDictionaryRefiner(nn.Module):
    """Lightweight dynamic class-dictionary refinement.

    It maintains learnable class IDs, builds image-specific prototypes from coarse
    probabilities, then performs two rounds of pixel-to-class attention. This tests the
    central D2LS hypothesis (dynamic class embeddings for intra-class variation) while
    keeping memory O(HW*K) with K=7.
    """

    def __init__(self, channels=256, num_classes=7, dict_dim=128, iterations=2):
        super().__init__()
        self.num_classes = int(num_classes)
        self.iterations = int(iterations)
        self.class_ids = nn.Parameter(torch.randn(num_classes, channels) * 0.02)
        self.coarse = nn.Conv2d(channels, num_classes, 1)
        self.q = nn.Conv2d(channels, dict_dim, 1, bias=False)
        self.k = nn.Linear(channels, dict_dim, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.dict_mix = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.pixel_refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.scale = dict_dim ** -0.5

    def _prototypes(self, feat, coarse_logits):
        b, c, h, w = feat.shape
        n = h * w
        probs = F.softmax(coarse_logits.reshape(b, self.num_classes, n), dim=2)
        f = feat.reshape(b, c, n)
        proto = torch.bmm(f, probs.transpose(1, 2)).transpose(1, 2)  # B,K,C
        return proto

    def forward(self, feat):
        b, c, h, w = feat.shape
        coarse = self.coarse(feat)
        dynamic = self._prototypes(feat, coarse)
        ids = self.class_ids.unsqueeze(0).expand(b, -1, -1)
        dictionary = self.dict_mix(torch.cat([ids, dynamic], dim=-1)) + ids

        x = feat
        for _ in range(self.iterations):
            q = self.q(x).flatten(2).transpose(1, 2)                 # B,N,D
            k = self.k(dictionary).transpose(1, 2)                  # B,D,K
            attn = F.softmax(torch.bmm(q, k) * self.scale, dim=-1)  # B,N,K
            v = self.v(dictionary)                                  # B,K,C
            ctx = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
            x = x + self.pixel_refine(torch.cat([x, ctx], dim=1))
            coarse = self.coarse(x)
            dynamic = self._prototypes(x, coarse)
            dictionary = dictionary + self.dict_mix(torch.cat([dictionary, dynamic], dim=-1))
        return x


class MiTB2DynamicDictionaryUPerNet(_UPerFeatureBase):
    """E021: MiT-B2 + UPerNet + dynamic class dictionary refinement."""

    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained, fpn_dim=fpn_dim)
        self.dictionary = DynamicDictionaryRefiner(fpn_dim, num_classes=num_classes)
        self.head = nn.Sequential(nn.Dropout2d(0.10), nn.Conv2d(fpn_dim, num_classes, 1))

    def forward(self, x):
        feat = self.dictionary(self._features(x))
        return self._finish(self.head(feat), x)


class ClassQueryMaskHead(nn.Module):
    """Fixed semantic class-query mask head.

    Each of the seven class queries produces a dense mask through feature/query
    similarity. It is Mask2Former-inspired but avoids Hungarian matching because the
    seven semantic classes have fixed identities.
    """

    def __init__(self, channels=256, num_classes=7, embed_dim=128):
        super().__init__()
        self.num_classes = int(num_classes)
        self.pixel_embed = nn.Sequential(
            nn.Conv2d(channels, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.query = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.scene_to_query = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, num_classes * embed_dim),
        )
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, feat):
        b, c, h, w = feat.shape
        pix = self.pixel_embed(feat)
        pix = F.normalize(pix, dim=1).flatten(2)                         # B,D,N
        scene = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        delta = self.scene_to_query(scene).reshape(b, self.num_classes, -1)
        q = F.normalize(self.query.unsqueeze(0) + 0.10 * delta, dim=-1) # B,K,D
        scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
        logits = torch.bmm(q, pix).reshape(b, self.num_classes, h, w)
        return scale * logits + self.bias.view(1, -1, 1, 1)


class MiTB2ClassQueryUPerNet(_UPerFeatureBase):
    """E022: MiT-B2 + UPerNet + fixed class-query mask head."""

    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained, fpn_dim=fpn_dim)
        self.mask_head = ClassQueryMaskHead(fpn_dim, num_classes=num_classes)

    def forward(self, x):
        logits = self.mask_head(self._features(x))
        return self._finish(logits, x)


class SceneContextEncoder(nn.Module):
    """Cheap scene encoder for a resized full 1024x1024 LoveDA tile."""

    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            _cbr(3, 32, 3, 1),
            _cbr(32, 64, 3, 1),
            nn.MaxPool2d(2),
            _cbr(64, 96, 3, 1),
            nn.MaxPool2d(2),
            _cbr(96, 128, 3, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(self.net(x))


class SceneFiLM(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.to_gamma_beta = nn.Linear(channels, channels * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, feat, scene):
        gb = self.to_gamma_beta(scene)
        gamma, beta = gb.chunk(2, dim=1)
        strength = torch.sigmoid(self.gate)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return feat * (1.0 + strength * gamma) + strength * beta


class MiTB2WideContextUPerNet(_UPerFeatureBase):
    """E023: local crop MiT-B2/UPerNet conditioned by the full-tile scene context.

    During training, x is the usual 512 crop while context_image is the entire LoveDA
    tile resized to 256. During validation x is the full image and context_image is its
    resized scene view. This directly tests the crop-context hypothesis highlighted by
    WiCoNet. If context_image is omitted, a 256x256 resize of x is used as fallback so
    smoke tests remain simple.
    """

    uses_context_view = True

    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained, fpn_dim=fpn_dim)
        self.scene_encoder = SceneContextEncoder(fpn_dim)
        self.scene_film = SceneFiLM(fpn_dim)
        self.head = nn.Sequential(nn.Dropout2d(0.10), nn.Conv2d(fpn_dim, num_classes, 1))

    def forward(self, x, context_image=None):
        if context_image is None:
            context_image = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        scene = self.scene_encoder(context_image)
        feat = self.scene_film(self._features(x), scene)
        return self._finish(self.head(feat), x)


__all__ = [
    "MiTB2AdaptiveRFUPerNet",
    "MiTB2FrequencyUPerNet",
    "MiTB2SelectivePyramid",
    "MiTB2DynamicDictionaryUPerNet",
    "MiTB2ClassQueryUPerNet",
    "MiTB2WideContextUPerNet",
]
