"""v7.1 literature-driven strict adapters on top of the E009 MiT-B2 + UPerNet baseline.

Design goal
-----------
Keep the complete E009 model (backbone, UPerNet decoder, classifier) intact and add
only residual correction modules whose initial contribution is exactly zero. This
allows a strict epoch-0 parity test with E009, followed by a two-stage training plan:
(1) freeze E009 and train only the new module; (2) jointly fine-tune with very small
learning rates.

These modules are *paper-inspired controlled adaptations*, not claims of exact paper
reproduction:
- WideContextAdapter: inspired by WiCoNet's larger-area context branch + selective
  context projection. Unlike the older v6 global-vector prototype, this keeps a 2-D
  context feature map and uses cross-attention from the local crop to the full tile.
- FrequencyRFAdapter: inspired by FADC's frequency-conditioned spatially adaptive
  receptive-field principle. Standard PyTorch convolutions cannot express the exact
  per-pixel dynamic dilation operator used by FADC, so this implementation uses
  frequency-conditioned spatial gates over multiple dilation branches.
- DynamicDictionaryAdapter: inspired by D2LS dynamic dictionary learning. It builds
  image-conditioned class prototypes and alternates dictionary<-image and
  image<-dictionary cross-attention twice before a zero-init residual projection.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .mit_b2_structures import MiTB2UPerNet


def _gn(channels: int):
    groups = 32
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


def _zero_conv(in_ch: int, out_ch: int, kernel_size=1, padding=0):
    layer = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=True)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class _V71Base(MiTB2UPerNet):
    ablation_base_model = "mit_b2_upernet"
    ablation_new_prefixes = ("adapter.",)
    uses_context_view = False

    def ablation_new_modules(self):
        return (self.adapter,)

    def _base_feature(self, x):
        return self.decoder(self.backbone(x))

    def _finish(self, feat, x):
        logits = self.head(feat)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


class WideContextAdapter(nn.Module):
    """Spatial wide-context residual adapter.

    Local UPerNet features query a 2-D feature map extracted from the complete LoveDA
    tile (resized to context_size by the dataset). The cross-attention is intentionally
    performed at a reduced spatial resolution for tractable memory use.
    """

    def __init__(self, channels=256, context_dim=128, heads=4, local_tokens_hw=24):
        super().__init__()
        self.local_tokens_hw = int(local_tokens_hw)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(3, 48, 5, stride=2, padding=2, bias=False),
            _gn(48), nn.GELU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1, bias=False),
            _gn(96), nn.GELU(),
            nn.Conv2d(96, context_dim, 3, stride=2, padding=1, bias=False),
            _gn(context_dim), nn.GELU(),
            nn.Conv2d(context_dim, context_dim, 3, stride=2, padding=1, bias=False),
            _gn(context_dim), nn.GELU(),
        )
        self.q_proj = nn.Linear(channels, context_dim, bias=False)
        self.attn = nn.MultiheadAttention(context_dim, heads, batch_first=True, dropout=0.0)
        self.ctx_to_feat = nn.Sequential(
            nn.Conv2d(context_dim, channels, 1, bias=False),
            _gn(channels),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1),
        )
        self.out = _zero_conv(channels, channels, 1)

    def forward(self, local_feat, context_image):
        b, c, h, w = local_feat.shape
        ctx = self.context_encoder(context_image)  # B,D,Hc,Wc, keeps spatial layout
        d = ctx.shape[1]

        lh = min(self.local_tokens_hw, h)
        lw = min(self.local_tokens_hw, w)
        local_small = F.adaptive_avg_pool2d(local_feat, (lh, lw))
        q = local_small.flatten(2).transpose(1, 2)  # B,N,C
        q = self.q_proj(q)                           # B,N,D
        kv = ctx.flatten(2).transpose(1, 2)         # B,M,D
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        attended = attended.transpose(1, 2).reshape(b, d, lh, lw)
        attended = self.ctx_to_feat(attended)
        attended = F.interpolate(attended, size=(h, w), mode="bilinear", align_corners=False)

        g = torch.sigmoid(self.gate(torch.cat([local_feat, attended], dim=1)))
        delta = self.out(g * attended)
        return local_feat + delta


class FrequencyRFAdapter(nn.Module):
    """Frequency-conditioned multi-receptive-field residual adapter.

    A high-frequency proxy (feature - local mean) drives per-pixel mixture weights over
    dilation branches. This is closer to the *principle* of FADC than a fixed global
    branch mixture, while remaining implementable with standard PyTorch operators.
    """

    def __init__(self, channels=256, hidden=96, dilations=(1, 2, 4, 6)):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            _gn(hidden), nn.GELU(),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=d, dilation=d, groups=hidden, bias=False),
                nn.Conv2d(hidden, hidden, 1, bias=False),
                _gn(hidden), nn.GELU(),
            ) for d in dilations
        ])
        self.freq_gate = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32, len(dilations), 1, bias=True),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(hidden, channels, 1, bias=False),
            _gn(channels), nn.GELU(),
        )
        self.out = _zero_conv(channels, channels, 1)

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        high = x - low
        # Frequency descriptors are channel-collapsed but spatially preserved.
        freq_desc = torch.cat([
            high.abs().mean(dim=1, keepdim=True),
            low.abs().mean(dim=1, keepdim=True),
        ], dim=1)
        weights = F.softmax(self.freq_gate(freq_desc), dim=1)

        z = self.pre(x)
        ys = [branch(z) for branch in self.branches]
        mixed = torch.zeros_like(ys[0])
        for i, y in enumerate(ys):
            mixed = mixed + weights[:, i:i+1] * y
        delta = self.out(self.mix(mixed))
        return x + delta


class DynamicDictionaryAdapter(nn.Module):
    """D2LS-inspired image-conditioned class dictionary adapter.

    The frozen/slow E009 classifier supplies coarse class probabilities. They are used
    to initialize one prototype per LoveDA class. Learnable class-ID embeddings are
    added, then two alternating cross-attention refinement steps are applied.
    """

    def __init__(self, channels=256, num_classes=7, token_hw=24, heads=4, iters=2):
        super().__init__()
        self.num_classes = int(num_classes)
        self.token_hw = int(token_hw)
        self.iters = int(iters)
        self.class_id = nn.Parameter(torch.zeros(num_classes, channels))
        nn.init.trunc_normal_(self.class_id, std=0.02)
        self.dict_from_image = nn.MultiheadAttention(channels, heads, batch_first=True, dropout=0.0)
        self.image_from_dict = nn.MultiheadAttention(channels, heads, batch_first=True, dropout=0.0)
        self.dict_norm = nn.LayerNorm(channels)
        self.image_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            _gn(channels), nn.GELU(),
        )
        self.out = _zero_conv(channels, channels, 1)

    def forward(self, feat, coarse_logits):
        b, c, h, w = feat.shape
        th = min(self.token_hw, h)
        tw = min(self.token_hw, w)
        feat_small = F.adaptive_avg_pool2d(feat, (th, tw))
        logits_small = F.interpolate(coarse_logits, size=(th, tw), mode="bilinear", align_corners=False)

        image_tokens = feat_small.flatten(2).transpose(1, 2)  # B,N,C
        probs = F.softmax(logits_small, dim=1).flatten(2)      # B,K,N
        denom = probs.sum(dim=2, keepdim=True).clamp_min(1e-6)
        proto = torch.bmm(probs, image_tokens) / denom          # B,K,C
        dictionary = proto + self.class_id.unsqueeze(0)

        for _ in range(self.iters):
            d_upd, _ = self.dict_from_image(dictionary, image_tokens, image_tokens, need_weights=False)
            dictionary = self.dict_norm(dictionary + d_upd)
            i_upd, _ = self.image_from_dict(image_tokens, dictionary, dictionary, need_weights=False)
            image_tokens = self.image_norm(image_tokens + i_upd)
            image_tokens = self.image_norm(image_tokens + self.ffn(image_tokens))

        refined = image_tokens.transpose(1, 2).reshape(b, c, th, tw)
        refined = self.refine(refined)
        refined = F.interpolate(refined, size=(h, w), mode="bilinear", align_corners=False)
        delta = self.out(refined)
        return feat + delta


class MiTB2UPerNetV71WideContext(_V71Base):
    uses_context_view = True

    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained, fpn_dim=fpn_dim)
        self.adapter = WideContextAdapter(fpn_dim)

    def forward(self, x, context_image=None):
        feat = self._base_feature(x)
        if context_image is None:
            # For engineering/smoke tests only. Real v7.1 training supplies the full tile.
            context_image = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        feat = self.adapter(feat, context_image)
        return self._finish(feat, x)


class MiTB2UPerNetV71FADCLite(_V71Base):
    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained, fpn_dim=fpn_dim)
        self.adapter = FrequencyRFAdapter(fpn_dim)

    def forward(self, x):
        return self._finish(self.adapter(self._base_feature(x)), x)


class MiTB2UPerNetV71DynamicDictionary(_V71Base):
    def __init__(self, num_classes=7, pretrained=True, fpn_dim=256):
        super().__init__(num_classes=num_classes, pretrained=pretrained, fpn_dim=fpn_dim)
        self.adapter = DynamicDictionaryAdapter(fpn_dim, num_classes=num_classes)

    def forward(self, x):
        feat = self._base_feature(x)
        # Use the already-trained E009 semantic head to seed image-conditioned class prototypes.
        coarse = self.head(feat)
        feat = self.adapter(feat, coarse)
        return self._finish(feat, x)
