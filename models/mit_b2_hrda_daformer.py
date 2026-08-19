"""
E043:
MiT-B2 + DAFormer-style decoder + HRDA-lite.

Relative to E042:
    SAME HRDA detail/context dataset
    SAME shared MiT-B2
    SAME class-wise HRDA scale attention
    SAME segmentation head
    SAME HRDA detail loss

ONLY major architectural change:
    UPerNet PPM/FPN decoder
        ->
    DAFormer-style multi-level context-aware decoder

DAFormer-inspired decoder:
    1. project C1,C2,C3,C4 -> common embedding dimension
    2. resize all levels to C1 resolution
    3. concatenate all levels
    4. depthwise-separable ASPP context fusion
       dilations = (1, 6, 12, 18)
    5. output 256-channel semantic feature
"""

import torch
from torch import nn
from torch.nn import functional as F

from .mit_b2_hrda_lite import MiTB2HRDALiteUPerNet


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        dilation=1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableDilatedConv(nn.Module):
    """
    Parameter-efficient dilated 3x3 convolution.

    depthwise:
        C -> C, grouped by channel
    pointwise:
        C -> out_channels
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dilation,
    ):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.pointwise(
            self.depthwise(x)
        )


class DAFormerStyleDecoder(nn.Module):
    """
    Lightweight PyTorch adaptation of the DAFormer decoder.

    Input MiT stages:
        C1 = 64
        C2 = 128
        C3 = 320
        C4 = 512

    All stages are projected to embed_dim=256,
    resized to C1 resolution and concatenated.

    The 1024-channel multi-level tensor is fused with
    parallel ASPP branches:
        dilation 1, 6, 12, 18.

    Output:
        B x 256 x H/4 x W/4
    """

    def __init__(
        self,
        in_channels=(64, 128, 320, 512),
        embed_dim=256,
        out_dim=256,
        dilations=(1, 6, 12, 18),
    ):
        super().__init__()

        self.embed_dim = int(embed_dim)
        self.out_dim = int(out_dim)

        # DAFormer uses per-level embeddings before fusion.
        # A 1x1 Conv is equivalent to a pixel-wise linear
        # projection for BCHW features.
        self.embed_layers = nn.ModuleList([
            nn.Conv2d(
                int(c),
                self.embed_dim,
                kernel_size=1,
                bias=True,
            )
            for c in in_channels
        ])

        fusion_channels = (
            self.embed_dim * len(in_channels)
        )

        branches = []

        for dilation in dilations:
            dilation = int(dilation)

            if dilation == 1:
                # Standard ASPP convention:
                # dilation=1 branch is a 1x1 projection.
                branch = ConvBNReLU(
                    fusion_channels,
                    self.out_dim,
                    kernel_size=1,
                )
            else:
                branch = DepthwiseSeparableDilatedConv(
                    fusion_channels,
                    self.out_dim,
                    dilation=dilation,
                )

            branches.append(branch)

        self.aspp = nn.ModuleList(branches)

        # Official DAFormer ASPP wrapper concatenates the
        # parallel context branches and applies a bottleneck.
        self.bottleneck = ConvBNReLU(
            self.out_dim * len(dilations),
            self.out_dim,
            kernel_size=3,
            padding=1,
        )

    def forward(self, feats):
        if len(feats) != 4:
            raise ValueError(
                f"Expected four MiT stages, got {len(feats)}"
            )

        target_hw = feats[0].shape[-2:]

        embedded = []

        for feat, layer in zip(
            feats,
            self.embed_layers,
        ):
            x = layer(feat)

            if x.shape[-2:] != target_hw:
                x = F.interpolate(
                    x,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )

            embedded.append(x)

        # Multi-level feature fusion.
        x = torch.cat(
            embedded,
            dim=1,
        )

        context = [
            branch(x)
            for branch in self.aspp
        ]

        x = torch.cat(
            context,
            dim=1,
        )

        return self.bottleneck(x)


class MiTB2HRDADAFormer(MiTB2HRDALiteUPerNet):
    """
    E043.

    Reuses E042 HRDA implementation exactly, but replaces
    UPerNet's PPM/FPN decoder with a DAFormer-style decoder.
    """

    def __init__(
        self,
        num_classes=7,
        pretrained=True,
        embed_dim=256,
        decoder_dim=256,
    ):
        # Build E042 first.
        #
        # This keeps backbone/head/scale-attention initialization
        # behavior identical under the same model seed.
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            fpn_dim=decoder_dim,
        )

        # ONLY structural replacement for E043.
        self.decoder = DAFormerStyleDecoder(
            in_channels=self.backbone.out_channels,
            embed_dim=embed_dim,
            out_dim=decoder_dim,
            dilations=(1, 6, 12, 18),
        )
