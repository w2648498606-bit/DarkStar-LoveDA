from torch import nn
from torch.nn import functional as F


# B0 keeps the same ImageNet-only initialization used in the local baseline.
# B1/B2 deliberately use the official ADE20K segmentation checkpoints: this
# initializes both the MiT encoder and the lightweight SegFormer decoder, while
# replacing only the final class classifier for LoveDA's 7 classes.
_SPECS = {
    "b0": {
        "checkpoint": "nvidia/mit-b0",
        "kind": "imagenet_encoder",
        "revision": None,
    },
    "b1": {
        "checkpoint": "nvidia/segformer-b1-finetuned-ade-512-512",
        "kind": "ade_segmentation",
        "revision": None,
    },
    # The official B2 repository has a verified safetensors conversion in PR #2.
    # Pin that revision so torch 2.5.x never needs to load pytorch_model.bin.
    "b2": {
        "checkpoint": "nvidia/segformer-b2-finetuned-ade-512-512",
        "kind": "ade_segmentation",
        "revision": "refs/pr/2",
    },
}


class SegFormer(nn.Module):
    """LoveDA SegFormer B0/B1/B2 wrapper.

    * B0: ImageNet-pretrained MiT-B0 encoder + fresh 7-class decoder.
    * B1/B2: ADE20K segmentation-pretrained encoder + decoder, but the final
      classifier is freshly initialized for LoveDA's 7 classes.

    All pretrained loading is forced through safetensors so AutoDL's existing
    torch 2.5.1 environment does not hit Transformers' torch.load restriction.
    """

    def __init__(self, variant="b0", num_classes=7, pretrained=True):
        super().__init__()
        variant = str(variant).lower()
        if variant not in _SPECS:
            raise ValueError(f"Unknown SegFormer variant: {variant}. Choose from {tuple(_SPECS)}")

        try:
            from transformers import (
                SegformerConfig,
                SegformerForImageClassification,
                SegformerForSemanticSegmentation,
            )
        except ImportError as exc:
            raise ImportError(
                "SegFormer requires `transformers` and `safetensors`. "
                "Run: python -m pip install -r requirements.txt"
            ) from exc

        spec = _SPECS[variant]
        checkpoint = spec["checkpoint"]
        revision = spec["revision"]
        common = {"revision": revision} if revision else {}

        # Use the official architecture config, then switch the target task to 7 classes.
        config = SegformerConfig.from_pretrained(checkpoint, **common)
        config.num_labels = int(num_classes)
        config.semantic_loss_ignore_index = 255
        config.id2label = {i: str(i) for i in range(num_classes)}
        config.label2id = {str(i): i for i in range(num_classes)}

        self.variant = variant
        self.checkpoint = checkpoint
        self.pretrain_kind = spec["kind"]
        self.net = SegformerForSemanticSegmentation(config)

        if pretrained and spec["kind"] == "imagenet_encoder":
            source = SegformerForImageClassification.from_pretrained(
                checkpoint,
                use_safetensors=True,
                **common,
            )
            self.net.segformer.load_state_dict(source.segformer.state_dict(), strict=True)
            del source

        elif pretrained and spec["kind"] == "ade_segmentation":
            source = SegformerForSemanticSegmentation.from_pretrained(
                checkpoint,
                use_safetensors=True,
                **common,
            )
            # Encoder: exact copy.
            self.net.segformer.load_state_dict(source.segformer.state_dict(), strict=True)

            # Decoder: copy all ADE20K-trained weights except the final classifier,
            # whose output channels are 150 in ADE20K but 7 in LoveDA.
            decoder_state = {
                k: v for k, v in source.decode_head.state_dict().items()
                if not k.startswith("classifier.")
            }
            missing, unexpected = self.net.decode_head.load_state_dict(decoder_state, strict=False)
            allowed_missing = {"classifier.weight", "classifier.bias"}
            if set(missing) != allowed_missing or unexpected:
                raise RuntimeError(
                    f"Unexpected decoder transfer mismatch. missing={missing}, unexpected={unexpected}"
                )
            del source

    def forward(self, x):
        h, w = x.shape[-2:]
        logits = self.net(pixel_values=x).logits
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits


class SegFormerB0(SegFormer):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__("b0", num_classes=num_classes, pretrained=pretrained)


class SegFormerB1(SegFormer):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__("b1", num_classes=num_classes, pretrained=pretrained)


class SegFormerB2(SegFormer):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__("b2", num_classes=num_classes, pretrained=pretrained)
