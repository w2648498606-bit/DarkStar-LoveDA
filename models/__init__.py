from .fcn_resnet18 import FCNResNet18
from .deeplabv3_resnet50 import DeepLabV3ResNet50
from .segformer import SegFormerB0, SegFormerB1, SegFormerB2
from .segformer_bgaux import SegFormerB2BGAux
from .mit_b2_structures import MiTB2UPerNet, MiTB2OCRFPN, MiTB2UPerOCR, MiTB2MDC
from .mit_b2_hierarchical import MiTB2HierUPerNet
from .mit_b2_forest_rescue import MiTB2ForestRescueUPerNet
from .mit_b2_hrda_lite import MiTB2HRDALiteUPerNet
from .mit_b2_hrda_daformer import MiTB2HRDADAFormer
from .mit_b2_hybrids import (CNNPreMiTB2UPerNet, MiTB2CNNStemUPerNet, MiTB2CNNPyramidUPerNet, MiTB2DetailUPerNet, MiTB2LocalConvUPerNet)
from .mit_b2_v7_ablation import MiTB2UPerNetV7RF, MiTB2UPerNetV7Frequency, MiTB2UPerNetV7LocalGlobal
from .mit_b2_v71_literature import MiTB2UPerNetV71WideContext, MiTB2UPerNetV71FADCLite, MiTB2UPerNetV71DynamicDictionary
from .mit_b2_advanced import (
    MiTB2AdaptiveRFUPerNet,
    MiTB2FrequencyUPerNet,
    MiTB2SelectivePyramid,
    MiTB2DynamicDictionaryUPerNet,
    MiTB2ClassQueryUPerNet,
    MiTB2WideContextUPerNet,
)

MODEL_NAMES = (
    "fcn_resnet18",
    "deeplabv3_resnet50",
    "segformer_b0",
    "segformer_b1",
    "segformer_b2",
    "segformer_b2_bgaux",
    "mit_b2_upernet",
    "mit_b2_hier_upernet",
    "mit_b2_forest_rescue_upernet",
    "mit_b2_hrda_lite_upernet",
    "mit_b2_hrda_daformer",
    "mit_b2_ocr_fpn",
    "mit_b2_uper_ocr",
    "mit_b2_mdc",
    "cnnpre_mit_b2_upernet",
    "mit_b2_cnnstem_upernet",
    "mit_b2_cnnpyr_upernet",
    "mit_b2_detail_upernet",
    "mit_b2_localconv_upernet",
    "mit_b2_adaptive_rf_upernet",
    "mit_b2_frequency_upernet",
    "mit_b2_selective_pyramid",
    "mit_b2_dynamic_dict_upernet",
    "mit_b2_class_query_upernet",
    "mit_b2_widecontext_upernet",
    "mit_b2_upernet_v7_rf",
    "mit_b2_upernet_v7_frequency",
    "mit_b2_upernet_v7_localglobal",
    "mit_b2_upernet_v71_widecontext",
    "mit_b2_upernet_v71_fadc_lite",
    "mit_b2_upernet_v71_dynamic_dict",
)

MODEL_LABELS = {
    "fcn_resnet18": "FCN-ResNet18",
    "deeplabv3_resnet50": "DeepLabV3-ResNet50",
    "segformer_b0": "SegFormer-B0",
    "segformer_b1": "SegFormer-B1 (ADE20K pretrained)",
    "segformer_b2": "SegFormer-B2 (ADE20K pretrained)",
    "segformer_b2_bgaux": "SegFormer-B2 + BG/FG auxiliary head",
    "mit_b2_upernet": "MiT-B2 + UPerNet-style PPM/FPN",
    "mit_b2_hier_upernet": "MiT-B2 + UPerNet + hierarchical BG/FG head",
    "mit_b2_forest_rescue_upernet": "MiT-B2 + UPerNet + Forest/BG rescue head",
    "mit_b2_hrda_lite_upernet": "MiT-B2 + UPerNet + HRDA-lite multi-resolution fusion",
    "mit_b2_hrda_daformer": "MiT-B2 + DAFormer-style decoder + HRDA-lite",
    "mit_b2_ocr_fpn": "MiT-B2 + Semantic-FPN + OCR class context",
    "mit_b2_uper_ocr": "MiT-B2 + UPerNet-style decoder + OCR class context",
    "mit_b2_mdc": "MiT-B2 + hierarchical multi-dilated CNN decoder (AerialFormer-inspired)",
    "cnnpre_mit_b2_upernet": "CNN preprocessor + MiT-B2 + UPerNet",
    "mit_b2_cnnstem_upernet": "MiT-B2 + early CNN stem fusion + UPerNet",
    "mit_b2_cnnpyr_upernet": "Parallel CNN pyramid + MiT-B2 + gated fusion + UPerNet",
    "mit_b2_detail_upernet": "MiT-B2 + UPerNet + high-resolution CNN detail branch",
    "mit_b2_localconv_upernet": "MiT-B2 + stage-wise local convolution refinement + UPerNet",
    "mit_b2_adaptive_rf_upernet": "MiT-B2 + UPerNet + adaptive receptive-field gating",
    "mit_b2_frequency_upernet": "MiT-B2 + UPerNet + adaptive frequency/spatial refinement",
    "mit_b2_selective_pyramid": "MiT-B2 + selectively gated pyramid fusion",
    "mit_b2_dynamic_dict_upernet": "MiT-B2 + UPerNet + dynamic class dictionary",
    "mit_b2_class_query_upernet": "MiT-B2 + UPerNet + fixed class-query mask head",
    "mit_b2_widecontext_upernet": "MiT-B2 + UPerNet + full-tile scene context",
    "mit_b2_upernet_v7_rf": "v7 strict: E009 UPerNet + zero-init multi-RF adapter",
    "mit_b2_upernet_v7_frequency": "v7 strict: E009 UPerNet + zero-init frequency adapter",
    "mit_b2_upernet_v7_localglobal": "v7 strict: E009 UPerNet + zero-init local/global adapter",
    "mit_b2_upernet_v71_widecontext": "v7.1: E009 + spatial wide-context cross-attention adapter (WiCoNet-inspired)",
    "mit_b2_upernet_v71_fadc_lite": "v7.1: E009 + frequency-conditioned multi-dilation adapter (FADC-inspired)",
    "mit_b2_upernet_v71_dynamic_dict": "v7.1: E009 + iterative dynamic class dictionary adapter (D2LS-inspired)",
}


def build_model(name: str, num_classes: int = 7, pretrained: bool = True):
    if name == "fcn_resnet18":
        return FCNResNet18(num_classes=num_classes, pretrained=pretrained)
    if name == "deeplabv3_resnet50":
        return DeepLabV3ResNet50(num_classes=num_classes, pretrained=pretrained, freeze_bn=True)
    if name == "segformer_b0":
        return SegFormerB0(num_classes=num_classes, pretrained=pretrained)
    if name == "segformer_b1":
        return SegFormerB1(num_classes=num_classes, pretrained=pretrained)
    if name == "segformer_b2":
        return SegFormerB2(num_classes=num_classes, pretrained=pretrained)
    if name == "segformer_b2_bgaux":
        return SegFormerB2BGAux(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_upernet":
        return MiTB2UPerNet(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_hier_upernet":
        return MiTB2HierUPerNet(
            num_classes=num_classes,
            pretrained=pretrained,
        )
    if name == "mit_b2_forest_rescue_upernet":
        return MiTB2ForestRescueUPerNet(
            num_classes=num_classes,
            pretrained=pretrained,
            rescue_scale=1.0,
        )
    if name == "mit_b2_hrda_lite_upernet":
        return MiTB2HRDALiteUPerNet(
            num_classes=num_classes,
            pretrained=pretrained,
        )
    if name == "mit_b2_hrda_daformer":
        return MiTB2HRDADAFormer(
            num_classes=num_classes,
            pretrained=pretrained,
        )
    if name == "mit_b2_ocr_fpn":
        return MiTB2OCRFPN(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_uper_ocr":
        return MiTB2UPerOCR(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_mdc":
        return MiTB2MDC(num_classes=num_classes, pretrained=pretrained)
    if name == "cnnpre_mit_b2_upernet":
        return CNNPreMiTB2UPerNet(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_cnnstem_upernet":
        return MiTB2CNNStemUPerNet(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_cnnpyr_upernet":
        return MiTB2CNNPyramidUPerNet(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_detail_upernet":
        return MiTB2DetailUPerNet(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_localconv_upernet":
        return MiTB2LocalConvUPerNet(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_adaptive_rf_upernet":
        return MiTB2AdaptiveRFUPerNet(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_frequency_upernet":
        return MiTB2FrequencyUPerNet(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_selective_pyramid":
        return MiTB2SelectivePyramid(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_dynamic_dict_upernet":
        return MiTB2DynamicDictionaryUPerNet(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_class_query_upernet":
        return MiTB2ClassQueryUPerNet(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_widecontext_upernet":
        return MiTB2WideContextUPerNet(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_upernet_v7_rf":
        return MiTB2UPerNetV7RF(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_upernet_v7_frequency":
        return MiTB2UPerNetV7Frequency(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_upernet_v7_localglobal":
        return MiTB2UPerNetV7LocalGlobal(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_upernet_v71_widecontext":
        return MiTB2UPerNetV71WideContext(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_upernet_v71_fadc_lite":
        return MiTB2UPerNetV71FADCLite(num_classes=num_classes, pretrained=pretrained)
    if name == "mit_b2_upernet_v71_dynamic_dict":
        return MiTB2UPerNetV71DynamicDictionary(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}. Available: {MODEL_NAMES}")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


__all__ = [
    "FCNResNet18",
    "DeepLabV3ResNet50",
    "SegFormerB0",
    "SegFormerB1",
    "SegFormerB2",
    "SegFormerB2BGAux",
    "MiTB2UPerNet",
    "MiTB2HierUPerNet",
    "MiTB2ForestRescueUPerNet",
    "MiTB2HRDALiteUPerNet",
    "MiTB2HRDADAFormer",
    "MiTB2OCRFPN",
    "MiTB2UPerOCR",
    "MiTB2MDC",
    "CNNPreMiTB2UPerNet",
    "MiTB2CNNStemUPerNet",
    "MiTB2CNNPyramidUPerNet",
    "MiTB2DetailUPerNet",
    "MiTB2LocalConvUPerNet",
    "MiTB2AdaptiveRFUPerNet",
    "MiTB2FrequencyUPerNet",
    "MiTB2SelectivePyramid",
    "MiTB2DynamicDictionaryUPerNet",
    "MiTB2ClassQueryUPerNet",
    "MiTB2WideContextUPerNet",
    "MiTB2UPerNetV7RF",
    "MiTB2UPerNetV7Frequency",
    "MiTB2UPerNetV7LocalGlobal",
    "MiTB2UPerNetV71WideContext",
    "MiTB2UPerNetV71FADCLite",
    "MiTB2UPerNetV71DynamicDictionary",
    "MODEL_NAMES",
    "MODEL_LABELS",
    "build_model",
    "count_parameters",
]
