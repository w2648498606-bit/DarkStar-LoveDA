# LoveDA Semantic Segmentation Research

A lean-source, full-record release of a systematic LoveDA remote-sensing semantic-segmentation project.

The repository is organized around one principle:

> **Final model source stays concise; experimental evidence stays complete.**

## Current own best

```text
E090
MiT-B2 + UPerNet-style PPM/FPN
+ multi-scale training
+ rare-class crop
+ CE + 0.5 Lovasz
+ lightweight train-only frequency-aware logit perturbation

Val mIoU          0.545243
Hidden Test mIoU  0.525061
```

E090-lite is BLV-inspired lightweight logit perturbation, not a faithful BLV implementation.

## Repository policy

### Source code

Only the final E090 code path and the local modules it depends on are retained.

The public repository does **not** include nearly one hundred obsolete experiment wrappers.

Main entry points include:

```text
train_e090.py
predict_e090_lite_direct_test.py
train.py
e037_next_common.py
dg_sweep_common.py
datasets/
models/
utils/
losses.py
```

### Experimental records

All available experiment records from the packaged working tree are retained:

```text
logs/
├── histories/       # all available outputs/experiments/*/history.csv
├── raw/             # all available raw .log files
└── text_artifacts/  # compact txt/csv/json experiment metadata
```

A project-wide index is available at:

```text
docs/EXPERIMENT_SUMMARY.md
docs/EXPERIMENT_SUMMARY.csv
```

## Experimental trajectory

The project progressed through:

1. FCN / DeepLabV3 / SegFormer engineering baselines;
2. MiT-B2 + UPerNet baseline construction;
3. multi-scale / rare crop / Lovasz controlled ablations;
4. Urban/Rural and weak-class diagnostics;
5. context and HRDA-style experiments;
6. domain-generalization sweeps;
7. return to simple strong regularization;
8. BLV/mechanism/protocol audits;
9. spatial-prior experiments;
10. lightweight decoder redesign;
11. long-training-budget audit.

The purpose of retaining the full histories/logs is to make both successful and negative experiments auditable.

## Dataset

LoveDA itself is not redistributed.

Expected split sizes:

- Train: 2522
- Val: 1669
- Test: 1796

Label convention:

```text
raw 0   -> ignore 255
raw 1-7 -> train 0-6
```

Hidden-Test submission masks use class IDs `0..6` directly, with no `+1`.

## Reproducibility / provenance

Main protocol:

- Train is used for gradient updates.
- Val is independent for model/checkpoint selection.
- Hidden Test is not used to sweep every ablation.
- Initialization provenance and seeds should be reported explicitly.

Important provenance notes:

- E053 is an official MTP/RVSA LoveDA-finetuned checkpoint reproduction/inference and is **not** an original model contribution.
- Older MiT-B2 runs may inherit the E007 LoveDA-trained encoder initialization.
- Those runs should not be described as training from scratch.
- E090-lite is BLV-inspired, not faithful BLV.

## Before publishing

Review:

```text
docs/THIRD_PARTY_AUDIT.md
docs/SECRET_SCAN.json
docs/OPEN_SOURCE_CHECKLIST.md
docs/MERGE_REPORT.md
```

No model weights, LoveDA images, prediction masks or submission ZIPs are included in this repository.
