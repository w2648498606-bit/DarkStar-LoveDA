# Reproducibility

- Train is used for gradients.
- Val is independent for checkpoint/model selection in the main protocol.
- Hidden Test is not used to sweep every variant.
- Record initialization provenance and seeds.
- Submission labels use 0..6 directly.
