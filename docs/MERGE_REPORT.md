# Merge report

This repository was merged from the two supplied GitHub packaging outputs.

## Merge policy

- The full-history/full-log package was used as the canonical base.
- The smaller key-log package was used only to recover genuinely missing records.
- Conflicting documentation files were replaced with the full-record versions or regenerated.
- Redundant `logs/key_histories/` and `logs/key_experiments/` copies were not duplicated.
- Nested copies of an earlier generated `_github_release_lean` were removed.
- Source code that was not part of the final E090 code path but was accidentally included by filename matching was removed.

## Cleanup performed

- Nested prior-release log entries removed: **29**
- Non-final source files removed: **3**
- Missing unique logs recovered from the second package: **1**

Recovered unique files:

- `logs/raw/E048_v81_E037_trainval_ft.log`

Removed non-final source files:

- `e111_e090_100ep_common.py`
- `summarize_e086_e090.py`
- `train_e111_e090_100ep_noval.py`

## Final record counts

- Source files: **25**
- Experiment history CSVs: **70**
- Raw logs: **70**
- Text experiment artifacts: **172**

The repository intentionally excludes model weights, LoveDA imagery/masks, prediction rasters and submission archives.
