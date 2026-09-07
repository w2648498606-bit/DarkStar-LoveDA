# Experiment summary

All available `outputs/experiments/*/history.csv` files are preserved under `logs/histories/`.
Historical experiment source wrappers are intentionally omitted from this lean source release.

Experiment directories indexed: **74**
History files copied: **70**

`EXPERIMENT_SUMMARY.csv` is retained as a raw experiment inventory. It does not encode protocol eligibility and includes diagnostic, non-independent, and no-Val runs; use the categorized sections below for model comparison.

## Independent validation results

Only runs whose validation samples were not used for gradient updates are listed here. These results are eligible for model comparison under the main protocol.

| ID | Run | Best Val mIoU | Best epoch | Last epoch |
|---|---|---:|---:|---:|
| E090 | `E090_e037_blv_lite` | 0.545243 | 5 | 12 |
| E042 | `E042_v76_hrda_lite` | 0.545099 | 5 | 12 |
| E090 | `E090_e037_blv_lite__previous_20260819_135410` | 0.545018 | 5 | 7 |
| E100 | `E100_e090lite_classaware_spatial_combined` | 0.544241 | 5 | 8 |
| E076 | `E076_mixstyle_only` | 0.543620 | 5 | 12 |
| E058 | `E058_barren_first_curriculum` | 0.542694 | 5 | 10 |
| E096 | `E096_e037_blv_repo_sigma4` | 0.541862 | 5 | 12 |
| E107 | `E107_e090lite_global_local_progressive` | 0.541532 | 5 | 12 |
| E097 | `E097_e037_blv_paper_sigma6` | 0.540856 | 5 | 12 |
| E082 | `E082_mixstyle_p030` | 0.537407 | 5 | 12 |
| E089 | `E089_e037_crop768_accum2` | 0.537283 | 4 | 12 |
| E087 | `E087_e037_boundary_aux` | 0.536628 | 5 | 12 |
| E044 | `E044_v78_augdg_color` | 0.536445 | 5 | 12 |
| E039 | `E039_v73_bg_bce` | 0.536298 | 5 | 12 |
| E037 | `E037_v72_multiscale_rare_ce_lovasz` | 0.536149 | 5 | 12 |
| E105 | `E105_e090lite_mlp_fusion` | 0.535924 | 5 | 12 |
| E081 | `E081_mixstyle_stage2_only` | 0.535649 | 4 | 12 |
| E106 | `E106_e090lite_progressive_fusion` | 0.535644 | 4 | 12 |
| E077 | `E077_balance_crossdomain_mixstyle` | 0.535620 | 5 | 12 |
| E086 | `E086_e037_hardpixel_ce_lovasz` | 0.535331 | 5 | 12 |
| E072 | `E072_balance_only` | 0.535152 | 5 | 12 |
| E080 | `E080_mixstyle_stage1_only` | 0.535003 | 5 | 12 |
| E041 | `E041_v74_forest_rescue` | 0.534484 | 5 | 12 |
| E088 | `E088_e037_ocr_lite` | 0.533289 | 4 | 12 |
| E074 | `E074_fosmix_only_mild` | 0.532849 | 5 | 12 |
| E038 | `E038_v73_rural_mixedforest` | 0.532375 | 5 | 12 |
| E085 | `E085_mixstyle_prototype_consistency` | 0.532366 | 4 | 12 |
| E035 | `E035_v72_multiscale_rare_ce` | 0.530609 | 5 | 12 |
| E083 | `E083_mixstyle_p070` | 0.530538 | 5 | 12 |
| E040 | `E040_v74_hier_bgfg` | 0.530239 | 5 | 12 |
| E084 | `E084_mixstyle_semantic_consistency` | 0.529991 | 6 | 12 |
| E036 | `E036_v72_multiscale_ce_lovasz` | 0.529723 | 5 | 12 |
| E073 | `E073_fosmix_only_beta005` | 0.525436 | 9 | 12 |
| E009 | `E009_mit_b2_upernet_10ep` | 0.524680 | 6 | 10 |
| E034 | `E034_v72_rare_basic_ce` | 0.523745 | 5 | 12 |
| E078 | `E078_balance_crossdomain_classmix` | 0.523220 | 5 | 12 |
| E079 | `E079_balance_fosmix_classmix__previous_20260817_125315` | 0.523080 | 5 | 12 |
| E079 | `E079_balance_fosmix_classmix` | 0.522976 | 5 | 12 |
| E075 | `E075_balance_fosmix_mild` | 0.522664 | 4 | 12 |
| E028 | `E028_v71_control_upernet` | 0.521899 | 1 | 5 |
| E033 | `E033_v72_multiscale_ce` | 0.520412 | 5 | 12 |
| E014 | `E014_mit_b2_cnnstem_upernet_10ep` | 0.520200 | 3 | 10 |
| E020 | `E020_mit_b2_selective_pyramid_10ep` | 0.519415 | 4 | 8 |
| E015 | `E015_mit_b2_cnnpyr_upernet_10ep` | 0.518697 | 5 | 10 |
| E019 | `E019_mit_b2_frequency_upernet_10ep` | 0.518691 | 2 | 10 |
| E029 | `E029_v71_widecontext` | 0.518262 | 1 | 5 |
| E013 | `E013_cnnpre_mit_b2_upernet_10ep` | 0.518137 | 7 | 10 |
| E043 | `E043_v77_hrda_daformer` | 0.517676 | 1 | 2 |
| E012 | `E012_mit_b2_uper_ocr_10ep` | 0.517584 | 5 | 10 |
| E029 | `E029A_v71_widecontext_warm` | 0.516626 | 2 | 2 |
| E025 | `E025_v7_rf_adapter` | 0.516517 | 5 | 8 |
| E027 | `E027_v7_localglobal_adapter` | 0.516325 | 6 | 8 |
| E026 | `E026_v7_frequency_adapter` | 0.515806 | 6 | 8 |
| E018 | `E018_mit_b2_adaptive_rf_upernet_10ep` | 0.515269 | 4 | 10 |
| E030 | `E030A_v71_fadc_warm` | 0.515102 | 2 | 2 |
| E024 | `E024_v7_control_upernet` | 0.514535 | 6 | 8 |
| E016 | `E016_mit_b2_detail_upernet_10ep` | 0.514407 | 5 | 10 |
| E007 | `E007_segformer_b2_20ep` | 0.514223 | 12 | 20 |
| E017 | `E017_mit_b2_localconv_upernet_10ep` | 0.512600 | 7 | 10 |
| E008 | `E008_segformer_b2_bgaux` | 0.512210 | 8 | 8 |
| E032 | `E032_v72_control_basic_ce` | 0.510874 | 5 | 12 |
| E011 | `E011_mit_b2_mdc_10ep` | 0.510218 | 8 | 10 |
| E010 | `E010_mit_b2_ocr_fpn_10ep` | 0.509618 | 2 | 10 |
| E006 | `E006_segformer_b2_3ep` | 0.492730 | 2 | 3 |
| E005 | `E005_segformer_b1_3ep` | 0.472244 | 3 | 3 |

Best independent validation result: **E090 = 0.545243** at epoch 5.

## Diagnostic / non-independent runs

The runs below are retained for protocol auditing and diagnosis. They are excluded from independent validation ranking and model selection.

| ID | Run | Recorded value | Protocol status |
|---|---|---:|---|
| E098 | `E098_e090lite_trainval_final__previous_20260819_163704` | 0.575365 | Train+Val fine-tuning; validation data participated in training. Val is no longer independent; excluded from model selection. |
| E098 | `E098_e090lite_trainval_final` | — | Train+Val final training; fixed final epoch rather than Val-based checkpoint selection. |
| E099 | `E099_e090lite_valonly_1epoch` | — | Val-only diagnostic fine-tuning; not an independent validation run. |
| E111 | `E111_e090lite_100ep_noval` | — | Long-budget run without comparable validation selection. |

E098's metadata records 2522 Train samples plus 1669 Val samples, for 4191 training samples in total. See [`logs/text_artifacts/E098_e090lite_trainval_final/E098_final_metadata.json`](../logs/text_artifacts/E098_e090lite_trainval_final/E098_final_metadata.json).

## Third-party checkpoint / initialization runs

Third-party checkpoint reproduction and third-party-derived initialization are reported separately from the main project ranking:

| ID | Run | Independent Val mIoU | Boundary |
|---|---|---:|---|
| E057 | `E057A_mtp_guided_mitb2` | 0.531731 | Project training run initialized from a third-party-derived encoder checkpoint; reported separately for provenance clarity. |

The existing release audit also labels E053 as a third-party checkpoint reproduction/inference. No corresponding E053 history or configuration artifact is present in the public package, so no E053 metric is ranked or independently verified here. See [`THIRD_PARTY_AUDIT.md`](THIRD_PARTY_AUDIT.md) and the corresponding configuration records for provenance details.

## Hidden Test result

The current project record reports **E090 = 0.525061** on Hidden Test. The submission was generated from E090's epoch-5 best checkpoint by the direct single-model prediction path, using clean logits without TTA or ensembling. See [`predict_e090_lite_direct_test.py`](../predict_e090_lite_direct_test.py) and [`logs/raw/E090_lite_direct_test.log`](../logs/raw/E090_lite_direct_test.log).

The public package contains the prediction-generation log, but it does not contain an official leaderboard export or score screenshot. Therefore, `0.525061` is presented as a project record rather than a score independently verifiable from the repository alone.

> Independent Val ranking is not equivalent to Hidden Test ranking. Hidden Test was not used to sweep every experiment.
