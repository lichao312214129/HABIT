# Configuration Files Guide

## Overview

Configs live under **`config/<module>/`**. Each file aims at **one primary
scenario** so templates do not duplicate each other without a clear name.
Shipped YAMLs should start with a **banner** (CLI, purpose, non-overlapping files).

See [`skills/CONFIG_SOURCES.md`](../skills/CONFIG_SOURCES.md) for the same paths in quick-reference tables.

Walkthrough files that reference **`demo_data/`** use explicit names
(`*_demo.yaml`, `config_preprocessing_demo_elastix.yaml`, `config_icc_demo.yaml`, …)
next to production templates in the same folder.

## Scenario index (distinct roles)

| File | Module | Scenario |
|------|--------|----------|
| `preprocessing/config_image_preprocessing.yaml` | preprocess | Main MRI pipeline template (your paths). |
| `preprocessing/config_image_preprocessing_dcm2nii.yaml` | preprocess | DICOM → NIfTI (+ optional follow-on steps). |
| `preprocessing/config_image_preprocessing_sort_dicom.yaml` | preprocess | DICOM sort/rename only. |
| `preprocessing/config_preprocessing_demo_elastix.yaml` | preprocess | **Demo**: elastix rigid registration on bundled NIfTI. |
| `preprocessing/files_preprocessing.yaml` | preprocess | **Demo manifest**: raw DICOM folders (delay phases). |
| `preprocessing/image_files.yaml` | preprocess | **Example manifest**: T1/T2/DWI/ADC paths (replace with yours). |
| `habitat/config_getting_habitat.yaml` | habitat | **Primary** habitat template (any `clustering_mode`). |
| `habitat/config_habitat_two_step.yaml` | habitat | **Demo train**: two-step on `demo_data` inputs. |
| `habitat/config_habitat_one_step.yaml` | habitat | One-step train (set paths locally). |
| `habitat/config_habitat_direct_pooling.yaml` | habitat | Direct-pooling train; often uses `file_habitat.yaml` as `data_dir`. |
| `habitat/config_habitat_*_predict.yaml` | habitat | Predict-only (saved pipeline + manifest paths). |
| `habitat/config_habitat_one_step_example.yaml` | habitat | Legacy **syntax** sample only. |
| `habitat/file_habitat.yaml` | habitat | **Demo manifest** of processed images + masks for `get-habitat`. |
| `feature_extraction/config_extract_features.yaml` | extract | Full extraction template (your paths). |
| `feature_extraction/config_extract_features_demo.yaml` | extract | **Demo** extraction into `demo_data` trees. |
| `machine_learning/config_machine_learning.yaml` | ML | Holdout train (generic). |
| `machine_learning/config_machine_learning_kfold.yaml` | ML | K-fold CV template. |
| `machine_learning/config_machine_learning_predict.yaml` | ML | **Only** predict config (load `.pkl` / joblib). |
| `machine_learning/config_machine_learning_radiomics.yaml` | ML | **Demo** training from radiomics CSV + demo splits. |
| `machine_learning/config_machine_learning_clinical.yaml` | ML | **Demo** training from clinical CSV + demo splits. |
| `machine_learning/config_machine_learning_kfold_demo.yaml` | ML | **Demo** k-fold on bundled CSV. |
| `model_comparison/config_model_comparison.yaml` | compare | Model comparison template (your prediction CSVs). |
| `model_comparison/config_model_comparison_demo.yaml` | compare | **Demo** compare on `demo_data/ml_data` outputs. |
| `auxiliary/config_icc_analysis.yaml` | auxiliary | ICC template (multi-group; your absolute paths). |
| `auxiliary/config_icc_demo.yaml` | auxiliary | **Demo** ICC on two bundled feature CSVs. |
| `auxiliary/config_test_retest.yaml` | auxiliary | Test–retest remap + tables (demo paths). |
| `radiomics/config_traditional_radiomics.yaml` | radiomics | Classical PyRadiomics CLI template. |

Shared PyRadiomics parameter YAMLs: `radiomics/parameter.yaml`, `radiomics/parameter_habitat.yaml`,
`radiomics/parameter_basic.yaml`, `radiomics/parameter_with_filters.yaml`,
`radiomics/params_voxel_radiomics.yaml`, `radiomics/params_supervoxel_radiomics.yaml`.

## How to use

Open the YAML for the command you are running; section headers and comments
explain options. Adjust `data_dir`, `out_dir`, and paths to match your machine.

## YAML format specification

**Important formatting rules**:

1. **Indentation**:
   - Use **2 spaces** (do not use Tab)
   - Keep hierarchy clear

2. **Colon**:
   - Space **required** after colon: `key: value`
   - Empty values can be omitted or set to `null`

3. **Lists**:
   - Start with `-` symbol
   - Space **required** after `-`

4. **Comments**:
   - Start with `#` symbol
   - Can be on separate line or at line end

5. **Strings**:
   - Usually no quotes needed
   - Use quotes when containing special characters

### Example

```yaml
# Correct format
data_dir: ./data
output: ./results
settings:
  key1: value1
  key2: value2
  list:
    - item1
    - item2
```

## Configuration usage (CLI)

```bash
habit get-habitat --config config/habitat/config_getting_habitat.yaml
habit get-habitat -c config/habitat/config_getting_habitat.yaml
```

## Related documentation

- **Main README**: [README.md](../README.md) / [README_en.md](../README_en.md)
- **Habitat Analysis**: [docs/source/user_guide/habitat_segmentation_zh.rst](../docs/source/user_guide/habitat_segmentation_zh.rst)
- **Machine Learning**: `docs/source/user_guide/machine_learning_modeling_zh.rst`

---

*Last Updated: 2026-05-15*
