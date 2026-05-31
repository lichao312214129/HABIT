# Configuration Files Guide

## YAML parameter reference (authoritative)

**Full field documentation** (types, **schema defaults**, nested keys, path resolution, habitat `config_hash`, etc.) is in the Sphinx page built from:

- [`docs/source/configuration_zh.rst`](../docs/source/configuration_zh.rst) → **「配置参考」** in the HTML docs (`configuration_zh.html` after `make html`)

This file is a **template index** and habitat-specific notes (`#%%` markers, Stage-1 parallelism). Prefer the configuration reference for per-key semantics; user guides link there for YAML details.

## Overview

Configs live under **`config/<module>/`**. Each file aims at **one primary
scenario** so templates do not duplicate each other without a clear name.
Shipped YAMLs should start with a **banner** (CLI, purpose, non-overlapping files).

See [`skills/CONFIG_SOURCES.md`](../skills/CONFIG_SOURCES.md) for the same paths in quick-reference tables.

Walkthrough files that reference **`demo_data/`** use explicit names
(`*_demo.yaml`, `config_preprocessing_demo.yaml`, `config_icc_demo.yaml`, …)
next to production templates in the same folder.

### Running on your data

Each scenario has **one script** under `tests/<module>/`. Open the script — the
config path is in the docstring. Edit that YAML (`#%%====` path blocks), then:

```bash
python tests/habitat/habitat_one_step_voxel_radiomics_train.py
python tests/habitat/habitat_two_step_raw_concat_train.py
python tests/preprocessing/preprocess_resample02.py
python tests/integration/workflow_preprocess_to_compare.py
```

Regenerate scripts after template changes: `python scripts/generate_manual_debug_tests.py`

## Scenario index (distinct roles)

| File | Module | Scenario |
|------|--------|----------|
| `preprocessing/config_image_preprocessing.yaml` | preprocess | Main MRI pipeline template (your paths). |
| `preprocessing/config_image_preprocessing_dcm2nii.yaml` | preprocess | resample_02 demo resample (T1/T2); dcm2nii commented for production. |
| `dicom_sort/config_sort_dicom.yaml` | sort-dicom | Production DICOM only (not resample_02). |
| `preprocessing/config_preprocessing_demo.yaml` | preprocess | **Demo (default)**: SimpleITK rigid registration on `.cursor/test/resample_02` (T1/T2). |
| `preprocessing/config_preprocessing_demo_elastix.yaml` | preprocess | **Demo (optional)**: elastix CLI registration; requires elastix/transformix on PATH. |
| `preprocessing/files_preprocessing.yaml` | preprocess | **Demo manifest**: resample_02 NIfTI paths (T1, T2). |
| `preprocessing/image_files.yaml` | preprocess | Same resample_02 manifest as `files_preprocessing.yaml` (legacy name). |
| `habitat/config_getting_habitat.yaml` | habitat | **Primary** habitat template (any `clustering_mode`). |
| `habitat/config_habitat_two_step.yaml` | habitat | **Demo train**: two-step on resample_02 inputs. |
| `habitat/config_habitat_one_step_*_{train,predict}.yaml` | habitat | One-step demo configs per feature recipe (`raw_concat`, `voxel_radiomics`, …). |
| `habitat/config_habitat_direct_pooling.yaml` | habitat | Direct-pooling train; often uses `file_habitat.yaml` as `data_dir`. |
| `habitat/config_habitat_*_predict.yaml` | habitat | Predict-only (saved pipeline + manifest paths). |
| `habitat/file_habitat.yaml` | habitat | **Demo manifest** of processed images + masks for `get-habitat`. |
| `feature_extraction/config_extract_features.yaml` | extract | Full extraction template (your paths). |
| `feature_extraction/config_extract_features_demo.yaml` | extract | **Demo** all standard `feature_types` (resample_02). |
| `machine_learning/config_machine_learning.yaml` | ML | Holdout train (generic). |
| `machine_learning/config_machine_learning_kfold.yaml` | ML | K-fold CV template. |
| `machine_learning/config_machine_learning_predict.yaml` | ML | **Only** predict config (load `.pkl` / joblib). |
| `machine_learning/config_machine_learning_radiomics.yaml` | ML | **Demo** single-table radiomics CSV + demo splits. |
| `machine_learning/config_machine_learning_clinical.yaml` | ML | **Demo** single-table clinical CSV + demo splits. |
| `machine_learning/config_machine_learning_multi_tables_demo.yaml` | ML | **Demo** multi-table merge (clinical + radiomics). |
| `machine_learning/config_machine_learning_kfold_demo.yaml` | ML | **Demo** k-fold on bundled CSV. |
| `model_comparison/config_model_comparison.yaml` | compare | Model comparison template (your prediction CSVs). |
| `model_comparison/config_model_comparison_demo.yaml` | compare | **Demo** compare on `demo_data/ml_data` outputs. |
| `auxiliary/config_icc_analysis.yaml` | auxiliary | ICC template (multi-group; your absolute paths). |
| `auxiliary/config_icc_demo.yaml` | auxiliary | **Demo** ICC on two bundled feature CSVs. |
| `auxiliary/config_test_retest.yaml` | auxiliary | Test–retest remap + tables (demo paths). |
| `radiomics/config_traditional_radiomics.yaml` | radiomics | Classical PyRadiomics CLI template. |

Shared PyRadiomics parameter YAMLs: `radiomics/parameter.yaml`, `radiomics/parameter_habitat.yaml`,
`radiomics/parameter_basic.yaml`, `radiomics/parameter_with_filters.yaml`,
`radiomics/params_voxel_radiomics.yaml` (voxel `voxel_radiomics` preset — explicit 21-feature GLCM list),
`radiomics/params_supervoxel_radiomics.yaml`.

For `voxel_radiomics` in habitat, use `params_voxel_radiomics.yaml` for GLCM: bare `glcm:` enables all
24 features and MCC/Imc1/Imc2 crash on small kernels. HABIT defaults unrestricted GLCM to the same
21 stable features when `glcm` is not explicitly listed. `kernelRadius`, `voxelBatch`, `useTorchRadiomics`,
`torchGpus`, `torchGpuCount`, `torchDevice`, and `torchDtype` belong in
`FeatureConstruction.voxel_level.params` (not in the PyRadiomics parameter YAML).
They are forwarded to the extractor even when omitted from the `method` expression.
Default `voxelBatch` is `1000`; use `-1` for no batching. Default `useTorchRadiomics` is `auto`.
Default `torchDtype` is `float32`.

For `supervoxel_radiomics` in habitat, use `params_supervoxel_radiomics.yaml` for feature classes
(firstorder, glcm, …). Habit-specific keys belong in
`FeatureConstruction.supervoxel_level.params` (not in the PyRadiomics parameter YAML):
`supervoxelBatch` (default `64`), `useSupervoxelCext` (default `auto`), `useTorchRadiomics`,
`torchGpus`, `torchGpuCount`, `torchDevice`, and `torchDtype`. When omitted under
`supervoxel_level`, torch keys inherit from `voxel_level.params`. Extraction uses
**union-mask binning** (one PyRadiomics discretization on all supervoxel labels), then
per-label ROI matrices. With `useSupervoxelCext: auto`, habit tries the native C-extension
batched matrix path when `_sv_cmatrices` is built; otherwise it falls back to the prior
Torch/PyRadiomics stacked-matrix path. With `useSupervoxelCext: false`, habit always uses
the Torch/PyRadiomics stacked-matrix path even when the C extension is built. Set this key
under `FeatureConstruction.supervoxel_level.params` (not in the PyRadiomics `params_file`).
`kernelRadius` applies to `voxel_radiomics` only, not `supervoxel_radiomics`.

### Habitat YAML comment markers

Must-edit parameters are wrapped in **`#%%================================================================================` … `#%%================================================================================` pairs**. Everything else is normal comments (no box).

| Marker | Meaning |
|--------|---------|
| `#%%================================================================================` (opening) | Start of must-edit block |
| `#%%================================================================================` (closing) | End of must-edit block |
| `CHANGE_ME` | Placeholder — replace before first run |

Example:

```yaml
#%%================================================================================
data_dir: CHANGE_ME
out_dir: CHANGE_ME
#%%================================================================================
```

Train templates may use a **second pair** for study-specific feature settings (e.g. `method` + `timestamps` in `config_getting_habitat.yaml`).

Quick audit:

```bash
rg "#%%====" config/habitat/
rg "CHANGE_ME" config/habitat/
```

### Habitat Stage-1 parallelism (top-level YAML keys)

These keys live at the **root** of habitat configs (see `config/habitat/config_getting_habitat.yaml`):

| Key | Default | Role |
|-----|---------|------|
| `processes` | `2` | Max concurrent subject workers in Stage 1; peak RAM ≈ `processes × per-subject memory`. |
| `cap_processes_to_gpu_pool` | `false` | When Torch CUDA radiomics is active: `true` caps workers to `len(torchGpus)` (one slot per GPU); `false` keeps full `processes` and shares GPUs via `gpuSlotIndex` (better CPU use on 1-GPU / many-CPU hosts; more GPU contention risk). |
| `individual_subject_timeout_sec` | `900` | Per-subject wall clock in Stage 1; `null` disables. Positive value forces spawn even when `processes: 1`. |
| `individual_subject_parallel_mode` | `persistent` | `persistent` (long-lived workers) or `isolated` (spawn per subject). |
| `individual_subject_auto_retry_rounds` | `2` | Same-run auto-retry for Stage-1 failures; `0` disables. |
| `oom_backoff` | `true` | Reduce workers after `MemoryError` when enabled. |
| `resume` / `checkpoint_dir` / `retry_failed_subjects` | — | Checkpoint resume; see habitat user guide. |
| `strict_checkpoint_hash` | `true` | With `resume: true`: when `true` (default), incompatible manifest hash or `run_mode` raises `CheckpointConfigHashError` instead of discarding checkpoint and starting fresh. Set `false` to auto-discard and restart. Stage-1-compatible legacy hash migration still resumes. |

Not in checkpoint `config_hash` (safe to change on resume): `processes`, `cap_processes_to_gpu_pool`, `strict_checkpoint_hash`, timeout, parallel mode, retry flags, `plot_curves`, `out_dir`, group-stage blocks.

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

- **YAML parameters (all modules)**: [docs/source/configuration_zh.rst](../docs/source/configuration_zh.rst) — build with `docs/make html`
- **Main README**: [README.md](../README.md) / [README_en.md](../README_en.md)
- **Habitat workflow** (checkpoint, clustering modes): [docs/source/user_guide/habitat_segmentation_zh.rst](../docs/source/user_guide/habitat_segmentation_zh.rst)
- **Machine Learning**: [docs/source/user_guide/machine_learning_modeling_zh.rst](../docs/source/user_guide/machine_learning_modeling_zh.rst)

---

*Last Updated: 2026-05-27*
