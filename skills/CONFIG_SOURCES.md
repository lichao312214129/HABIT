# HABIT skill — configuration source map

Use these **repository-root paths** for YAML templates. Filenames encode the
**scenario** (e.g. `_demo`, `demo_elastix`) so agents do not confuse walkthrough
configs with production templates. See [`config/README_CONFIG.md`](../config/README_CONFIG.md)
for a full scenario index.

## Habitat (`habit get-habitat`)

| Use case | Reference |
|----------|-----------|
| Primary template (any `clustering_mode`; your paths) | `config/habitat/config_getting_habitat.yaml` |
| **Demo** train — two-step on bundled data | `config/habitat/config_habitat_two_step.yaml` |
| **Demo** train — one_step / direct_pooling | `config/habitat/config_habitat_one_step_raw_concat_train.yaml`, `config_habitat_direct_pooling.yaml` |
| **Demo** predict (saved pipeline) | `config/habitat/config_habitat_*_predict.yaml` |
| **Demo** manifest (images + masks layout) | `config/habitat/file_habitat.yaml` |

## Preprocess (`habit preprocess`)

| Use case | Reference |
|----------|-----------|
| Main MRI / multimodal template | `config/preprocessing/config_image_preprocessing.yaml` |
| DICOM → NIfTI | `config/preprocessing/config_image_preprocessing_dcm2nii.yaml` |
| DICOM sort / rename only (`habit sort-dicom`) | `config/dicom_sort/config_sort_dicom.yaml` |
| **Demo** preprocessing / habitat input | `demo_data/preprocessed/processed_images`; YAML: `config_preprocessing_demo.yaml`, `config_habitat_two_step.yaml`, `file_habitat_demo.yaml` |
| **Demo** manifest — DICOM folders (delay phases) | `config/preprocessing/files_preprocessing.yaml` |
| Example manifest — author T1/T2/DWI/ADC paths | `config/preprocessing/image_files.yaml` |

## Feature extraction (`habit extract`)

| Use case | Reference |
|----------|-----------|
| habitat + traditional / MSI / ITH (full template) | `config/feature_extraction/config_extract_features.yaml` |
| bundled demo dataset paths | `config/feature_extraction/config_extract_features_demo.yaml` |

## PyRadiomics parameter YAMLs (`habit extract` / `habit radiomics`)

| Profile | Reference |
|---------|-----------|
| bundled defaults (when `params_file` omitted) | `habit/resources/radiomics/` (`voxel`, `supervoxel`, `roi`, `habitat` presets) |
| repo copies (explicit override paths) | `config/radiomics/` |
| ROI / extract non-habitat | `parameter.yaml` (preset `roi`) |
| habitat maps extract | `parameter_habitat.yaml` (preset `habitat`) |
| minimal (~70 features) | `config/radiomics/parameter_basic.yaml` |
| LoG + Wavelet (~1000+ features) | `config/radiomics/parameter_with_filters.yaml` |
| voxel preset (`voxel_radiomics`) | `params_voxel_radiomics.yaml` — CT R3B12, 21 stable GLCM |
| supervoxel preset | `params_supervoxel_radiomics.yaml` — full texture classes |

**`params_file` is optional** for voxel/supervoxel radiomics, `habit radiomics` (`paths.params_file`), and `habit extract` (`params_file_of_*`). Omit to use bundled presets; override with a path or `@preset:voxel`.

For **`voxel_radiomics`**, list override names in the `method` parentheses; GLCM safety rules still apply (use voxel preset or explicit 21-feature GLCM list).

Example habitat config (minimal — all CT R3B12 defaults):

```yaml
feature_construction:
  voxel_level:
    method: concat(voxel_radiomics(T2))
    params: {}
```

Override `kernel_radius` only when needed (e.g. MRI → `1`):

```yaml
    method: concat(voxel_radiomics(T2, kernel_radius))
    params:
      kernel_radius: 1
```

For **`supervoxel_radiomics`**, `params_file` is also optional (bundled full-set preset). Set in `feature_construction.supervoxel_level.params`:
`supervoxel_batch` (default 64), `use_supervoxel_cext` (default auto),
and the same torch keys as above (inherit from `voxel_level.params` when omitted). Extraction
discretizes once on the union supervoxel mask (`sv_map > 0`), then runs per-label ROI matrices.
With `use_supervoxel_cext: auto`, habit uses the native C-extension batched matrix path when
`_sv_cmatrices` is built; otherwise it falls back to the prior Torch/PyRadiomics stacked-matrix
path. Torch feature evaluation still follows `use_torch_radiomics`. `kernel_radius` is **not** used
by `supervoxel_radiomics` (whole-ROI texture, not voxel kernel).

## Machine learning (`habit model` / `habit cv`)

| Use case | Reference |
|----------|-----------|
| holdout train + templates | `config/machine_learning/config_machine_learning.yaml` |
| k-fold | `config/machine_learning/config_machine_learning_kfold.yaml` |
| predict on new data | `config/machine_learning/config_machine_learning_predict.yaml` |
| **Demo** radiomics / clinical / k-fold CSV paths | `config/machine_learning/config_machine_learning_radiomics.yaml`, `config_machine_learning_clinical.yaml`, `config_machine_learning_kfold_demo.yaml` |

## Model comparison (`habit compare`)

| Use case | Reference |
|----------|-----------|
| template | `config/model_comparison/config_model_comparison.yaml` |
| demo paths | `config/model_comparison/config_model_comparison_demo.yaml` |

## Traditional radiomics (`habit radiomics`)

| Use case | Reference |
|----------|-----------|
| CLI template | `config/radiomics/config_traditional_radiomics.yaml` |

## Auxiliary CLIs (`habit icc`, `habit retest`, …)

| Use case | Reference |
|----------|-----------|
| ICC — full template (your paths, many groups) | `config/auxiliary/config_icc_analysis.yaml` |
| ICC — **demo** two CSVs under `demo_data` | `config/auxiliary/config_icc_demo.yaml` |
| test–retest remap | `config/auxiliary/config_test_retest.yaml` |
