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
| **Demo** train — one_step / direct_pooling | `config/habitat/config_habitat_one_step.yaml`, `config_habitat_direct_pooling.yaml` |
| **Demo** predict (saved pipeline) | `config/habitat/config_habitat_*_predict.yaml` |
| **Demo** manifest (images + masks layout) | `config/habitat/file_habitat.yaml` |
| Legacy syntax-only example | `config/habitat/config_habitat_one_step_example.yaml` |

## Preprocess (`habit preprocess`)

| Use case | Reference |
|----------|-----------|
| Main MRI / multimodal template | `config/preprocessing/config_image_preprocessing.yaml` |
| DICOM → NIfTI | `config/preprocessing/config_image_preprocessing_dcm2nii.yaml` |
| DICOM sort / rename only | `config/preprocessing/config_image_preprocessing_sort_dicom.yaml` |
| **Demo** elastix registration (`demo_data`) | `config/preprocessing/config_preprocessing_demo_elastix.yaml` |
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
| package default (MRI-oriented starter) | `config/radiomics/parameter.yaml` |
| habitat maps (when distinct) | `config/radiomics/parameter_habitat.yaml` |
| minimal (~70 features) | `config/radiomics/parameter_basic.yaml` |
| LoG + Wavelet (~1000+ features) | `config/radiomics/parameter_with_filters.yaml` |
| voxel / supervoxel presets | `config/radiomics/params_voxel_radiomics.yaml`, `params_supervoxel_radiomics.yaml` |

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
