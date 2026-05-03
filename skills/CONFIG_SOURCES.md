# HABIT skill — configuration source map

Agent skills (`skills/*/SKILL.md`) should **not** duplicate long parameter docs.
Use this table: **scaffold** (short example) → **annotated** (full comments) → **standard** (concise daily config under `config/`).

All scaffold paths are relative to the **repository root**.

## Habitat (`habit get-habitat`)

| Use case | Scaffold | Annotated | Standard |
|----------|----------|-----------|----------|
| one_step minimal | `config_templates/skill_scaffolds/habitat_one_step_minimal.yaml` | `config_templates/config_getting_habitat_annotated.yaml` | `config/config_getting_habitat.yaml` |
| two_step minimal | `config_templates/skill_scaffolds/habitat_two_step_minimal.yaml` | same | same |
| DCE kinetic | `config_templates/skill_scaffolds/habitat_kinetic_dce.yaml` | same | same |
| voxel radiomics | `config_templates/skill_scaffolds/habitat_voxel_radiomics.yaml` | same | same |
| Legacy one-step syntax sample | `config_templates/config_habitat_one_step_example.yaml` | same | — |

## Preprocess (`habit preprocess`)

| Use case | Scaffold | Annotated | Standard |
|----------|----------|-----------|----------|
| minimal shell | `config_templates/skill_scaffolds/preprocess_minimal.yaml` | `config_templates/config_image_preprocessing_annotated.yaml` | `config/config_image_preprocessing.yaml` |
| DICOM → NIfTI | `config_templates/skill_scaffolds/preprocess_dcm2nii.yaml` | `config_templates/config_image_preprocessing_dcm2nii_annotated.yaml` | `config/config_image_preprocessing_dcm2nii.yaml` |
| MRI multimodal | `config_templates/skill_scaffolds/preprocess_mri_multimodal.yaml` | `config_templates/config_image_preprocessing_annotated.yaml` | `config/config_image_preprocessing.yaml` |
| DCE-MRI | `config_templates/skill_scaffolds/preprocess_dce_mri.yaml` | same | same |
| CT only | `config_templates/skill_scaffolds/preprocess_ct_only.yaml` | same | same |

## Feature extraction (`habit extract`)

| Use case | Scaffold | Annotated | Standard |
|----------|----------|-----------|----------|
| minimal | `config_templates/skill_scaffolds/extract_features_minimal.yaml` | `config_templates/config_extract_features_annotated.yaml` | `config/config_extract_features.yaml` |
| publication set | `config_templates/skill_scaffolds/extract_features_publication.yaml` | same | same |
| MSI + ITH only | `config_templates/skill_scaffolds/extract_features_msi_ith_only.yaml` | same | same |

## PyRadiomics parameter YAMLs (shared)

| Profile | Scaffold |
|---------|----------|
| basic (~70 features) | `config_templates/skill_scaffolds/pyradiomics_parameter_basic.yaml` |
| + LoG + Wavelet | `config_templates/skill_scaffolds/pyradiomics_parameter_with_filters.yaml` |
| full-class example | `config_templates/skill_scaffolds/pyradiomics_parameter_example.yaml` |

Shipped defaults in the package (often referenced from configs): `config/parameter.yaml`, `config/parameter_habitat.yaml`.

## Machine learning (`habit model` / `habit cv`)

| Use case | Scaffold | Annotated | Standard |
|----------|----------|-----------|----------|
| train minimal | `config_templates/skill_scaffolds/ml_train_minimal.yaml` | `config_templates/config_machine_learning_annotated.yaml` | `config/config_machine_learning.yaml` |
| predict | `config_templates/skill_scaffolds/ml_predict_minimal.yaml` | same | — |
| k-fold | `config_templates/skill_scaffolds/ml_kfold_minimal.yaml` | `config_templates/config_machine_learning_kfold_annotated.yaml` | `config/config_machine_learning_kfold.yaml` |
| full pipeline (radiomics) | `config_templates/skill_scaffolds/ml_pipeline_radiomics_std.yaml` | `config_templates/config_machine_learning_annotated.yaml` | `config/config_machine_learning.yaml` |
| high-dim pipeline | `config_templates/skill_scaffolds/ml_pipeline_high_dim.yaml` | same | same |

## Model comparison (`habit compare`)

| Use case | Scaffold | Annotated | Standard |
|----------|----------|-----------|----------|
| minimal | `config_templates/skill_scaffolds/model_comparison_minimal.yaml` | `config_templates/config_model_comparison_annotated.yaml` | `config/config_model_comparison.yaml` |
| two models | `config_templates/skill_scaffolds/model_comparison_two_models.yaml` | same | same |
| three models | `config_templates/skill_scaffolds/model_comparison_three_models.yaml` | same | same |

## Traditional radiomics (`habit radiomics`)

| Use case | Scaffold | Annotated | Standard |
|----------|----------|-----------|----------|
| minimal | `config_templates/skill_scaffolds/radiomics_minimal.yaml` | `config_templates/config_traditional_radiomics_annotated.yaml` | `config/config_traditional_radiomics.yaml` |

## Auxiliary CLIs (`habit icc`, `habit retest`, …)

| Use case | Scaffold | Annotated | Standard |
|----------|----------|-----------|----------|
| ICC | `config_templates/skill_scaffolds/auxiliary_icc_minimal.yaml` | `config_templates/config_icc_analysis_annotated.yaml` | `config/config_icc_analysis.yaml` |
| test–retest remap | `config_templates/skill_scaffolds/auxiliary_test_retest_minimal.yaml` | — | `demo_data/config_test_retest.yaml` |
