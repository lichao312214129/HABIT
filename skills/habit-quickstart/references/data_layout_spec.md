# HABIT Data Layout Specification

Most HABIT commands assume a strict per-subject folder layout. This document
defines the canonical structure for raw input, preprocessed output, and habitat
output, so the agent can validate user data BEFORE writing configs.

## Canonical raw / preprocessed input

```
data_dir/
├── <subject_id_1>/
│   ├── images/
│   │   ├── <modality_1>/<modality_1>.nii.gz
│   │   ├── <modality_2>/<modality_2>.nii.gz
│   │   └── ...
│   └── masks/
│       ├── <modality_1>/<modality_1>.nii.gz
│       ├── <modality_2>/<modality_2>.nii.gz
│       └── ...
├── <subject_id_2>/
│   └── ...
```

### Rules

1. **Subject ID** = the immediate child folder name (`subject_id_1`, `subject_id_2`, ...). It must be unique. Avoid spaces, Chinese characters, and slashes. Use snake_case or camelCase.
2. **Modality folder name** is the same string the user will write in YAML configs (`images: [T1, T2, DWI, ADC]`). Case-sensitive.
3. **File name inside the modality folder** is conventionally `<modality>.nii.gz`, but HABIT picks up the first NIfTI/NRRD/MHA file found.
4. **Mask layout mirrors images layout** — one mask per modality. For purely whole-tumor work, you can supply a single mask under one modality folder; HABIT will resample it to others during registration if needed.
5. **Supported file extensions**: `.nii`, `.nii.gz`, `.nrrd`, `.mha`, `.mhd`. Prefer `.nii.gz` for everything.

## Preprocessed output (after `habit preprocess`)

```
out_dir/
├── processed_images/                 # FINAL output (always written)
│   ├── images/<subject>/<modality>/<modality>.nii.gz
│   └── masks/<subject>/<modality>/<modality>.nii.gz
├── n4_correction_01/                 # only if save_intermediate: true
├── resample_02/
├── registration_03/
└── preprocess.log
```

The `processed_images/` subdirectory becomes the `data_dir` (or `raw_img_folder`) input for downstream steps:
- `habit get-habitat` reads from `data_dir = <out_dir>/processed_images`
- `habit extract` reads from `raw_img_folder = <out_dir>/processed_images`

## Habitat output (after `habit get-habitat`)

```
out_dir/
├── <subject_1>/
│   ├── <subject_1>_supervoxel.nrrd            # only in two_step
│   ├── <subject_1>_habitats.nrrd              # always
│   └── <subject_1>_habitats_remapped.nrrd     # only in two_step
├── <subject_2>/
│   └── ...
├── habitats.csv                                # cohort-level habitat fractions
├── supervoxel2habitat_clustering_model.pkl     # two_step train mode only
├── mean_values_of_all_supervoxels_features.csv
└── visualizations/
    ├── habitat_clustering/
    └── supervoxel_clustering/
```

### `habitat_pattern` field

When passing this to `habit extract`:
- one_step output → `'*_habitats.nrrd'`
- two_step output → `'*_habitats_remapped.nrrd'` (recommended; consistent labels across subjects)

## Quick checks the agent should run before each command

| Before running | Check that exists |
|---|---|
| `habit preprocess` | `data_dir/<subject>/images/<mod>/` for at least one subject |
| `habit get-habitat` | `data_dir/<subject>/images/...` AND `masks/...` |
| `habit extract` | `raw_img_folder/<subject>/images/` AND `habitats_map_folder/<subject>/<subject>_habitats*.nrrd` |
| `habit model` | feature CSV exists, has `subject_id_col` and `label_col` |
| `habit compare` | each prediction CSV exists, has the columns named in `files_config` |

The validation scripts in `habit-quickstart/scripts/` automate these checks:
- `check_data_layout.py <data_dir>` — checks raw layout
- `validate_preprocess_output.py <out_dir>` — after preprocess
- `validate_habitat_output.py <out_dir>` — after get-habitat
- `inspect_feature_csv.py <csv>` — before ML
