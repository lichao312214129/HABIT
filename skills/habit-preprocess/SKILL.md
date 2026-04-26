---
name: habit-preprocess
description: Preprocess medical images (CT/MRI) for HABIT habitat analysis. Use when the user needs DICOM-to-NIfTI conversion, image resampling, multi-modal registration (e.g. align ADC to T2), N4 bias field correction for MRI, Z-score / histogram standardization, or CLAHE contrast enhancement. Triggers on phrases like "图像预处理", "重采样", "配准", "DICOM 转 NIfTI", "N4 校正", "标准化", "preprocess images", "register T2 and ADC".
---

# HABIT Image Preprocessing

Run the `habit preprocess` CLI command. Pipeline is fully driven by a YAML config — each step is optional and applied in the order listed.

## CLI

```bash
habit preprocess --config <your_config.yaml>
```

## Available Methods (in execution order)

| Method | Purpose | Required for |
|---|---|---|
| `dcm2nii` | DICOM → NIfTI conversion via dcm2niix | Raw DICOM data only |
| `n4_correction` | N4 bias field correction | MRI (T1/T2/FLAIR) |
| `resample` | Standardize voxel spacing | All modalities (recommended) |
| `registration` | Align moving images to a fixed image | Multi-modal analyses |
| `zscore_normalization` | Standardize intensity distribution | MRI/CT before ML |
| `histogram_standardization` | Nyúl method, percentile landmarks | Multi-center MRI |
| `adaptive_histogram_equalization` | CLAHE local contrast | Optional enhancement |

## Decision Tree

1. **Does the user have DICOM?**
   - Yes → start with `dcm2nii`. Need `dcm2niix.exe` path. Use `references/config_dcm2nii_template.yaml`.
   - No (already NIfTI) → skip `dcm2nii`.

2. **MRI or CT?**
   - MRI → strongly recommend `n4_correction` then `resample` then `zscore_normalization`.
   - CT → just `resample` (CT intensity is already standardized as HU).

3. **Multi-modal (e.g. T1 + T2 + DWI + ADC)?**
   - Yes → add `registration`. Pick one as `fixed_image` (usually T2 for brain/abdomen).

4. **Multi-center cohort?**
   - Yes → add `histogram_standardization` after N4.

## Standard Multi-Modal MRI Preprocessing

```yaml
data_dir: ./data/raw_images
out_dir: ./data/preprocessed_images

Preprocessing:
  n4_correction:
    images: [T1, T2, FLAIR]
    num_fitting_levels: 4

  resample:
    images: [T1, T2, DWI, ADC]
    target_spacing: [1.0, 1.0, 1.0]   # mm

  registration:
    images: [T2, T1, DWI, ADC]
    fixed_image: T2
    moving_images: [T1, DWI, ADC]
    type_of_transform: SyNRA           # ANTs deformable

  zscore_normalization:
    images: [T1, T2]
    only_inmask: false

processes: 4
random_state: 42
```

## Save Intermediate Steps (debugging)

```yaml
save_options:
  save_intermediate: true
  intermediate_steps: []   # empty = save every step
```

Output then becomes:
```
out_dir/
├── n4_correction_01/...
├── resample_02/...
├── registration_03/...
└── processed_images/   (final)
```

## Reference Templates

The full annotated templates live in the project (always direct the user to them):
- `config_templates/config_image_preprocessing_annotated.yaml` — full reference, every parameter explained
- `config_templates/config_image_preprocessing_dcm2nii_annotated.yaml` — DICOM conversion variant

A minimal copy is in `references/config_preprocess_minimal.yaml` for quick scaffolding.

## Common Pitfalls

1. **dcm2niix path on Windows** must use forward slashes or escaped backslashes:
   ```yaml
   dcm2niix_path: ./software/dcm2niix/dcm2niix.exe
   ```
2. **`fixed_image` must NOT be in `moving_images`** — it stays fixed.
3. **`only_inmask: true`** requires `mask_key` to be set.
4. **Mask files are NOT modified** by preprocessing (only images). Tell the user this — it's a frequent question.
5. **N4 + CT is wrong** — N4 is for MRI bias field. Don't apply to CT.
6. **Registration target_spacing** — apply `resample` BEFORE registration so all images share grid.

## Output Structure

```
out_dir/processed_images/
├── images/
│   └── sub001/
│       ├── T1/T1.nii.gz
│       └── T2/T2.nii.gz
└── masks/
    └── sub001/
        ├── T1/mask_T1.nii.gz
        └── T2/mask_T2.nii.gz
```

This structure is the **expected input** for `habit get-habitat` and `habit extract`.

## Verification

After running, check:
- All subjects have outputs in `out_dir/processed_images/`
- Open one case in ITK-SNAP and overlay registered modalities to confirm alignment
- Histogram should look reasonable (no extreme tails)

## Logs

Preprocessing log is saved to `out_dir/preprocess.log`. Always check it on errors.
