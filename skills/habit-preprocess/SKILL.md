---
name: habit-preprocess
description: Preprocess medical images (CT/MRI) for HABIT — DICOM-to-NIfTI conversion, resampling, multi-modal registration, N4 bias correction, z-score / histogram standardization, CLAHE. Use when the user mentions "图像预处理", "重采样", "配准", "DICOM 转 NIfTI", "N4", "标准化", "preprocess", "register", "bias field". Runs the `habit preprocess` CLI.
---

# HABIT Image Preprocessing

Drives the `habit preprocess` CLI command. The pipeline is fully YAML-driven —
each step is optional and applied in the order listed in the config.

## CLI

```bash
habit preprocess --config <your_config.yaml>
```

## Required Information

Before generating a config, confirm:

| Field | Notes |
|---|---|
| `data_dir` | path to subject folders (raw or DICOM) |
| `out_dir` | output root |
| Modality folder names | e.g. `T1, T2, DWI, ADC` |
| MRI or CT? | drives whether to use `n4_correction` |
| Multi-modal? | drives whether to use `registration` |
| Registration `fixed_image` | usually T2 |
| (DICOM only) `dcm2niix_path` | full path to `dcm2niix.exe` |

## Available methods (in execution order)

| Method | Purpose | Use for |
|---|---|---|
| `dcm2nii` | DICOM → NIfTI | raw DICOM input |
| `n4_correction` | bias field correction | MRI only |
| `resample` | unify voxel spacing | always |
| `registration` | align modalities | multi-modal |
| `zscore_normalization` | standardize intensity | MRI before ML (NOT for DCE phases or CT HU) |
| `histogram_standardization` | Nyúl method | multi-center MRI |
| `adaptive_histogram_equalization` | CLAHE | optional contrast boost |

## Decision tree

1. **DICOM input?** → use `dcm2nii` first. See `config_templates/skill_scaffolds/preprocess_dcm2nii.yaml`.
2. **MRI?** → enable `n4_correction` then `resample` then `zscore_normalization`.
3. **CT?** → just `resample`. NEVER N4 (CT has no bias field). See `config_templates/skill_scaffolds/preprocess_ct_only.yaml`.
4. **Multi-modal?** → add `registration` with one image as `fixed_image` (usually T2).
5. **DCE-MRI?** → use `config_templates/skill_scaffolds/preprocess_dce_mri.yaml`. **Do NOT z-score phases independently.**
6. **Multi-center cohort?** → add `histogram_standardization` after N4.

## Reference templates

Config index: `skills/CONFIG_SOURCES.md`.

| File | Use for |
|---|---|
| `config_templates/skill_scaffolds/preprocess_minimal.yaml` | starting scaffold (any) |
| `config_templates/skill_scaffolds/preprocess_dcm2nii.yaml` | DICOM → NIfTI |
| `config_templates/skill_scaffolds/preprocess_mri_multimodal.yaml` | T1+T2(+DWI/ADC) standard MRI |
| `config_templates/skill_scaffolds/preprocess_dce_mri.yaml` | DCE-MRI dynamic |
| `config_templates/skill_scaffolds/preprocess_ct_only.yaml` | CT (resample only) |

Full annotated reference: `config_templates/config_image_preprocessing_annotated.yaml`.

## Save intermediate outputs (debugging)

```yaml
save_options:
  save_intermediate: true
  intermediate_steps: []   # empty = save every step
```

Each step then writes to `<out_dir>/<stage>_NN/...` (e.g. `n4_correction_01/`).
The final aggregated output is always at `<out_dir>/processed_images/`.

## Validate output (MANDATORY after run)

```bash
python skills/habit-preprocess/scripts/validate_preprocess_output.py <out_dir> --modalities T1 T2 DWI ADC
```

This script checks every subject has all required modalities, no constant or
all-zero volumes, masks present and non-empty.

## Output structure

```
out_dir/processed_images/
├── images/<subject>/<modality>/<modality>.nii.gz
└── masks/<subject>/<modality>/<modality>.nii.gz
```

This is the expected input for `habit get-habitat` and `habit extract`.

## Common pitfalls

1. **Windows paths must use forward slashes** in YAML (`./software/dcm2niix.exe`).
2. **`fixed_image` must NOT appear in `moving_images`** — it stays fixed.
3. **`only_inmask: true`** requires a `mask_key:` value.
4. **Mask files are NOT modified** by preprocessing (only resampled if registration runs).
5. **N4 + CT is wrong** — N4 is MRI-specific.
6. **Apply `resample` BEFORE `registration`** so all images share the same grid.

For specific error messages, see `habit-troubleshoot/references/errors_preprocess.md`.

## Logs

Preprocessing log is at `<out_dir>/preprocess.log`. Always check it on errors.

## Next step

After preprocessing succeeds, the typical next skill is `habit-habitat-analysis`.
