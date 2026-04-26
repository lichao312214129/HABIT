---
name: habit-radiomics
description: Extract traditional PyRadiomics features from medical images at the whole-tumor ROI level (no habitat segmentation needed). Use when the user wants classical radiomics features (first-order, shape, GLCM, GLRLM, GLSZM, NGTDM, GLDM) without doing habitat analysis. Triggers on phrases like "传统影像组学", "PyRadiomics", "radiomics features", "提取影像组学特征", "shape and texture features", "first-order features".
---

# HABIT Traditional Radiomics

Extract whole-tumor PyRadiomics features (no habitat clustering involved). This is the **classical** radiomics workflow — one row per subject, one feature per column.

If the user wants **habitat-based** features (per-region, MSI, ITH), redirect to the `habit-feature-extraction` skill instead.

## CLI

```bash
habit radiomics --config <config_traditional_radiomics.yaml>
```

## Required Inputs

1. **PyRadiomics parameter file** (`parameter.yaml`) — defines what features to extract
2. **Images folder** — preprocessed NIfTI images with masks
3. **Output directory**

## PyRadiomics Parameter File

User must provide a `parameter.yaml` (PyRadiomics standard format). Example:

```yaml
imageType:
  Original: {}
  LoG:                       # Laplacian of Gaussian (multi-scale)
    sigma: [1.0, 2.0, 3.0]
  Wavelet: {}                # 8 wavelet decompositions

featureClass:
  firstorder:                # 18 features: mean, energy, entropy, etc.
  shape:                     # 14 features: volume, sphericity, etc.
  glcm:                      # 24 texture features
  glrlm:                     # 16 run-length features
  glszm:                     # 16 size-zone features
  ngtdm:                     # 5 neighborhood features
  gldm:                      # 14 dependence features

setting:
  binWidth: 25               # for fixed bin width discretization
  resampledPixelSpacing: [1, 1, 1]
  interpolator: sitkBSpline
  normalize: false           # set true if not pre-normalized
```

If user doesn't have a `parameter.yaml`, point them to:
- PyRadiomics docs: https://pyradiomics.readthedocs.io/en/latest/customization.html
- Or use the simpler example above as starting point.

## Standard Config

```yaml
paths:
  params_file: ./config/parameter.yaml
  images_folder: ./data/preprocessed_images
  out_dir: ./results/radiomics

processing:
  n_processes: 4
  save_every_n_files: 5            # checkpoint frequency
  process_image_types:
    - T1
    - T2
    - DWI
    - ADC

export:
  export_by_image_type: true       # one CSV per modality
  export_combined: true            # merged CSV with all modalities (prefixed)
  export_format: csv               # csv | json | pickle
  add_timestamp: true

logging:
  level: INFO
  console_output: true
  file_output: true
```

## Reference Template

- Full annotated: `config_templates/config_traditional_radiomics_annotated.yaml`
- Minimal scaffold: `references/config_radiomics_minimal.yaml`

## Output Files

With default settings (`add_timestamp: true`, both export options on):

```
out_dir/
├── radiomics_features_T1_2026-04-26_10-30.csv      # per-modality
├── radiomics_features_T2_2026-04-26_10-30.csv
├── radiomics_features_DWI_2026-04-26_10-30.csv
├── radiomics_features_ADC_2026-04-26_10-30.csv
├── radiomics_features_combined_2026-04-26_10-30.csv  # all modalities merged
└── extraction.log
```

The combined CSV is the typical input for `habit model`.

## Differences from `habit extract`

| Aspect | `habit radiomics` | `habit extract` |
|---|---|---|
| Operates on | Whole-tumor ROI | Habitat sub-regions |
| Requires | Just images + masks | Habitat maps (.nrrd) too |
| Output features | Per modality | Per habitat + traditional + MSI + ITH |
| Use case | Classical radiomics study | Habitat-based study |

If the user has habitat maps and wants the most comprehensive analysis, prefer `habit extract` (which also extracts traditional whole-tumor features as one of its options).

## Common Pitfalls

1. **`process_image_types` mismatch** — names must exactly match folder names under `images/<subject>/<modality>/`. Case-sensitive.
2. **PyRadiomics fails on small ROIs** — ROIs <30 voxels can fail GLCM/GLRLM. Increase `binWidth` or check ROI size.
3. **Resampling mismatch** — `resampledPixelSpacing` in `parameter.yaml` may resample again. Disable if images are already resampled in preprocessing.
4. **Memory blow-up with Wavelet + LoG** — each filter multiplies features by 8-10×. Start with `Original: {}` only for testing.
5. **Mask label value** — by default uses label=1. If user has multi-label mask (e.g. tumor=1, edema=2), specify in parameter.yaml: `label: 1`.

## Verification

After running:
- Open one of the CSVs — should have ~100-1500 columns depending on parameter file
- First column = subject ID
- No NaN columns (NaN means failed extraction → check log)
