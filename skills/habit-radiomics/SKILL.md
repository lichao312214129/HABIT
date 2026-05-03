---
name: habit-radiomics
description: Extract traditional PyRadiomics features (firstorder, shape, GLCM, GLRLM, GLSZM, NGTDM, GLDM) from medical images at the whole-tumor ROI level — no habitat segmentation. Use when the user wants classical radiomics features without doing habitat analysis. Triggers on "传统影像组学", "PyRadiomics", "radiomics features", "shape features", "GLCM". Runs `habit radiomics`.
---

# HABIT Traditional Radiomics

Extract whole-tumor PyRadiomics features (no habitat clustering involved). The
classical radiomics workflow — one row per subject, one feature per column.

If the user wants **habitat-based** features, redirect to `habit-feature-extraction`.

## CLI

```bash
habit radiomics --config <config_traditional_radiomics.yaml>
```

## Required Information

| Field | Stop if missing |
|---|---|
| `paths.params_file` | yes — PyRadiomics parameter YAML |
| `paths.images_folder` | yes |
| `paths.out_dir` | yes |
| `processing.process_image_types` | yes — modality folder names |

## PyRadiomics parameter file

Three options for the params file:

| File | Use |
|---|---|
| `config_templates/skill_scaffolds/pyradiomics_parameter_example.yaml` | generic full set |
| `config_templates/skill_scaffolds/pyradiomics_parameter_basic.yaml` | minimal (~70 features) |
| `config_templates/skill_scaffolds/pyradiomics_parameter_with_filters.yaml` | full with LoG+Wavelet (~1500 features) |

Choosing guide: `references/parameter_choice_guide.md`.

If the user has no params file, use the basic template as a starting point.

## Standard config

```yaml
paths:
  params_file: ./config/parameter.yaml
  images_folder: ./data/preprocessed_images
  out_dir: ./results/radiomics

processing:
  n_processes: 4
  save_every_n_files: 5
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

## Reference templates

Config index: `skills/CONFIG_SOURCES.md`.

| File | Use |
|---|---|
| `config_templates/skill_scaffolds/radiomics_minimal.yaml` | scaffold |
| `config_templates/skill_scaffolds/pyradiomics_parameter_example.yaml` | starter PyRadiomics params |
| `references/parameter_choice_guide.md` | how to pick filters and feature classes |

Full annotated reference: `config_templates/config_traditional_radiomics_annotated.yaml`.

## Validate output (after run)

```bash
python skills/habit-feature-extraction/scripts/inspect_feature_csv.py \
  <out_dir>/radiomics_features_combined_*.csv --subject-id-col subjID
```

(The same CSV inspector used for habitat features works here too.)

## Output files

With default settings (`add_timestamp: true`, both export options on):

```
out_dir/
├── radiomics_features_T1_<timestamp>.csv      # per-modality
├── radiomics_features_T2_<timestamp>.csv
├── radiomics_features_DWI_<timestamp>.csv
├── radiomics_features_ADC_<timestamp>.csv
├── radiomics_features_combined_<timestamp>.csv  # all modalities merged
└── extraction.log
```

The combined CSV is the typical input for `habit model`.

## Differences from `habit extract`

| Aspect | `habit radiomics` | `habit extract` |
|---|---|---|
| Operates on | Whole-tumor ROI | Habitat sub-regions |
| Requires | Just images + masks | Habitat maps too |
| Output features | Per modality | Per habitat + traditional + MSI + ITH |
| Use case | Classical radiomics | Habitat-based study |

If the user has habitat maps and wants comprehensive features, prefer
`habit extract` (which can include traditional whole-tumor features as one
of its options).

## Common pitfalls

1. **`process_image_types` mismatch** — names must exactly match folder names. Case-sensitive.
2. **PyRadiomics fails on small ROIs** — ROIs <30 voxels can fail GLCM/GLRLM. Increase `binWidth` or check ROI size.
3. **Resampling mismatch** — `resampledPixelSpacing` in `parameter.yaml` may resample again. Comment out if already resampled in preprocessing.
4. **Memory blow-up with Wavelet + LoG** — each filter multiplies features ~8-10×. Start with `Original: {}` only for testing.
5. **Mask label value** — default is label=1. If multi-label mask, specify `label: 1` in `parameter.yaml`.

For more, see `habit-troubleshoot/references/errors_extraction.md`.

## Verification

After running:
- Open one CSV — should have ~100-1500 columns depending on params
- First column = subject ID
- No NaN columns (NaN = failed extraction → check log)

## Next step

After radiomics extraction, proceed to `habit-machine-learning` to train models.
