# Recipe — CT Classical Radiomics (No Habitat)

For users who just want a traditional radiomics study on CT data: extract
PyRadiomics features at the whole-tumor ROI level, then train a classifier.
No habitat clustering involved.

## When to use this (vs habitat workflow)

- Single-modality CT data
- Whole-tumor analysis is sufficient (no need to discover sub-regions)
- You want the simpler, faster pipeline (skip habitat step entirely)

If the user wants sub-region analysis, redirect them to
`recipe_mri_habitat_full.md` and adapt the modality list.

## Required inputs

| # | Question | Stop if missing |
|---|---|---|
| 1 | Path to CT data root | yes |
| 2 | CT modality folder name (e.g. `CT`, `arterial_CT`) | yes |
| 3 | Output root | yes |
| 4 | Path to PyRadiomics `parameter.yaml` (or accept default basic params) | no — can default |
| 5 | Path to clinical labels CSV | yes (for ML) |
| 6 | Subject ID column + label column in that CSV | yes |

## CLI sequence

```bash
# 0) Sanity checks
python skills/habit-quickstart/scripts/check_environment.py
python skills/habit-quickstart/scripts/check_data_layout.py <DATA_ROOT> --modalities CT

# 1) Preprocess (CT-specific: resample only)
habit preprocess --config configs/01_ct_preprocess.yaml
python skills/habit-preprocess/scripts/validate_preprocess_output.py <PREPROCESS_OUT> --modalities CT

# 2) Traditional radiomics extraction (NO habitat step)
habit radiomics --config configs/02_ct_radiomics.yaml

# 3) Inspect the output CSV before ML
python skills/habit-feature-extraction/scripts/inspect_feature_csv.py \
  <RADIOMICS_OUT>/radiomics_features_combined_*.csv \
  --subject-id-col subjID

# 4) Merge features with clinical labels
habit merge-csv <RADIOMICS_OUT>/radiomics_features_combined_*.csv <CLINICAL_CSV> \
  -o configs/ml_input.csv --index-col subjID

# 5) Train ML
habit model --config configs/03_ct_ml.yaml --mode train
```

## Configs to generate

| Step | Template |
|---|---|
| 01_ct_preprocess | `config_templates/skill_scaffolds/preprocess_ct_only.yaml` |
| 02_ct_radiomics | `config_templates/skill_scaffolds/radiomics_minimal.yaml` |
| 03_ct_ml | `config_templates/skill_scaffolds/ml_pipeline_radiomics_std.yaml` |

## Notes specific to CT

- Use `binWidth: 25` in the PyRadiomics params (CT HU values are in a
  fixed range, fixed-bin-width is the standard choice).
- Do NOT enable `normalize: true` in PyRadiomics — CT HU values already
  carry physical meaning.
- If multiple kV / different scanners, document this in your methods; HABIT
  does not auto-correct for scanner effects on CT (use ICC analysis later
  to filter unstable features).

## Validation checkpoints

1. After preprocess: per-subject CT volumes have expected dimensions
   (`validate_preprocess_output.py` checks min/max/mean).
2. After radiomics: `inspect_feature_csv.py` reports no NaN columns.
3. After ML: `roc_curve.pdf` exists and test AUC > 0.5.

## Common pitfalls

- **HU outside [-1000, 3000]** → check raw DICOM rescale slope/intercept
  was applied during conversion.
- **Mask label != 1** → adjust `setting.label` in the params YAML.
- **Air around the body included in feature extraction** → check the mask
  isn't accidentally inverted or covering air.
