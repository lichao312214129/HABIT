# Recipe — DCE-MRI / Dynamic CT Kinetic Habitat Analysis

For multi-phase contrast studies where the biological signal is in the
**time-intensity curve** (wash-in / wash-out behavior). Habitat clustering
uses the `kinetic()` voxel feature method, which requires a per-subject
timestamps Excel file.

## Required inputs

| # | Question | Stop if missing |
|---|---|---|
| 1 | Path to data root | yes |
| 2 | Phase folder names IN ACQUISITION ORDER (e.g. `pre_contrast, LAP, PVP, delay_3min`) | yes |
| 3 | Path to timestamps Excel (subject IDs + per-phase scan time in minutes) | yes |
| 4 | Output root | yes |
| 5 | Number of habitats expected (typical DCE: 3–5; default 4) | no |
| 6 | Path to clinical labels CSV for ML | yes (if doing ML) |

## Critical pre-checks

1. **Timestamps Excel format**:
   ```
   subjID | pre_contrast | LAP | PVP | delay_3min
   sub001 | 0.0          | 0.5 | 1.5 | 3.0
   sub002 | 0.0          | 0.4 | 1.6 | 3.0
   ```
   - Times in **minutes from injection**
   - Subject IDs MUST match folder names exactly
   - Column names MUST match the phase folder names exactly
2. **Do NOT z-score normalize phases independently** in preprocessing — it
   destroys relative enhancement signal. Use the DCE preprocessing template.

## CLI sequence

```bash
# 0) Sanity checks
python skills/habit-quickstart/scripts/check_environment.py
python skills/habit-quickstart/scripts/check_data_layout.py <DATA_ROOT> --modalities pre_contrast LAP PVP delay_3min

# 1) Preprocess (DCE-aware: N4 + resample + register; NO z-score)
habit preprocess --config configs/01_dce_preprocess.yaml
python skills/habit-preprocess/scripts/validate_preprocess_output.py <PREPROCESS_OUT> \
  --modalities pre_contrast LAP PVP delay_3min

# 2) Habitat clustering (two_step recommended for DCE)
habit get-habitat --config configs/02_dce_habitat.yaml
python skills/habit-habitat-analysis/scripts/validate_habitat_output.py <HABITAT_OUT> --two-step

# 3) Feature extraction
habit extract --config configs/03_extract.yaml

# 4) ML training (after merging features + clinical labels)
habit merge-csv <HABITAT_OUT>/features/whole_habitat_radiomics.csv \
  <HABITAT_OUT>/features/msi_features.csv \
  <HABITAT_OUT>/features/ith_scores.csv \
  <CLINICAL_CSV> \
  -o configs/ml_input.csv --index-col subjID

habit model --config configs/04_ml_train.yaml --mode train
```

## Configs to generate

| Step | Template |
|---|---|
| 01_dce_preprocess | `config_templates/skill_scaffolds/preprocess_dce_mri.yaml` |
| 02_dce_habitat | `config_templates/skill_scaffolds/habitat_kinetic_dce.yaml` |
| 03_extract | `config_templates/skill_scaffolds/extract_features_publication.yaml` |
| 04_ml_train | `config_templates/skill_scaffolds/ml_pipeline_radiomics_std.yaml` |

## Output highlights

- `<HABITAT_OUT>/<subject>/<subject>_habitats_remapped.nrrd` — habitat label
  map. Open in ITK-SNAP overlaid on the **arterial** (LAP) phase to see how
  habitats correspond to enhancement patterns.
- `<HABITAT_OUT>/visualizations/cluster_centroids.png` — shows the kinetic
  curve of each habitat centroid; expect e.g. "wash-out", "persistent",
  "plateau", "necrotic" patterns.

## Validation checkpoints

1. After preprocess: every subject has all phases preserved (use
   `validate_preprocess_output.py` with full phase list).
2. After habitat: `validate_habitat_output.py --two-step` returns 0.
3. **MANUAL**: open `cluster_centroids.png` and confirm habitat curves are
   biologically distinct. If two habitats have nearly identical curves,
   reduce `fixed_n_clusters`.

## Troubleshooting

- "kinetic() failed: subject sub001 missing timestamps" → fix the Excel.
- "All habitats have similar curves" → reduce `fixed_n_clusters` or check
  preprocessing didn't accidentally normalize the phases independently.
- "Pre-contrast image looks artifacted" → N4 with `num_fitting_levels: 4`
  is sometimes too aggressive on pre-contrast; try 3.
