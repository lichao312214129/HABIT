# Recipe — Test-Retest Reproducibility Study

For studies where the same patients are imaged twice (or by two raters / two
scanners) and the goal is to identify reproducible features and habitats.
Combines DICE, ICC, habitat label re-mapping, and a final ICC-filtered ML run.

## Required inputs

| # | Question | Stop if missing |
|---|---|---|
| 1 | Path to scan-1 root and scan-2 root | yes |
| 2 | Modality folder names (must match across scans) | yes |
| 3 | Output root | yes |
| 4 | Same masks across scans, OR re-segmented in each scan? | yes |

## Pipeline overview

```
   scan1                 scan2
     |                     |
   preprocess            preprocess
     |                     |
   habitat               habitat
     |                     |
   features              features
       \                 /
        \               /
       habit dice  -- mask agreement
       habit retest  -- map habitat labels consistently
       habit icc   -- per-feature ICC across scans
                |
        ICC-filtered features
                |
            habit model (k-fold CV recommended)
```

## CLI sequence

```bash
# 0) Sanity checks for both scans
python skills/habit-quickstart/scripts/check_data_layout.py <SCAN1_ROOT> --modalities <MODALITIES>
python skills/habit-quickstart/scripts/check_data_layout.py <SCAN2_ROOT> --modalities <MODALITIES>

# 1) Preprocess each scan separately (same config, different in/out paths)
habit preprocess --config configs/01_preprocess_scan1.yaml
habit preprocess --config configs/01_preprocess_scan2.yaml

# 2) Mask agreement (Dice) between the two scans
habit dice \
  --input1 <SCAN1_OUT>/processed_images/masks \
  --input2 <SCAN2_OUT>/processed_images/masks \
  --output configs/dice_results.csv \
  --mask-keyword masks --label-id 1

# 3) Habitat clustering on each scan
habit get-habitat --config configs/02_habitat_scan1.yaml
habit get-habitat --config configs/02_habitat_scan2.yaml

# 4) Re-map habitat labels so 'habitat 2' means the same thing in both scans
habit retest --config configs/03_test_retest.yaml

# 5) Feature extraction on both scans (use the remapped habitats)
habit extract --config configs/04_extract_scan1.yaml
habit extract --config configs/04_extract_scan2.yaml

# 6) Per-feature ICC between the two CSVs
habit icc --config configs/05_icc.yaml

# 7) ML with ICC-based feature selection (use the resulting JSON)
habit model --config configs/06_ml_icc_filtered.yaml --mode train
```

## Configs to generate

| Step | Template |
|---|---|
| 01_preprocess_* | `config_templates/skill_scaffolds/preprocess_mri_multimodal.yaml` (or CT) |
| 02_habitat_* | `config_templates/skill_scaffolds/habitat_two_step_minimal.yaml` |
| 03_test_retest | `config_templates/skill_scaffolds/auxiliary_test_retest_minimal.yaml` |
| 04_extract_* | `config_templates/skill_scaffolds/extract_features_publication.yaml` |
| 05_icc | `config_templates/skill_scaffolds/auxiliary_icc_minimal.yaml` |
| 06_ml_icc_filtered | Custom — see below |

## ICC-filtered ML config snippet

```yaml
# In your habit model config:
feature_selection_methods:
  # First: filter unstable features by ICC
  - method: icc
    params:
      icc_results: ./configs/icc_results.json
      keys: [whole_habitat_radiomics]   # match keys produced by step 5
      threshold: 0.8                     # keep ICC >= 0.8 (excellent reproducibility)
      before_z_score: false

  # Then the usual chain
  - method: variance
    params: {threshold: 0.2, before_z_score: true}
  - method: correlation
    params: {threshold: 0.85, before_z_score: false}
  - method: lasso
    params: {cv: 10, before_z_score: false}
```

## Validation checkpoints

1. Dice CSV: per-subject DSC should mostly be > 0.7. If < 0.7 for many,
   the segmentations differ too much and habitat reproducibility will be
   poor regardless.
2. After retest mapping: check `mapping_quality.csv` — most subjects
   should have similarity > 0.7.
3. After ICC: the JSON should report most features as Good (≥ 0.6) or
   Excellent (≥ 0.75). If not, increase `threshold` cautiously.

## Reporting

For a paper, report:
- Mask Dice mean ± std
- Habitat label remapping success rate (subjects with similarity > 0.7)
- Feature ICC distribution (% Excellent, % Good, % Fair, % Poor)
- Test AUC of the ICC-filtered model vs the unfiltered model — quantifies
  the cost of reproducibility filtering
