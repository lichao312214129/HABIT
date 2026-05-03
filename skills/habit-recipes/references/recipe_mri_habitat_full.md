# Recipe — Multi-Modal MRI Habitat Analysis (Full Pipeline)

The most common HABIT scenario. User has multi-sequence MRI (T1, T2, optionally DWI/ADC/FLAIR) with manually-drawn tumor masks, and wants the complete chain: preprocess → habitat clustering → feature extraction → ML modeling → comparison.

## Required inputs

Ask the user and confirm before starting:

| # | Question | Stop if missing |
|---|---|---|
| 1 | Path to data root (per-subject folders) | yes |
| 2 | List of modality folder names (e.g. `T1, T2, DWI, ADC`) | yes |
| 3 | Are images already preprocessed? | yes |
| 4 | Output root directory | yes |
| 5 | Path to feature/label CSV (clinical labels) for ML | yes |
| 6 | Subject ID column name and label column name in that CSV | yes |
| 7 | one_step or two_step habitat? (default: one_step for pilot, two_step for publication) | no — default to one_step |

## CLI sequence

```bash
# 0) Environment + data sanity check
python skills/habit-quickstart/scripts/check_environment.py
python skills/habit-quickstart/scripts/check_data_layout.py <DATA_ROOT> --modalities T1 T2 DWI ADC

# 1) Preprocess (if not already done)
habit preprocess --config configs/01_preprocess.yaml
python skills/habit-preprocess/scripts/validate_preprocess_output.py <PREPROCESS_OUT_DIR> --modalities T1 T2 DWI ADC

# 2) Habitat clustering (one_step example)
habit get-habitat --config configs/02_habitat.yaml
python skills/habit-habitat-analysis/scripts/validate_habitat_output.py <HABITAT_OUT_DIR>

# 3) Feature extraction
habit extract --config configs/03_extract.yaml

# 4) Merge feature CSVs with clinical labels (single ML input file)
habit merge-csv \
  <HABITAT_OUT>/results/features/whole_habitat_radiomics.csv \
  <HABITAT_OUT>/results/features/msi_features.csv \
  <HABITAT_OUT>/results/features/ith_scores.csv \
  <CLINICAL_CSV> \
  -o configs/ml_input.csv \
  --index-col subjID \
  --join inner

python skills/habit-feature-extraction/scripts/inspect_feature_csv.py configs/ml_input.csv \
  --subject-id-col subjID --label-col label

# 5) Train ML models (multiple models in one config, then compare)
habit model --config configs/04_ml_train.yaml --mode train

# 6) (Optional) Compare against a clinical-only model trained the same way
habit compare --config configs/05_compare.yaml
```

## Configs to generate

| Step | Template to start from | Notes |
|---|---|---|
| 01_preprocess | `config_templates/skill_scaffolds/preprocess_mri_multimodal.yaml` | Pick fixed_image (usually T2) |
| 02_habitat | `config_templates/skill_scaffolds/habitat_one_step_minimal.yaml` for one_step, or `habitat_two_step_minimal.yaml` for two_step | Use `concat(raw(T1), raw(T2), ...)` |
| 03_extract | `config_templates/skill_scaffolds/extract_features_publication.yaml` | Need PyRadiomics params |
| 04_ml_train | `config_templates/skill_scaffolds/ml_pipeline_radiomics_std.yaml` | At least LogisticRegression + RandomForest |
| 05_compare | `config_templates/skill_scaffolds/model_comparison_two_models.yaml` | Only if comparing to clinical baseline |

## Expected outputs

```
<OUTPUT_ROOT>/
├── preprocessed/processed_images/...
├── habitat/
│   ├── <subject>/
│   │   ├── <subject>_habitats.nrrd          # view in ITK-SNAP
│   │   └── <subject>_supervoxel.nrrd        # only two_step
│   ├── habitats.csv
│   └── visualizations/
├── features/
│   ├── whole_habitat_radiomics.csv
│   ├── msi_features.csv
│   ├── ith_scores.csv
│   ├── habitat_basic_features.csv
│   └── raw_image_radiomics.csv
└── ml/
    ├── all_prediction_results.csv
    ├── roc_curve.pdf
    ├── calibration_curve.pdf
    ├── decision_curve.pdf
    └── *_model.pkl
```

## Validation checkpoints

After each step, the agent should verify before continuing:

1. After preprocess → `validate_preprocess_output.py` returns 0
2. After habitat → `validate_habitat_output.py` returns 0
3. After extract → at least 5 CSVs exist under `features/`, no empty
4. After ML → `roc_curve.pdf` exists; `evaluation_metrics.csv` test AUC > 0.5
5. After compare → `delong_results.json` parses and contains "test" block

## Common deviations from this recipe

- User has only T2 (single modality) → swap step 2 to `voxel_level.method: raw(T2)`, skip registration
- User wants k-fold CV → replace step 5 with `habit cv` and use `config_templates/skill_scaffolds/ml_pipeline_radiomics_std.yaml` as base (remove `split_method` / train-test fields for `habit cv`; see `config_templates/skill_scaffolds/ml_kfold_minimal.yaml`)
- User has external test cohort → use `split_method: custom` with explicit train/test ID files; generate via `habit-machine-learning/scripts/prepare_split_files.py`
