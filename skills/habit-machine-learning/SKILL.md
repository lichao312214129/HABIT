---
name: habit-machine-learning
description: Train, predict, or k-fold cross-validate ML classifiers on habitat / radiomics feature CSVs using HABIT. Supports LogisticRegression, RandomForest, XGBoost, SVM, MLP, AutoGluon, and 12+ feature selection methods (LASSO, mRMR, RFECV, ICC, correlation, ANOVA, ...). Use when the user wants to build a prediction model from feature CSVs. Triggers on "训练模型", "建模", "K折交叉验证", "k-fold", "LASSO", "feature selection", "predict mode". Runs `habit model` or `habit cv`.
---

# HABIT Machine Learning

Train classification models on extracted features. Three sub-commands:

| Command | Purpose | Template |
|---|---|---|
| `habit model --mode train` | Train on a fixed train/test split | `config_templates/skill_scaffolds/ml_train_minimal.yaml` |
| `habit model --mode predict` | Apply a trained model to new data | `config_templates/skill_scaffolds/ml_predict_minimal.yaml` |
| `habit cv` | K-fold cross-validation | `config_templates/skill_scaffolds/ml_kfold_minimal.yaml` |

## Required Information

| Field | Stop if missing |
|---|---|
| Path to feature CSV(s) | yes |
| `subject_id_col` | yes |
| `label_col` (binary 0/1) | yes |
| `output` directory | yes |
| Split strategy: `stratified` / `custom` / k-fold | yes |
| (custom only) `train_ids_file` and `test_ids_file` | yes |
| Models to train (≥1) | yes |
| (predict only) trained `.pkl` model path | yes |

If split files are needed and the user doesn't have them, generate via:

```bash
python skills/habit-machine-learning/scripts/prepare_split_files.py <csv_path> \
  --subject-id-col subjID --label-col label --test-size 0.3 --output-dir ./splits
```

## Decision tree

**Have a pre-defined train/test split?**
- Yes (multi-center, internal/external) → `habit model --mode train` with `split_method: custom`
- No, single cohort → use `habit cv` (more robust for small datasets)

**Already have a trained model?**
- Yes → `habit model --mode predict` with `config_templates/skill_scaffolds/ml_predict_minimal.yaml`

## Standard train pipeline

Use `config_templates/skill_scaffolds/ml_pipeline_radiomics_std.yaml`. The selection chain is:

1. `variance(0.2)` — drop near-constant features (set `before_z_score: true`)
2. `correlation(0.85)` — drop redundant features (Spearman)
3. `statistical_test(p<0.05)` — keep label-relevant features
4. `lasso(cv=10)` — final L1 selection

For high-dimensional cases (>1000 features), use `config_templates/skill_scaffolds/ml_pipeline_high_dim.yaml`
which inserts `mrmr` between step 3 and step 4.

Detailed feature selection guidance: `references/feature_selection_guide.md`.

## Models

`LogisticRegression`, `SVM`, `RandomForest`, `XGBoost`, `KNN`, `MLP`,
`GaussianNB`, `GradientBoosting`, `AdaBoost`, `DecisionTree`, `AutoGluonTabular`.

Recommendations: `references/model_choice_guide.md`.

Tips:
- `AutoGluonTabular` requires Python 3.10 — warn the user.
- For radiomics with <500 patients: `LogisticRegression`, `RandomForest`, `XGBoost` are safe defaults.
- Multiple models in one config = trained simultaneously and compared.

## K-fold cross-validation

Use `habit cv` instead of `habit model` when:
- Small dataset (<200 patients)
- No predefined train/test split
- Want mean ± std performance estimates

Same config structure as train, but:
```yaml
n_splits: 5         # 5 or 10 typical
stratified: true    # preserve class distribution
random_state: 42
# No split_method, no train_ids_file/test_ids_file
```

Feature selection runs **inside each fold** to avoid data leakage.

## Predict mode

```bash
habit model --config <predict_config.yaml> --mode predict
```

Predict config needs:
- Path to a CSV of NEW patient features (same columns as training)
- Path to the saved `.pkl` model (`<output>/<ModelName>_model.pkl`)
- Output directory for predictions

See `config_templates/skill_scaffolds/ml_predict_minimal.yaml`.

## Reference templates

Config index: `skills/CONFIG_SOURCES.md`.

| File | Use |
|---|---|
| `config_templates/skill_scaffolds/ml_train_minimal.yaml` | scaffold |
| `config_templates/skill_scaffolds/ml_kfold_minimal.yaml` | k-fold |
| `config_templates/skill_scaffolds/ml_predict_minimal.yaml` | predict on new data |
| `config_templates/skill_scaffolds/ml_pipeline_radiomics_std.yaml` | standard radiomics pipeline (battle-tested) |
| `config_templates/skill_scaffolds/ml_pipeline_high_dim.yaml` | for >1000 features |
| `references/feature_selection_guide.md` | selection chain recipes |
| `references/model_choice_guide.md` | which model to pick |

Full annotated references:
- `config_templates/config_machine_learning_annotated.yaml`
- `config_templates/config_machine_learning_kfold_annotated.yaml`

## Validate inputs (MANDATORY before run)

```bash
python skills/habit-feature-extraction/scripts/inspect_feature_csv.py <input_csv> \
  --subject-id-col subjID --label-col label
```

Catches duplicate IDs, NaN columns, non-binary labels BEFORE training.

## Output files

```
output/
├── <ModelName>_model.pkl                    # if is_save_model: true
├── <ModelName>_predictions.csv              # train + test predictions
├── all_prediction_results.csv               # combined for all models
├── feature_selection_<method>.png           # if visualize: true
├── roc_curves.pdf                           # if is_visualize: true
├── calibration_curves.pdf
├── confusion_matrix_<model>.pdf
├── decision_curves.pdf
└── ml.log
```

For k-fold, additionally:
- `aggregated_results.json` — mean ± std across folds
- `fold_*/` — per-fold details

## Common pitfalls

1. **`subject_id_col` mismatch across input files** → fail with "no overlapping subjects".
2. **`label_col` value not in {0, 1}** → re-encode.
3. **Multi-file merge produces NaN columns** → some subjects miss features in one file.
4. **`variance` selector with `before_z_score: false`** → variances all = 1, threshold useless.
5. **AutoGluon ImportError** → user is on Python 3.8; need 3.10.
6. **Feature selection dropped all features** → thresholds too strict; loosen.

For more, see `habit-troubleshoot/references/errors_ml.md`.

## Next step

After training multiple models, use `habit-model-comparison` to generate
publication-quality ROC / DCA / calibration plots and DeLong tests.
