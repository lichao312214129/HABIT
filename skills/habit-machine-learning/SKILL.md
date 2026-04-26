---
name: habit-machine-learning
description: Train, predict, or k-fold cross-validate machine learning classifiers on habitat/radiomics features using HABIT. Supports LogisticRegression, RandomForest, XGBoost, SVM, MLP, AutoGluon, and 10+ feature selection methods (LASSO, mRMR, RFECV, ICC, correlation, ANOVA, etc.). Use when the user wants to build a prediction model from feature CSVs. Triggers on phrases like "训练模型", "机器学习建模", "特征选择", "K折交叉验证", "k-fold CV", "train classifier", "feature selection", "LASSO", "predict mode".
---

# HABIT Machine Learning

Train classification models on extracted features. Three sub-commands:

| Command | Purpose | Config template |
|---|---|---|
| `habit model --mode train` | Train on a fixed train/test split | `config_machine_learning_annotated.yaml` |
| `habit model --mode predict` | Apply a trained model to new data | Same template, predict variant |
| `habit cv` | K-fold cross-validation (no train/test split needed) | `config_machine_learning_kfold_annotated.yaml` |

## Decision Tree

**Does the user have a pre-defined train/test split?**
- Yes (e.g. multi-center cohort with internal/external) → `habit model --mode train` with `split_method: custom`
- No, just one cohort → use `habit cv` (k-fold, more robust for small datasets)

**Does the user already have a trained model?**
- Yes → `habit model --mode predict`

## Required Inputs

User MUST tell you:
1. **Path to feature CSV(s)** — output from `habit extract` or `habit radiomics`
2. **Subject ID column name** (e.g. `subjID`, `PatientID`)
3. **Label column name** (binary 0/1)
4. **Output directory**

If using `split_method: custom`, also need:
- `train_ids_file` and `test_ids_file` (one subject ID per line)

## Standard Train Config

```yaml
input:
  - path: ./results/features/whole_habitat_radiomics.csv
    name: habitat_           # prefix for feature names
    subject_id_col: subjID
    label_col: label
    features:                # empty = use all

  # Multiple files merged on subject_id_col:
  - path: ./results/features/msi_features.csv
    name: msi_
    subject_id_col: subjID
    label_col: label

output: ./results/ml

split_method: custom
train_ids_file: ./data/train_ids.txt
test_ids_file: ./data/test_ids.txt
# Or for random split:
# split_method: stratified
# test_size: 0.3
# random_state: 42

normalization:
  method: z_score              # z_score | min_max | robust | max_abs | quantile | power

feature_selection_methods:
  # Methods run sequentially. before_z_score: true = before normalization.
  - method: variance
    params:
      threshold: 0.2
      before_z_score: true       # MUST be true (z-score makes all variances=1)

  - method: correlation
    params:
      threshold: 0.80
      method: spearman
      before_z_score: false

  - method: lasso
    params:
      cv: 10
      n_alphas: 100
      visualize: true
      before_z_score: false

models:
  LogisticRegression:
    params:
      random_state: 42
      max_iter: 1000
      C: 1.0
      penalty: l2
      solver: lbfgs

  RandomForest:
    params:
      random_state: 42
      n_estimators: 100
      max_features: sqrt
      class_weight: balanced

  XGBoost:
    params:
      random_state: 42
      n_estimators: 100
      max_depth: 3
      learning_rate: 0.1

is_visualize: true
is_save_model: true
```

## Available Feature Selection Methods

Each runs sequentially in the order listed. Set `before_z_score: true` to run before normalization (only `variance` strictly needs this).

| Method | Use when |
|---|---|
| `icc` | Have test-retest data; want only stable features |
| `variance` | Remove near-constant features (always include this first) |
| `statistical_test` | t-test or Mann-Whitney by class |
| `vif` | Remove multicollinear features |
| `correlation` | Remove pairs with \|r\| > threshold (always include) |
| `anova` | F-test per feature |
| `chi2` | Non-negative features only |
| `rfecv` | Recursive elimination with CV |
| `mrmr` | Min-redundancy max-relevance |
| `lasso` | L1 regularization (most popular for radiomics) |
| `univariate_logistic` | Per-feature LR p-value |
| `stepwise` | Forward/backward AIC/BIC |

**Recommended pipeline for radiomics**:
1. `variance` (before_z_score=true) — drop constant features
2. `correlation` (after) — drop redundancy
3. `statistical_test` or `anova` — keep label-relevant features
4. `lasso` — final selection

## Available Models

`LogisticRegression`, `SVM`, `RandomForest`, `XGBoost`, `KNN`, `MLP`, `GaussianNB`, `GradientBoosting`, `AdaBoost`, `DecisionTree`, `AutoGluonTabular`.

**Tips**:
- `AutoGluonTabular` requires Python 3.10 — warn user.
- For radiomics with <500 patients: `LogisticRegression`, `RandomForest`, `XGBoost` are safe defaults.
- Multiple models in same config = trained simultaneously and compared.

## K-Fold Cross-Validation

Use `habit cv` instead of `habit model` when:
- Small dataset (<200 patients)
- No predefined train/test split
- Want mean ± std performance estimates

Same config structure, but:
```yaml
n_splits: 5         # 5 or 10 typical
stratified: true    # preserve class distribution
random_state: 42

# No split_method, no train_ids_file/test_ids_file
```

Feature selection runs **inside each fold** to avoid data leakage.

## Predict Mode

```bash
habit model --config <predict_config.yaml> --mode predict
```

Predict config needs:
- Path to a CSV of NEW patient features (same columns as training data)
- Path to the saved `.pkl` model file (from a previous train run)
- Output directory for predictions

## Reference Templates

- Train: `config_templates/config_machine_learning_annotated.yaml`
- K-fold: `config_templates/config_machine_learning_kfold_annotated.yaml`
- Minimal scaffolds: `references/config_train_minimal.yaml`, `references/config_kfold_minimal.yaml`

## Output Files

```
output/
├── <ModelName>_model.pkl                    # if is_save_model: true
├── <ModelName>_predictions.csv              # train + test predictions, with prob & pred columns
├── all_prediction_results.csv               # combined predictions for all models
├── feature_selection_<method>.png           # if visualize: true in selection params
├── roc_curves.pdf                           # if is_visualize: true
├── calibration_curves.pdf
├── confusion_matrix_<model>.pdf
├── decision_curves.pdf
└── ml.log
```

For k-fold, additionally:
- `aggregated_results.json` — mean ± std across folds
- `fold_*/` — per-fold details

## Common Pitfalls

1. **`subject_id_col` mismatch across input files** → fail with "no overlapping subjects". Make sure all CSVs use same column.
2. **`label_col` value not in {0, 1}** → re-encode first. Multi-class is not officially supported.
3. **Multi-file merge produces NaN columns** → some subjects miss features in one file. Use inner join (default) or pre-merge with `habit merge-csv`.
4. **`variance` selector with `before_z_score: false`** → all variances = 1, threshold useless. Always set true.
5. **AutoGluon import error** → user is on Python 3.8; tell them to use Python 3.10 env.
6. **Feature selection dropped all features** → thresholds too strict; loosen incrementally.

## Next Steps

After training multiple models, use `habit-model-comparison` skill to generate publication-quality comparison plots and DeLong tests.
