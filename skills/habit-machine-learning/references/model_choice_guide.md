# Model Choice Guide

HABIT supports 11+ classifiers. Configure as many as you want in the same
`models:` block — they all train and evaluate side-by-side, then
`habit-model-comparison` can plot them together.

## Decision matrix

| Cohort size | Feature dimension | Recommended models |
|---|---|---|
| < 100 patients | < 30 features | `LogisticRegression`, `GaussianNB` |
| < 100 patients | 30–500 features | `LogisticRegression` (with strong selection) |
| 100–500 | < 100 features | `LogisticRegression`, `RandomForest`, `XGBoost` |
| 100–500 | 100–1000 features | `LogisticRegression`, `RandomForest`, `XGBoost`, `SVM` (RBF) |
| > 500 | any | All of above + `MLP`, `AutoGluonTabular` |

## Per-model notes

### LogisticRegression
- **Default first choice for radiomics**. Linear, interpretable, well-calibrated.
- After LASSO, basically becomes a parametric MLE on the selected features.
- Set `class_weight: balanced` if class ratio > 1:3.

```yaml
LogisticRegression:
  params:
    random_state: 42
    max_iter: 1000
    C: 1.0
    penalty: l2
    solver: lbfgs
```

### RandomForest
- Robust, no scaling needed (but HABIT z-scores anyway).
- Good when nonlinear interactions are expected.
- `class_weight: balanced` helps imbalanced data.
- More trees (`n_estimators: 200-500`) help; max_depth=null is fine.

```yaml
RandomForest:
  params:
    random_state: 42
    n_estimators: 200
    max_depth: null
    max_features: sqrt
    class_weight: balanced
```

### XGBoost
- Often the best test-AUC on tabular data.
- More hyperparameters to tune than RF; defaults below are safe.
- Risk of overfitting on small data — keep `max_depth: 3-4`.

```yaml
XGBoost:
  params:
    random_state: 42
    n_estimators: 100
    max_depth: 3
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
```

### SVM
- Only with `probability: true` (HABIT needs probabilities for ROC/DCA).
- RBF kernel works well after dimensionality reduction (LASSO/mRMR).
- Slow on > 5000 samples; not recommended for large cohorts.

```yaml
SVM:
  params:
    random_state: 42
    C: 1.0
    kernel: rbf
    gamma: scale
    probability: true
```

### MLP
- Tabular MLP rarely beats RF/XGBoost; use only if you really want a NN baseline.
- Easy to overfit small data — keep one hidden layer.

```yaml
MLP:
  params:
    random_state: 42
    hidden_layer_sizes: [64]
    activation: relu
    solver: adam
    alpha: 0.001
    max_iter: 500
    early_stopping: true
```

### AutoGluonTabular
- Trains an ensemble automatically; usually the best AUC out of the box.
- **Requires Python 3.10**. Warn the user.
- Time-bounded by `time_limit` (seconds).

```yaml
AutoGluonTabular:
  params:
    path: ./ml_data/autogluon_models
    label: label
    time_limit: 300                    # 5 minutes
    presets: high_quality              # best_quality | high_quality | medium_quality | fast
    eval_metric: roc_auc
    verbosity: 1
```

### Less common: KNN, GaussianNB, GradientBoosting, AdaBoost, DecisionTree
Available but rarely chosen for serious radiomics work. Use only if a baseline
or sanity check is desired.

## How many models to train at once

- **2-3 models** for publication: covers a linear baseline + a nonlinear ensemble.
- **5+** if you also want a comparison table — but each adds compute time and
  inflates multiple-comparison corrections in DeLong tests.
- For **K-fold CV**, prefer 2-3 models because k * m models get trained.

## Picking the "main" model for your paper

After training, compare AUCs in the test split. Tie-breakers in order:
1. Higher test AUC
2. Smaller train-test AUC gap (less overfitting)
3. Better calibration (smaller Brier score)
4. Simpler model (LogisticRegression > XGBoost > AutoGluon)

The simpler model wins when AUC is within 0.02; reviewers prefer interpretability.
