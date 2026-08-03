# Model Choice Guide

HABIT supports 13+ classifiers. Configure as many as you want in the same
`models:` block — they all train and evaluate side-by-side, then
`habit-model-comparison` can plot them together.

## Setting parameters

The keys shown in this guide are the *recommended* ones, not a whitelist. Any
parameter of the underlying estimator can be set under `params:`, including
parameters added by a newer library version. A key the estimator does not accept
is reported as a warning and skipped; set `strict_model_params: true` at the top
level of the config to make it an error instead. To see exactly which parameters
will be applied before starting a run:

```bash
habit check-config -c config/machine_learning/config_machine_learning.yaml -w model
```

## Decision matrix

| Cohort size | Feature dimension | Recommended models |
|---|---|---|
| < 100 patients | < 30 features | `LogisticRegression`, `GaussianNB` |
| < 100 patients | 30–500 features | `LogisticRegression` (with strong selection) |
| 100–500 | < 100 features | `LogisticRegression`, `RandomForest`, `XGBoost` |
| 100–500 | 100–1000 features | `LogisticRegression`, `RandomForest`, `XGBoost`, `SVC` (RBF) |
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

### SVM vs SVC
Two separate models — pick by whether you need a kernel:

- `SVM` is a **LinearSVC**: fast, linear boundary only. It has no `kernel`,
  `gamma` or `probability` parameter; probabilities are approximated from the
  decision function.
- `SVC` is the **kernel SVM** with native probability estimates. RBF works well
  after dimensionality reduction (LASSO/mRMR). Slow on > 5000 samples;
  `probability: true` slows it further because sklearn fits an internal
  calibration model.

```yaml
SVM:                      # linear, fast
  params:
    random_state: 42
    C: 1.0
    max_iter: 1000
```

```yaml
SVC:                      # kernel, supports non-linear boundaries
  params:
    random_state: 42
    C: 1.0
    kernel: rbf           # linear | poly | rbf | sigmoid
    gamma: scale
    probability: true     # required for ROC/DCA
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
- AutoGluon has a two-part API, mirrored by two YAML blocks: `predictor:` is
  passed to `TabularPredictor(...)`, `fit:` to `TabularPredictor.fit(...)`. Any
  AutoGluon parameter can be used, not just the ones shown here.

```yaml
AutoGluonTabular:
  params:
    random_state: 42
    predictor:
      path: ./ml_data/autogluon_models
      label: label
      eval_metric: roc_auc
      verbosity: 1
    fit:
      time_limit: 300                  # 5 minutes
      presets: high_quality            # best_quality | high_quality | good_quality | medium_quality | optimize_for_deployment
      num_bag_folds: 5                 # any other fit() parameter also works
```

The older flat form (`time_limit` / `presets` / `eval_metric` directly under
`params`) is still accepted and routed automatically.

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
