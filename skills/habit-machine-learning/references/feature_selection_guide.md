# Feature Selection — Recipe Guide

`feature_selection_methods` is a **list run sequentially**. Each method
narrows the feature set; the next one operates on the narrowed set. Order
matters.

## Selection methods at a glance

| Method | What it does | When to put it in the chain | Needs `before_z_score` |
|---|---|---|---|
| `icc` | Drop features with ICC < threshold (test-retest stability) | First, if you have ICC results JSON | false |
| `variance` | Drop near-constant features | First (always recommended) | **true** |
| `correlation` | Drop pairwise redundancy (\|r\| > threshold) | After variance | false |
| `statistical_test` | t-test / Mann-Whitney by class | Middle | false |
| `anova` | F-test per feature | Alternative to statistical_test for >2 classes | false |
| `chi2` | Chi-square (non-negative features only) | Rare; only for categorical-encoded features | false |
| `vif` | Drop multicollinear features (VIF > threshold) | Optional; after correlation | false |
| `mrmr` | Min-redundancy max-relevance | Before lasso when feature_count > 200 | false |
| `rfecv` | Recursive elimination with CV | Alternative to lasso; slow | false |
| `lasso` | L1 regularization | Last (final selection) | false |
| `univariate_logistic` | Single-feature LR p-value | Alternative to statistical_test | false |
| `stepwise` | Forward/backward AIC/BIC | Rare; alternative to lasso | false |

## Recommended chains by scenario

### Standard radiomics study (50–500 features)
```yaml
feature_selection_methods:
  - method: variance
    params: {threshold: 0.2, before_z_score: true}
  - method: correlation
    params: {threshold: 0.85, method: spearman, before_z_score: false}
  - method: statistical_test
    params: {p_threshold: 0.05, before_z_score: false}
  - method: lasso
    params: {cv: 10, n_alphas: 100, visualize: true, before_z_score: false}
```

### High-dimensional radiomics (>1000 features, small cohort)
```yaml
feature_selection_methods:
  - method: variance
    params: {threshold: 0.2, before_z_score: true}
  - method: correlation
    params: {threshold: 0.95, method: spearman, before_z_score: false}
  - method: statistical_test
    params: {p_threshold: 0.10, before_z_score: false}
  - method: mrmr
    params: {n_features: 50, before_z_score: false}
  - method: lasso
    params: {cv: 10, before_z_score: false}
```

### Test-retest reproducibility-aware
```yaml
feature_selection_methods:
  - method: icc
    params: {icc_results: ./results/icc_results.json, threshold: 0.8, before_z_score: false}
  - method: variance
    params: {threshold: 0.2, before_z_score: true}
  - method: correlation
    params: {threshold: 0.85, before_z_score: false}
  - method: lasso
    params: {cv: 10, before_z_score: false}
```

### Small clinical-only dataset (<50 features)
```yaml
feature_selection_methods:
  - method: univariate_logistic
    params: {threshold: 0.10, before_z_score: false}
```
Don't over-engineer; with so few features, just keep univariate-significant ones.

## Hard rules

1. **`variance` MUST set `before_z_score: true`**. After z-score, every
   feature has variance 1 and the threshold becomes useless.
2. **`correlation` works on standardized OR raw**; convention is
   `before_z_score: false`.
3. **`lasso` MUST be after z-score**; L1 is scale-sensitive.
4. **Order matters**: cheap filters first (variance, correlation), expensive
   modeling last (lasso, rfecv).
5. **In k-fold CV** every selector runs **inside each fold** — no leakage.
6. **For `chi2`**: features must be non-negative. Apply `min_max` instead of
   `z_score` upstream, otherwise chi2 will error.

## When LASSO drops everything

If after a LASSO run you see "0 features selected", it means alpha was too
strong. Tune by:
- Lowering the smallest alpha tested (`alphas: [0.001, 0.01, ...]`)
- Increasing CV folds (`cv: 20`) for a smoother path
- Or replace `lasso` with `mrmr` (deterministic top-N selection)

## When statistical_test drops everything

Loosen `p_threshold` from 0.05 to 0.10 or 0.20. For small cohorts
(< 100 patients), strict p-thresholds eliminate signal that real cross-fold
ML can use.
