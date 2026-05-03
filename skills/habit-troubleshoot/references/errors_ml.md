# Machine Learning & Comparison Errors

## Symptom: `KeyError: '<column_name>'` when loading input CSV

**Cause**: `subject_id_col` or `label_col` doesn't match an actual column.

**Fix**: Inspect the CSV first:
```bash
python skills/habit-feature-extraction/scripts/inspect_feature_csv.py <csv_path>
```

The script prints the first 5 column names. Update the config to use one
of those.

## Symptom: `No overlapping subjects across input files`

**Cause**: Multiple input CSVs but they share no subject IDs.

**Fix**:
1. Print the first column of each:
   ```bash
   python -c "import pandas as pd; print(pd.read_csv('a.csv').iloc[:5,0])"
   python -c "import pandas as pd; print(pd.read_csv('b.csv').iloc[:5,0])"
   ```
2. Common issues: column name differences (`subjID` vs `PatientID`),
   case differences (`SUB001` vs `sub001`), trailing spaces.
3. Pre-merge with `habit merge-csv` using the right `--index-col` and `--join inner`.

## Symptom: variance threshold drops all features

**Cause**: User set `before_z_score: false` for variance — but after z-score
all variances equal 1, so the threshold removes everything.

**Fix**: Always set `before_z_score: true` for variance:
```yaml
- method: variance
  params:
    threshold: 0.2
    before_z_score: true   # MUST be true
```

## Symptom: LASSO selects 0 features

**Cause**: Alpha grid was too aggressive, or the upstream selectors already
removed everything informative.

**Fix**:
1. Run upstream selectors more loosely:
   ```yaml
   - method: correlation
     params: {threshold: 0.95, ...}   # was 0.85
   - method: statistical_test
     params: {p_threshold: 0.20, ...} # was 0.05
   ```
2. Or replace lasso with mrmr (deterministic top-N):
   ```yaml
   - method: mrmr
     params: {n_features: 20}
   ```

## Symptom: `AutoGluonTabular` errors on import

**Cause**: AutoGluon requires Python 3.10. User is on 3.8.

**Fix**:
- Either remove `AutoGluonTabular:` from `models:`
- Or create a new env: `conda create -n habit310 python=3.10 -y && conda activate habit310 && pip install -e . autogluon`

## Symptom: train AUC = 1.0 but test AUC = 0.5

**Cause**: Severe overfitting. Common causes:
1. Test set leaked into training (e.g. same subject in both)
2. Feature selection done on the full data (data leakage); should be in pipeline
3. Model too complex for cohort size (e.g. RandomForest with `n_estimators: 1000`
   on 50 patients)

**Fix**:
1. Verify train/test IDs disjoint:
   ```bash
   diff <(sort train_ids.txt) <(sort test_ids.txt)
   # should output nothing
   ```
2. HABIT runs feature selection inside the pipeline already. If you
   pre-selected features manually before passing to HABIT, redo without
   pre-selection.
3. Switch to a simpler model: `LogisticRegression` only.

## Symptom: `delong_results.json` has all `p > 0.05`

**Cause**: All models perform similarly. This is a **finding**, not a bug.

**Fix** (if a result is needed):
- Add a more diverse feature set (try habitat features if you only had clinical)
- Increase cohort size (DeLong is sample-size-sensitive)
- Use a stronger model (XGBoost, AutoGluon)

## Symptom: `prob_col not found` when running `habit compare`

**Cause**: When `habit model` trains multiple models in one config, output
columns are prefixed: `LogisticRegression_prob`, `RandomForest_prob`, etc.

**Fix**: Update the comparison config:
```yaml
files_config:
  - path: ./results/ml/all_prediction_results.csv
    name: Radiomics
    prob_col: LogisticRegression_prob   # was just 'prob'
    pred_col: LogisticRegression_pred
```

## Symptom: ROC curve looks random (AUC ~0.5)

**Cause** (in order of likelihood):
1. Label column is shuffled relative to features
2. Feature CSV has wrong subject_id column → join produced random label assignment
3. Cohort really has no predictive signal in those features

**Fix**:
1. Verify label distribution per subject in the merged CSV:
   ```python
   import pandas as pd
   df = pd.read_csv('ml_input.csv')
   print(df['label'].value_counts())
   print(df.head(10))
   ```
2. Reverify the merge with explicit `--index-col` and inner join.
3. If still 0.5, the data may genuinely lack signal.

## Symptom: K-fold ROC curves vary wildly fold-to-fold

**Cause**: Small cohort (<100 patients) → high fold variance.

**Fix**:
- Switch from 10-fold to 5-fold (more samples per fold)
- Use `stratified: true` to ensure balanced folds
- Report mean ± std AUC instead of any single fold result

## Symptom: predict mode says "feature columns mismatch"

**Cause**: New patient CSV doesn't have the exact columns the trained model expects.

**Fix**: The new CSV must have:
- Same `subject_id_col` name
- Same feature columns (after the same prefix)
- The label column can be dummy (all 0s) since predict doesn't use it
- Run `inspect_feature_csv.py` on both training and predict CSVs to compare
