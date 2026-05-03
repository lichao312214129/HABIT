# Reading the Model Comparison Outputs

`habit compare` writes 4 plots, 1 stats JSON, and 3 metric CSVs. This guide
explains what each one tells the user and what reviewers will ask about.

## ROC Curves (`roc_curves.pdf`)
- X = 1 - specificity, Y = sensitivity
- Diagonal = random classifier (AUC = 0.5)
- The further toward the top-left, the better
- AUC values in the legend; hover or read manually
- Reviewers will ask: **AUC + 95% CI** in the test split

## Decision Curves (`decision_curves.pdf`) — DCA
- X = threshold probability, Y = net benefit
- A model is clinically useful if its curve is above both the "treat all"
  and "treat none" reference lines for the threshold range that matches
  the clinical decision (typically 10%-50% for cancer treatment)
- Curves crossing the reference lines = the model adds no value at that
  threshold

## Calibration Curves (`calibration_curves.pdf`)
- X = predicted probability, Y = observed fraction of positives
- Perfect calibration = diagonal line
- A curve **below** the diagonal = over-confident model (predicts high prob
  for cases that are mostly negative)
- A curve **above** the diagonal = under-confident
- Reviewers may ask for **Brier score** or **ECE** (compute manually if needed)

## Precision-Recall Curves (`precision_recall_curves.pdf`)
- More informative than ROC when classes are imbalanced (positive rate < 20%)
- AUC-PR (PR area under curve) is the headline number
- Useful for rare-event prediction (e.g. recurrence in early-stage cancer)

## DeLong test (`delong_results.json`)

Tests whether two AUCs are statistically different. JSON format:

```json
{
  "test": {
    "Clinical_vs_Radiomics": {"z": 2.31, "p": 0.021, "auc1": 0.74, "auc2": 0.81},
    "Clinical_vs_Habitat":   {"z": 3.05, "p": 0.002, "auc1": 0.74, "auc2": 0.85},
    "Radiomics_vs_Habitat":  {"z": 1.18, "p": 0.238, "auc1": 0.81, "auc2": 0.85}
  },
  "train": { ... }
}
```

Interpretation:
- `p < 0.05` → AUCs are significantly different
- Always report DeLong **on the test split**, not train
- For 3+ models, mention you applied (or didn't apply) Bonferroni correction

## Metric CSVs

### `basic_metrics.csv`
Fixed-threshold (0.5) metrics: accuracy, sensitivity, specificity, PPV, NPV.

### `youden_metrics.csv`
Same metrics but at the threshold maximizing Youden's index = sens + spec - 1.
This is the "best operating point" reviewers usually want.

### `target_metrics.csv`
Metrics at YOUR specified target (e.g. specificity ≥ 0.7). Useful for
"if we want sensitivity at least X, what's the corresponding specificity?"

## Reporting checklist for a paper

1. **Test AUC + 95% CI** for every model
2. **DeLong p-values** between key model pairs (test split)
3. **Calibration plot** + Brier score
4. **DCA** showing clinical utility
5. **Confusion matrix** + Youden-optimized sensitivity/specificity
6. **Decision threshold rationale** (why 0.5? why Youden? why 0.3?)

## Common gotchas

1. **`prob_col` mismatch**: `habit model` outputs columns like
   `LogisticRegression_prob`. If you ran multiple models in one config,
   pick the right column name in `files_config[*].prob_col`.
2. **Different subjects across files**: `habit compare` keeps the
   intersection. Check the log for "N subjects dropped" warnings.
3. **`split_col: split` but no `split` column**: set `split.enabled: false`,
   OR add a `split` column to your CSV via pandas before running.
4. **Probabilities outside [0,1]**: some models output logits or class
   scores. Check the CSV — if you see values like `-2.3` or `5.1`, you
   have logits, not probabilities. ROC will still work but DCA and
   calibration will be nonsense.
