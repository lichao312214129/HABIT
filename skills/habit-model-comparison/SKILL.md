---
name: habit-model-comparison
description: Compare multiple trained classification models with publication-quality plots — ROC curves, decision curve analysis (DCA), calibration curves, precision-recall curves, and DeLong's test for AUC differences. Use when the user has prediction CSVs from 2+ models and wants to compare performance. Triggers on phrases like "模型比较", "ROC 对比", "DeLong 检验", "决策曲线", "校准曲线", "model comparison", "compare AUC", "DCA plot", "calibration plot".
---

# HABIT Model Comparison

Generate side-by-side performance comparisons across multiple models. Supports unlimited number of models in one figure.

## CLI

```bash
habit compare --config <config_model_comparison.yaml>
```

## Required Inputs

For each model, user needs a **prediction CSV** containing:
- Subject ID column
- True label column (0/1)
- Predicted probability column (0.0-1.0)
- Predicted class column (0/1)
- Split column (e.g. `train` / `test`) — optional but recommended

These CSVs are typically produced by `habit model` (output: `all_prediction_results.csv` or `<ModelName>_predictions.csv`).

## Standard Config

```yaml
output_dir: ./results/comparison

files_config:
  - path: ./results/clinical_model/all_prediction_results.csv
    name: Clinical Model           # used in legends; must be unique
    subject_id_col: subjID
    label_col: true_label
    prob_col: prob                 # check actual column name in CSV!
    pred_col: pred
    split_col: split

  - path: ./results/radiomics_model/all_prediction_results.csv
    name: Radiomics Model
    subject_id_col: subjID
    label_col: true_label
    prob_col: LogisticRegression_prob   # multi-model output uses prefixed names
    pred_col: LogisticRegression_pred
    split_col: split

  - path: ./results/habitat_model/all_prediction_results.csv
    name: Habitat Model
    subject_id_col: subjID
    label_col: true_label
    prob_col: LogisticRegression_prob
    pred_col: LogisticRegression_pred
    split_col: split

merged_data:
  enabled: true
  save_name: combined_predictions.csv

split:
  enabled: true                    # generate separate plots for train and test

visualization:
  roc:
    enabled: true
    save_name: roc_curves.pdf
    title: ROC Curves
  dca:
    enabled: true
    save_name: decision_curves.pdf
    title: Decision Curves
  calibration:
    enabled: true
    save_name: calibration_curves.pdf
    n_bins: 5
    title: Calibration Curves
  pr_curve:
    enabled: true
    save_name: precision_recall_curves.pdf
    title: Precision-Recall Curves

delong_test:
  enabled: true
  save_name: delong_results.json   # pairwise AUC comparison

metrics:
  basic_metrics:
    enabled: true                  # accuracy, sensitivity, specificity, PPV, NPV
  youden_metrics:
    enabled: true                  # threshold by max Youden index
  target_metrics:
    enabled: true
    targets:
      sensitivity: 0.7
      specificity: 0.7
```

## Reference Templates

- Full annotated: `config_templates/config_model_comparison_annotated.yaml`
- Minimal scaffold: `references/config_comparison_minimal.yaml`

## Decision Helpers

**Q: User has only 1 model — should they use this?**
A: No. This tool is for ≥2 models. Direct them to `habit model` with `is_visualize: true` for single-model plots.

**Q: User's CSVs have different column names — what to do?**
A: That's fine; this tool was designed exactly for that. Each `files_config` entry specifies its own column names independently.

**Q: User wants only specific plots?**
A: Set `enabled: false` on the ones they don't want. All four plot types are independent.

**Q: What does DeLong's test give?**
A: Pairwise comparison of AUCs across models, output as JSON with p-values. Standard for radiomics publications.

## Common Pitfalls

1. **Subject ID mismatch across files** → only common subjects are compared. Tell user how many were dropped.
2. **`prob_col` column not found** → check the actual CSV column names. Multi-model outputs from `habit model` use prefixed names like `LogisticRegression_prob`, `RandomForest_prob`.
3. **`split_col: split`** but CSV has no `split` column → set `split.enabled: false` or add a `split` column to CSVs.
4. **Probabilities outside [0,1]** → some models output logits. Make sure CSV has true probabilities.
5. **`name` field duplicated** across files → must be unique; this is the legend label.

## Output Files

```
output_dir/
├── combined_predictions.csv         # merged data from all models
├── roc_curves.pdf                   # ROC for train + test
├── decision_curves.pdf              # DCA
├── calibration_curves.pdf
├── precision_recall_curves.pdf
├── delong_results.json              # pairwise AUC p-values
├── basic_metrics.csv                # one row per model × split
├── youden_metrics.csv
├── target_metrics.csv
└── model_comparison.log
```

All plots are PDF, vector format, ready for publication. **Labels are in English by project rule** — never put Chinese in plots.

## Verification

- Open `roc_curves.pdf` — each model should appear with a distinct color and AUC value in legend
- Check `delong_results.json` — p-values < 0.05 indicate statistically different AUCs
- `combined_predictions.csv` — useful for further custom analysis (e.g. NRI/IDI)
