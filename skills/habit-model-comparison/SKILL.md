---
name: habit-model-comparison
description: Compare multiple trained classification models with publication-quality plots — ROC, DCA, calibration, precision-recall, DeLong's AUC test. Use when the user has prediction CSVs from 2+ models and wants side-by-side comparison. Triggers on "模型比较", "ROC 对比", "DeLong 检验", "决策曲线", "校准曲线", "model comparison", "compare AUC", "DCA". Runs `habit compare`.
---

# HABIT Model Comparison

Generate side-by-side performance comparisons across multiple models.
Supports unlimited models in one figure.

## CLI

```bash
habit compare --config <config_model_comparison.yaml>
```

## Required Information

Per model file, the agent needs:

| Field | Notes |
|---|---|
| `path` | prediction CSV path |
| `name` | unique legend label |
| `subject_id_col` | which column has subject IDs |
| `label_col` | true label column |
| `prob_col` | predicted probability column |
| `pred_col` | predicted class column |
| `split_col` | optional; train/test column |

`output_dir` is also required.

These prediction CSVs are typically `<output>/all_prediction_results.csv`
or `<output>/<ModelName>_predictions.csv` from `habit model`.

## Standard config

```yaml
output_dir: ./results/comparison

files_config:
  - path: ./results/clinical_model/all_prediction_results.csv
    name: Clinical Model
    subject_id_col: subjID
    label_col: true_label
    prob_col: prob                 # check actual column name in the CSV!
    pred_col: pred
    split_col: split

  - path: ./results/radiomics_model/all_prediction_results.csv
    name: Radiomics Model
    subject_id_col: subjID
    label_col: true_label
    prob_col: LogisticRegression_prob   # multi-model output uses prefixed names
    pred_col: LogisticRegression_pred
    split_col: split

merged_data:
  enabled: true
  save_name: combined_predictions.csv

split:
  enabled: true                    # generate separate plots for train and test

visualization:
  roc:           {enabled: true, save_name: roc_curves.pdf, title: ROC Curves}
  dca:           {enabled: true, save_name: decision_curves.pdf, title: Decision Curves}
  calibration:   {enabled: true, save_name: calibration_curves.pdf, n_bins: 5, title: Calibration Curves}
  pr_curve:      {enabled: true, save_name: precision_recall_curves.pdf, title: Precision-Recall Curves}

delong_test:
  enabled: true
  save_name: delong_results.json   # pairwise AUC comparison

metrics:
  basic_metrics:  {enabled: true}
  youden_metrics: {enabled: true}
  target_metrics:
    enabled: true
    targets: {sensitivity: 0.7, specificity: 0.7}
```

## Reference templates

Config index: `skills/CONFIG_SOURCES.md`.

| File | Use |
|---|---|
| `config_templates/skill_scaffolds/model_comparison_minimal.yaml` | scaffold |
| `config_templates/skill_scaffolds/model_comparison_two_models.yaml` | 2-model (clinical vs radiomics) |
| `config_templates/skill_scaffolds/model_comparison_three_models.yaml` | 3-model (clinical vs radiomics vs habitat) |
| `references/interpretation_guide.md` | how to read every output |

Full annotated reference: `config_templates/config_model_comparison_annotated.yaml`.

## Decision helpers

**Q: User has only 1 model — should they use this?**
A: No. This tool is for ≥2 models. Direct them to `habit model` with
`is_visualize: true` for single-model plots.

**Q: User's CSVs have different column names — what to do?**
A: That's fine. Each `files_config` entry specifies its own column names
independently.

**Q: User wants only specific plots?**
A: Set `enabled: false` on the ones they don't want. All four plot types
are independent.

**Q: What does DeLong's test give?**
A: Pairwise AUC comparison across models, output as JSON with p-values.
Standard for radiomics publications.

## Common pitfalls

1. **Subject ID mismatch across files** → only common subjects are compared. Tell user how many were dropped.
2. **`prob_col` column not found** → check actual CSV columns. Multi-model outputs use prefixed names like `LogisticRegression_prob`.
3. **`split_col: split` but no split column** → set `split.enabled: false`, OR add a `split` column.
4. **Probabilities outside [0,1]** → some models output logits. Make sure CSV has true probabilities.
5. **`name` field duplicated** across files → must be unique; this is the legend label.

For more, see `habit-troubleshoot/references/errors_ml.md`.

## Output files

```
output_dir/
├── combined_predictions.csv         # merged data from all models
├── roc_curves.pdf                   # train + test
├── decision_curves.pdf              # DCA
├── calibration_curves.pdf
├── precision_recall_curves.pdf
├── delong_results.json              # pairwise AUC p-values
├── basic_metrics.csv                # one row per model × split
├── youden_metrics.csv
├── target_metrics.csv
└── model_comparison.log
```

All plots are PDF (vector format) ready for publication. **English labels
only** by project rule.

## Validation

- Open `roc_curves.pdf` — each model should appear with distinct color and AUC in legend
- Check `delong_results.json` — `p < 0.05` indicates statistically different AUCs
- `combined_predictions.csv` is useful for further custom analysis (NRI / IDI)

For full interpretation guidance, see `references/interpretation_guide.md`.
