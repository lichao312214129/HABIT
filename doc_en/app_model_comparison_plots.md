# Model Comparison and Visualization Module User Guide

## Overview

The Model Comparison module is a dedicated tool for comparing and evaluating the performance of multiple machine learning models. It can read the prediction results from multiple models, merge the evaluation data, generate various performance evaluation charts and metrics, and supports analysis grouped by dataset (e.g., training set, test set).

## 🚀 Quick Start

### Using CLI (Recommended) ✨

```bash
# Use default configuration
habit compare

# Use specified configuration file
habit compare --config config/config_model_comparison.yaml

# Short form
habit compare -c config/config_model_comparison.yaml
```

### Using Traditional Scripts (Legacy Compatible)

```bash
python scripts/app_model_comparison_plots.py --config <config_file_path>
```

## Command-Line Arguments

| Argument | Description |
|---|---|
| `--config` | Path to the YAML configuration file (required) |

## Configuration File Format

`app_model_comparison_plots.py` uses a YAML configuration file with the following main sections:

### Basic Configuration

```yaml
# Output directory configuration
output_dir: "./results/model_comparison"  # Directory where all comparison results will be saved
```

### Model Prediction Files Configuration

```yaml
# Each entry defines a model prediction file to be included in the comparison
files_config:
  - path: "path/to/model_a_predictions.csv"  # Path to the CSV file with model predictions
    model_name: "ModelA"                     # Display name for the model
    subject_id_col: "subjid"                 # Column name for subject identifiers
    label_col: "true_label"                  # Column name for true outcome labels (0/1)
    prob_col: "prediction_probability"       # Column name for predicted probabilities
    pred_col: "prediction_class"             # Column name for discrete predictions (optional)
    split_col: "split"                       # Column name indicating the data split (e.g., "train" or "test")

  - path: "path/to/model_b_predictions.csv"
    model_name: "ModelB"
    # ... other parameters

  - path: "path/to/model_c_predictions.csv"
    model_name: "ModelC"
    # ... other parameters
```

### Merged Data Configuration

```yaml
# Controls how prediction data from different models is merged
merged_data:
  enabled: true                           # Whether to merge predictions from different models into a single dataset
  save_name: "combined_predictions.csv"   # Filename for the saved merged dataset
```

### Split Configuration

```yaml
# Controls whether to analyze training and test sets separately
split:
  enabled: true                           # Whether to generate separate analyses for different data splits
```

### Visualization Configuration

```yaml
# Controls which performance charts are generated and their properties
visualization:
  # ROC curve configuration
  roc:
    enabled: true                         # Whether to generate ROC curve plots
    save_name: "roc_curves.pdf"           # Filename for the saved ROC curve plot
    title: "ROC Curves Comparison"        # Title displayed on the ROC curve plot
  
  # Decision Curve Analysis (DCA) configuration
  dca:
    enabled: true                         # Whether to generate DCA plots
    save_name: "decision_curves.pdf"      # Filename for the saved DCA plot
    title: "Decision Curve Analysis"      # Title for the DCA plot
  
  # Calibration curve configuration
  calibration:
    enabled: true                         # Whether to generate calibration curve plots
    save_name: "calibration_curves.pdf"   # Filename for the saved calibration curve plot
    n_bins: 10                            # Number of bins for calibration curve calculation
    title: "Calibration Curves"           # Title for the calibration curve plot
  
  # Precision-Recall curve configuration
  pr_curve:
    enabled: true                         # Whether to generate PR curve plots
    save_name: "precision_recall_curves.pdf"  # Filename for the saved PR curve plot
    title: "Precision-Recall Curves"      # Title for the PR curve plot
```

### DeLong Test Configuration

```yaml
# Controls statistical comparison between ROC curves using the DeLong test
delong_test:
  enabled: true                           # Whether to perform the DeLong test to compare AUCs
  save_name: "delong_results.json"        # Filename for saving the DeLong test results
```

### Metrics Configuration

```yaml
# Configuration for metrics calculation
metrics:
  # Basic metrics configuration
  basic_metrics:
    enabled: true                         # Whether to calculate basic metrics
  
  # Youden's index metrics configuration
  youden_metrics:
    enabled: true                         # Whether to calculate Youden's index metrics
  
  # Target metrics configuration
  target_metrics:
    enabled: true                         # Whether to calculate target metrics
    targets:                              # Target metrics to calculate
      sensitivity: 0.9                    # Sensitivity target
      specificity: 0.8                    # Specificity target
    save_name: "target_metrics.json"      # Filename for saving target metrics
```

## Functional Modules

### 1. Data Merging and Grouping

The tool can read model predictions from multiple files, merge them into a single dataset, and optionally group them for analysis based on a specified split column (e.g., train/test).

**Core Functions**:
- Read multiple model prediction files.
- Merge predictions from different models into a unified dataset.
- Group data by the `split` column (e.g., training set vs. test set).
- Save the merged dataset for further analysis.

### 2. Generating Visualizations

Generates the following visualizations for each model and each data split (if enabled):

**ROC Curves**:
- Displays the Receiver Operating Characteristic curve for each model.
- Includes the AUC value as a model performance metric.

**Decision Curve Analysis (DCA)**:
- Shows the clinical net benefit of each model at different threshold probabilities.
- Includes "Treat All" and "Treat None" baselines.

**Calibration Curves**:
- Evaluates the calibration of predicted probabilities (i.e., how well predicted probabilities match actual outcomes).
- Supports a custom number of bins.

**Precision-Recall Curves**:
- Shows the trade-off between precision and recall for each model at different thresholds.
- Suitable for handling imbalanced datasets.

### 3. Calculating Performance Metrics

Calculates and compares various performance metrics, supporting calculations at different thresholds:

**Basic Metrics**:
- Accuracy, Precision, Recall, F1-score
- Specificity, Sensitivity
- AUC, log loss, etc.

**Youden's Index Metrics**:
- Calculates metrics at the threshold that maximizes Youden's index (Sensitivity + Specificity - 1).
- For train/test analysis, the threshold determined from the training set is applied to all datasets.

**Target Metrics**:
- Calculates the optimal threshold based on user-specified sensitivity/specificity targets.
- Computes comprehensive performance metrics at the threshold that meets the target.
- Supports searching for a threshold that satisfies multiple targets simultaneously.

### 4. Model Comparison

**DeLong Test**:
- Performs the DeLong test to compare the ROC curves of different models.
- Generates a p-value matrix indicating significant differences between models.

**Metrics Summary**:
- Summarizes the metrics for all models and data splits into a single JSON file.
- Supports comparison and analysis across different metrics.

## Execution Flow

1.  Parse command-line arguments to get the configuration file path.
2.  Create a `ModelComparisonTool` instance and read the configuration.
3.  Read prediction files and prepare the data.
    -   Read the prediction file for each model.
    -   Merge the data and add split information.
    -   Create data subsets by group (if enabled).
4.  Save the merged prediction data.
5.  Perform model evaluation.
    -   Generate visualizations for each data split.
    -   Calculate and compare performance metrics.
    -   Perform the DeLong test to compare ROC curves.
6.  Save all calculated metrics to a JSON file.

## Output

After execution, the script will generate the following in the specified output directory:

1.  `combined_predictions.csv`: A merged dataset containing predictions from all models.
2.  Subdirectories for each split (e.g., "train", "test"), each containing:
    -   ROC curve plots
    -   Decision Curve Analysis plots
    -   Calibration curve plots
    -   Precision-Recall curve plots
    -   DeLong test results
3.  `metrics/metrics.json`: Performance metrics for all models and splits.

## Notes

1.  Ensure all prediction files contain the required columns (`subject_id_col`, `label_col`, `prob_col`).
2.  For grouped analysis, the threshold determined on the training set will be applied to all datasets.
3.  Ensure subject IDs are in a consistent format across all files for correct data merging.
4.  This tool is primarily designed for binary classification problems and does not support multi-class scenarios.
5.  All models in the same analysis should be for the same target variable.
