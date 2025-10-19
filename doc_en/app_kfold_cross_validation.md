# K-Fold Cross-Validation Guide

This document describes how to use the K-Fold cross-validation functionality in the HABIT package for model evaluation and training.

## Overview

K-Fold cross-validation is a widely used model evaluation method that divides the dataset into K subsets (folds). In each iteration, K-1 subsets are used for training and the remaining subset is used for validation. This process repeats K times, with each subset serving as the validation set once.

## 🚀 Quick Start

### Using CLI (Recommended) ✨

```bash
# Use default configuration
habit kfold

# Use specified configuration file
habit kfold --config config/config_machine_learning_kfold.yaml

# Short form
habit kfold -c config/config_machine_learning_kfold.yaml
```

### Using Traditional Scripts (Legacy Compatible)

```bash
python scripts/app_kfold_cv.py --config config/config_machine_learning_kfold.yaml
```

### Advantages of K-Fold Cross-Validation

1. **More reliable performance estimation**: Compared to a single train-test split, K-Fold provides more stable and reliable performance estimates
2. **Efficient data usage**: Every sample is used for both training and validation, especially suitable for small datasets
3. **Reduced randomness**: Multiple validations reduce the impact of random splitting
4. **Model stability assessment**: Standard deviation across folds evaluates model stability on different data subsets

### Differences from Standard Modeling Pipeline

| Feature | Standard Pipeline | K-Fold Cross-Validation |
|---------|------------------|------------------------|
| Data splitting | Single split (train + test) | K splits |
| Training iterations | 1 per model | K per model |
| Evaluation metrics | Single test set result | Mean ± Std across K folds |
| Feature selection | On entire training set | Independently within each fold (avoids data leakage) |
| Use cases | Large datasets, rapid prototyping | Small datasets, reliable evaluation |
| Computation time | Fast | Slow (K times) |

## 📋 Configuration File

**📖 Configuration File Links**:
- 📄 [Current Configuration](../config/config_machine_learning_kfold.yaml) - Actual configuration file in use
- 🇬🇧 Detailed English Configuration (Coming Soon) - Complete English comments and instructions
- 🇨🇳 详细中文配置 (Coming Soon) - Includes complete Chinese comments and instructions

> 💡 **Tip**: Detailed annotated configuration files are being prepared. Please refer to the configuration instructions below for now.

K-Fold cross-validation uses a dedicated configuration file: `config_machine_learning_kfold.yaml`.

### Configuration Example

```yaml
# Data Input Configuration
input:
  - path: ./ml_data/breast_cancer_dataset.csv
    name: clinical_
    subject_id_col: subjID
    label_col: label
    features:

# Output Directory Configuration
output: ./ml_data/kfold_results

# K-Fold Cross-Validation Configuration
n_splits: 5          # Number of folds (commonly 5 or 10)
stratified: true     # Whether to use stratified k-fold (recommended for imbalanced data)
random_state: 42     # Random seed for reproducibility

# Normalization Configuration
normalization:
  method: z_score    # Normalization method

# Feature Selection Configuration
# Note: Feature selection is performed within each fold to avoid data leakage
feature_selection_methods:
  - method: correlation
    params:
      threshold: 0.80
      method: spearman
      visualize: false
      before_z_score: false

# Model Configuration
models:
  LogisticRegression:
    params:
      random_state: 42
      max_iter: 1000
      C: 1.0
  
  RandomForest:
    params:
      random_state: 42
      n_estimators: 100
  
  XGBoost:
    params:
      random_state: 42
      n_estimators: 100
      max_depth: 3

# Visualization and Saving Configuration
is_visualize: false
is_save_model: false
```

### Important Parameters

#### n_splits
- **Description**: Number of folds in K-Fold
- **Common values**: 5 or 10
- **Selection guidelines**:
  - Small datasets (<100 samples): Use 10-fold or more
  - Medium datasets (100-1000 samples): Use 5-fold or 10-fold
  - Large datasets (>1000 samples): Use 5-fold
  - Very small samples: Consider Leave-One-Out CV (n_splits = number of samples)

#### stratified
- **Description**: Whether to use stratified K-Fold
- **Default**: true
- **Recommendations**:
  - Imbalanced classes: Must use true
  - Balanced classes: Recommended to use true
  - Regression problems: Set to false

## Usage

### 1. Command-Line Usage

Create a Python script (e.g., `run_kfold_cv.py`):

```python
from habit.core.machine_learning.machine_learning_kfold import run_kfold_modeling
import yaml

# Load configuration
with open('config/config_machine_learning_kfold.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Run k-fold cross-validation
modeling = run_kfold_modeling(config)

print("K-Fold cross-validation completed!")
```

Then run:
```bash
python run_kfold_cv.py
```

### 2. Interactive Usage

```python
from habit.core.machine_learning.machine_learning_kfold import ModelingKFold
import yaml

# Load configuration
with open('config/config_machine_learning_kfold.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create modeling instance
modeling = ModelingKFold(config)

# Run pipeline step by step
modeling.read_data()
modeling.preprocess_data()
modeling._create_kfold_splits()
modeling.run_kfold_cv()

# Access results
print("Aggregated results:", modeling.cv_results['aggregated'])
```

## Output Results

### 1. Console Output

During execution, the following will be displayed:
- Processing progress for each fold
- Performance metrics for each model in each fold
- Aggregated results across all folds

Example output:
```
================================================================================
Processing Fold 1/5
================================================================================
Selected 25 features for fold 1

Training LogisticRegression on fold 1...
LogisticRegression - Val AUC: 0.8523, Acc: 0.7800

Training RandomForest on fold 1...
RandomForest - Val AUC: 0.8912, Acc: 0.8200

...

LogisticRegression - Overall AUC: 0.8456 ± 0.0234
RandomForest - Overall AUC: 0.8834 ± 0.0189
```

### 2. Result Files

#### kfold_cv_results.json
Contains complete cross-validation results:
```json
{
  "n_splits": 5,
  "stratified": true,
  "aggregated": {
    "LogisticRegression": {
      "fold_metrics": {
        "auc_mean": 0.8456,
        "auc_std": 0.0234,
        "accuracy_mean": 0.7820,
        "accuracy_std": 0.0156
      },
      "overall_metrics": {
        "auc": 0.8467,
        "accuracy": 0.7830,
        "sensitivity": 0.8100,
        "specificity": 0.7560
      }
    }
  }
}
```

#### kfold_performance_summary.csv
Performance summary table:

| Model | AUC_mean | AUC_std | AUC_overall | Accuracy_mean | Accuracy_std | Accuracy_overall |
|-------|----------|---------|-------------|---------------|--------------|------------------|
| LogisticRegression | 0.8456 | 0.0234 | 0.8467 | 0.7820 | 0.0156 | 0.7830 |
| RandomForest | 0.8834 | 0.0189 | 0.8845 | 0.8230 | 0.0123 | 0.8240 |

### 3. Fold-Level Output

Each fold creates a subdirectory in the output directory:
```
kfold_results/
├── fold_1/
│   └── feature_selection/
│       └── correlation_heatmap.pdf
├── fold_2/
│   └── feature_selection/
├── ...
├── kfold_cv_results.json
└── kfold_performance_summary.csv
```

## Interpreting Results

### Performance Metric Explanation

1. **xxx_mean**: Average across folds
   - Represents model's average performance on different data subsets
   
2. **xxx_std**: Standard deviation across folds
   - Indicates model stability
   - Smaller std = more stable model
   
3. **xxx_overall**: Overall metric using all fold predictions
   - Calculated by combining predictions from all folds
   - Usually more accurate than mean

### How to Choose a Model

1. **Prioritize overall metrics**: 
   - Model with highest overall AUC is usually the best choice

2. **Consider stability**:
   - Models with smaller std are more reliable
   - Example: Model A (AUC=0.85±0.02) is better than Model B (AUC=0.86±0.08)

3. **Balance performance and complexity**:
   - If two models have similar performance, choose the simpler one
   - Simple models are easier to interpret and deploy

## Avoiding Data Leakage

A key issue in K-Fold cross-validation is **data leakage**. This implementation avoids data leakage through:

### 1. In-Fold Feature Selection

**Wrong approach** (causes data leakage):
```python
# Feature selection on all data
selected_features = feature_selection(X, y)

# Then k-fold cross-validation
for train_idx, val_idx in kfold.split(X):
    X_train = X[train_idx][selected_features]
    X_val = X[val_idx][selected_features]
    # Train and evaluate...
```

**Correct approach** (used in this implementation):
```python
# Independent feature selection within each fold
for train_idx, val_idx in kfold.split(X):
    X_train = X[train_idx]
    X_val = X[val_idx]
    
    # Select features only on training set
    selected_features = feature_selection(X_train, y_train)
    
    # Apply to both training and validation sets
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    # Train and evaluate...
```

### 2. In-Fold Data Normalization

Normalization must also be performed independently within each fold:

```python
for train_idx, val_idx in kfold.split(X):
    # Fit scaler on training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform validation set using training scaler
    X_val_scaled = scaler.transform(X_val)
```

## Best Practices

### 1. Choose Appropriate K Value

```python
# Small datasets (< 100 samples)
config['n_splits'] = 10  # or more

# Medium datasets (100-1000 samples)
config['n_splits'] = 5   # or 10

# Large datasets (> 1000 samples)
config['n_splits'] = 5
```

### 2. Use Stratified K-Fold

For classification problems, especially with class imbalance:
```yaml
stratified: true  # Maintain class proportions in each fold
```

### 3. Fix Random Seed

Ensure reproducibility:
```yaml
random_state: 42  # Fixed random seed
```

### 4. Reasonable Number of Models

- Don't train too many models simultaneously (will be slow)
- Recommend 3-5 models for comparison

### 5. Feature Selection Strategy

```yaml
feature_selection_methods:
  # First remove highly correlated features
  - method: correlation
    params:
      threshold: 0.90
      
  # Then perform importance filtering
  - method: anova
    params:
      n_features_to_select: 20
```

## Choosing Between K-Fold and Standard Pipeline

### When to Use K-Fold Cross-Validation

1. **Small datasets** (< 500 samples)
2. **Need reliable performance evaluation**
3. **Model selection and hyperparameter tuning**
4. **Publishing papers requiring rigorous evaluation**
5. **High data collection costs**

### When to Use Standard Pipeline

1. **Large datasets** (> 10000 samples)
2. **Rapid prototyping**
3. **Limited computational resources**
4. **Model and parameters already determined**
5. **Real-time prediction systems**

## FAQ

### Q: How long does K-Fold cross-validation take?
A: Approximately K times the standard pipeline (e.g., 5-fold takes 5x time). Can be accelerated by:
   - Reducing number of models
   - Reducing feature selection methods
   - Using simpler models
   - Reducing K value

### Q: Will selected features differ across folds?
A: Yes, this is normal. Each fold has a slightly different training set, so feature selection results will vary. Final feature importance should be synthesized across all folds.

### Q: How to save models from K-Fold cross-validation?
A: K-Fold is mainly for evaluation, models are typically not saved. If you need a final model:
   1. Use K-Fold to select best model and parameters
   2. Retrain that model on all data
   3. Save the retrained model

### Q: What does large standard deviation indicate?
A: Large std indicates:
   - Model performance is unstable across different data subsets
   - Model may be sensitive to data distribution
   - Consider:
     - Increasing sample size
     - Using more stable models
     - Improving feature engineering
     - Adjusting hyperparameters

### Q: Difference between K-Fold and Leave-One-Out?
A: 
   - Leave-One-Out is a special case of K-Fold (K = number of samples)
   - Advantage: Maximum data utilization
   - Disadvantage: Extremely high computational cost
   - Recommendation: Only use for very small samples (<30)

## Compatibility with Model Comparison Tool

K-Fold cross-validation results can be directly used for model comparison analysis without additional format conversion.

### Output Files

After running K-Fold cross-validation, the following files are automatically generated:

#### Required Outputs

1. **kfold_cv_results.json** - Detailed cross-validation results
2. **kfold_performance_summary.csv** - Performance summary table
3. **all_prediction_results.csv** - Compatible prediction results (for model comparison)

#### Visualization Outputs (when `is_visualize: true`)

4. **kfold_roc_curves.pdf** - ROC curve comparison plot
5. **kfold_calibration_curves.pdf** - Calibration curves
6. **kfold_dca_curves.pdf** - Decision Curve Analysis (DCA)
7. **kfold_confusion_matrix_{model_name}.pdf** - Confusion matrices for each model

**Configure Visualization**:
```yaml
# In configuration file
is_visualize: true  # Enable visualization (default: false)
is_save_model: false  # Whether to save models from each fold
```

### Prediction Results Format

The `all_prediction_results.csv` file format is fully compatible with the standard machine learning pipeline:

```csv
subject_id,true_label,split,RandomForest_pred,RandomForest_prob,XGBoost_pred,XGBoost_prob
patient_001,1,Test set,1,0.85,1,0.88
patient_002,0,Test set,0,0.23,0,0.19
...
```

**Note**: 
- The `split` column is labeled as "Test set", indicating these are validation set predictions
- In K-Fold, each sample is predicted once as a validation set in one fold
- Therefore, all samples have `split` as "Test set"
- When `is_visualize: true`, visualization plots (ROC, DCA, calibration curves, confusion matrices) are automatically generated
- Visualization plots are generated using aggregated predictions across all folds, comprehensively reflecting model performance

### Using the Model Comparison Tool

After running K-Fold cross-validation, you can directly use the model comparison tool:

```bash
# 1. Run K-Fold cross-validation
python -m habit kfold -c config/config_machine_learning_kfold.yaml

# 2. Use comparison tool to analyze results
python -m habit compare -c config/config_model_comparison.yaml
```

Configure in `config_model_comparison.yaml`:

```yaml
output_dir: ./results/model_comparison

files_config:
  - path: ./ml_data/kfold_results/all_prediction_results.csv
    model_names:
      - RandomForest
      - LogisticRegression
      - XGBoost
    prob_suffix: "_prob"
    pred_suffix: "_pred"
    label_col: "true_label"
    split_col: "split"

split:
  enabled: false  # K-Fold results usually don't need splitting

statistical_tests:
  enabled: true
  methods:
    - delong
    - mcnemar

visualization:
  enabled: true
  plot_types:
    - roc
    - calibration
    - confusion
    - dca
```

### Comparing Different Training Methods

You can also compare K-Fold cross-validation with standard train/test split results:

```yaml
files_config:
  # Standard training results
  - path: ./results/standard_ml/all_prediction_results.csv
    model_names: [RandomForest, XGBoost]
    # ...
  
  # K-Fold validation results
  - path: ./results/kfold/all_prediction_results.csv
    model_names: [RandomForest, XGBoost]
    # ...
```

This allows you to evaluate:
- Impact of different training strategies on model performance
- Model stability under different evaluation methods
- Choose the most suitable training method for your data

## Related Documentation

- [Machine Learning Configuration Guide](app_of_machine_learning.md)
- [Machine Learning Models Guide](app_machine_learning_models.md)
- [Model Comparison Tool](app_model_comparison_plots.md)
- [Feature Selection Methods](../habit/core/machine_learning/feature_selectors/README.md)
- [CLI Usage Guide](../HABIT_CLI.md)

