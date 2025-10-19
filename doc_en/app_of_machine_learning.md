# Machine Learning Module User Guide

## Overview

The Machine Learning module provides a complete machine learning workflow, including data preprocessing, feature selection, model training, performance evaluation, and prediction on new data. It supports various machine learning algorithms for classification and regression tasks on radiomics features.

## 🚀 Quick Start

### Using CLI (Recommended) ✨

```bash
# Training mode
habit ml --config config/config_machine_learning.yaml --mode train

# Prediction mode
habit ml --mode predict \
  --model ./ml_data/model_package.pkl \
  --data ./new_data.csv \
  --output ./predictions/

# Prediction with evaluation
habit ml --mode predict \
  --model ./ml_data/model_package.pkl \
  --data ./test_data.csv \
  --evaluate
```

### Using Traditional Scripts (Legacy Compatible)

```bash
# Training mode
python scripts/app_of_machine_learning.py --config <config_file_path> --mode train

# Prediction mode
python scripts/app_of_machine_learning.py --config <config_file_path> --mode predict --model <model_file_path> --data <data_file_path> [--output <output_dir>] [--model_name <model_name>] [--evaluate]
```

## Command-Line Arguments

| Argument | Description |
|---|---|
| `--config` | Path to the YAML configuration file (required) |
| `--mode` | Running mode: 'train' or 'predict', defaults to 'train' |
| `--model` | Path to the model package file (.pkl), required for prediction mode |
| `--data` | Path to the prediction data file (.csv), required for prediction mode |
| `--output` | Path to save prediction results |
| `--model_name` | Specific model name to use for prediction |
| `--evaluate` | Whether to evaluate model performance and generate plots |

## Configuration File Format

`app_of_machine_learning.py` uses a YAML configuration file with the following main sections:

### Basic Configuration

```yaml
# Data and output paths
input:
  - path: <input_data_file_path>
    name: <feature_name_prefix, defaults to empty>
    subject_id_col: <patient_id_column_name>
    label_col: <label_column_name>
    features: <optional_list_of_specific_features>
output: <output_directory_path>
```

### Data Splitting Configuration

```yaml
# Data splitting method: 'random', 'stratified', or 'custom'
split_method: <split_method>
test_size: <test_set_proportion>  # Used when split_method is 'random' or 'stratified'

# Used when split_method is 'custom'
train_ids_file: <path_to_train_ids_file>
test_ids_file: <path_to_test_ids_file>
```

### Feature Selection Configuration

```yaml
feature_selection_methods:
  # Multiple feature selection methods can be configured and will be executed sequentially
  - method: <feature_selection_method_name>
    params:
      <param1>: <value1>
      <param2>: <value2>
      ...
```

### Machine Learning Model Configuration

```yaml
models:
  <model_name>:
    params:
      <param1>: <value1>
      <param2>: <value2>
      ...
```

### Visualization and Saving Configuration

```yaml
is_visualize: <true_or_false_to_generate_performance_visualizations>
is_save_model: <true_or_false_to_save_the_trained_model>
```

### Data Normalization Configuration

```yaml
# Data standardization/normalization configuration
normalization:
  method: <normalization_method_name>  # Supports various normalization methods
  params:
    <param1>: <value1>  # Parameters for the specific normalization method
    <param2>: <value2>
```

## Supported Data Preprocessing Methods

### Missing Value Imputation Methods

- `mean`: Mean imputation
- `median`: Median imputation
- `most_frequent`: Mode imputation
- `constant`: Constant value imputation
- `knn`: K-Nearest Neighbors imputation

## Supported Feature Selection Methods

### Feature Selection Timing
All feature selection methods support a new parameter `before_z_score` to control whether the method is executed before or after Z-score normalization:
- `before_z_score: true` - The method will be executed before Z-score normalization.
- `before_z_score: false` - The method will be executed after Z-score normalization (default behavior).

For variance-sensitive methods (like the variance threshold filter), it is recommended to set `before_z_score: true`, as Z-score normalization sets the variance of all features to 1, rendering the variance filter ineffective.

### ICC (Intraclass Correlation Coefficient) Method
- `method: 'icc'`: Selects features based on their reproducibility.
- Parameters:
  - `icc_results`: Path to the ICC results JSON file.
  - `keys`: Keys of the ICC results to use.
  - `threshold`: Minimum ICC value to retain a feature (0.0-1.0).
  - `before_z_score`: Whether to execute before Z-score normalization, defaults to false.

### VIF (Variance Inflation Factor) Method
- `method: 'vif'`: Removes features with high multicollinearity.
- Parameters:
  - `max_vif`: Maximum allowed VIF value.
  - `visualize`: Whether to generate a visualization of VIF values.
  - `before_z_score`: Whether to execute before Z-score normalization, defaults to false.

### Correlation Method
- `method: 'correlation'`: Removes highly correlated features.
- Parameters:
  - `threshold`: Correlation threshold.
  - `method`: Correlation calculation method ('pearson', 'spearman', or 'kendall').
  - `visualize`: Whether to generate a correlation heatmap.
  - `before_z_score`: Whether to execute before Z-score normalization, defaults to false.

### ANOVA Method
- `method: 'anova'`: Selects features based on the ANOVA F-value.
- Parameters:
  - `p_threshold`: P-value threshold, defaults to 0.05 (selects features with p-value < threshold).
  - `n_features_to_select`: Optional, number of features to select (overrides `p_threshold` if specified).
  - `plot_importance`: Whether to plot feature importances, defaults to True.
  - `before_z_score`: Whether to execute before Z-score normalization, defaults to false.

### Chi2 Method
- `method: 'chi2'`: Selects features based on the chi-squared statistic (for non-negative features in classification problems).
- Parameters:
  - `p_threshold`: P-value threshold, defaults to 0.05.
  - `n_features_to_select`: Optional, number of features to select.
  - `plot_importance`: Whether to plot feature importances, defaults to True.
  - `visualize`: Whether to generate visualizations.
  - `before_z_score`: Defaults to false.

### Statistical Test Method
- `method: 'statistical_test'`: Selects features based on statistical tests (t-test or Mann-Whitney U-test).
- Parameters:
  - `p_threshold`: P-value threshold, defaults to 0.05.
  - `n_features_to_select`: Optional, number of features to select.
  - `normality_test_threshold`: Shapiro-Wilk normality test threshold, defaults to 0.05.
  - `plot_importance`: Whether to plot feature importances, defaults to True.
  - `force_test`: Force a specific test ('ttest' or 'mannwhitney'), defaults to automatic selection.
  - `before_z_score`: Defaults to false.

### Variance Threshold Method
- `method: 'variance'`: Selects features based on variance, removing low-variance features.
- Parameters:
  - `threshold`: Variance threshold, defaults to 0.0 (retains features with variance > threshold).
  - `plot_variances`: Whether to plot feature variances, defaults to True.
  - `before_z_score`: Recommended to be set to `true` as normalization makes all variances 1.
  - `top_k`: Select the top k features with the highest variance (overrides `threshold`).
  - `top_percent`: Select the top percentage of features with the highest variance (0-100, overrides `threshold`).

### mRMR (Minimum Redundancy Maximum Relevance) Method
- `method: 'mrmr'`: Selects features that are highly correlated with the target but have low redundancy among themselves.
- Parameters:
  - `target`: Name of the target variable.
  - `n_features`: Number of features to select (default 10).
  - `method`: MRMR method, 'MIQ' (Mutual Information Quotient) or 'MID' (Mutual Information Difference).
  - `visualize`: Whether to generate visualizations.
  - `outdir`: Output directory for visualizations.
  - `before_z_score`: Defaults to false.

### LASSO (L1 Regularization) Method
- `method: 'lasso'`: Uses L1 regularization for feature selection.
- Parameters:
  - `cv`: Number of cross-validation folds to select the best alpha.
  - `n_alphas`: Number of alpha values to try.
  - `random_state`: Random seed.
  - `visualize`: Whether to generate a visualization of feature coefficients.
  - `before_z_score`: Defaults to false.

### RFECV (Recursive Feature Elimination with Cross-Validation) Method
- `method: 'rfecv'`: Uses recursive feature elimination with cross-validation.
- Parameters:
  - `estimator`: Base estimator, supports various models for classification and regression.
  - `step`: Number of features to remove at each iteration (default 1).
  - `cv`: Number of cross-validation folds (default 5).
  - `scoring`: Evaluation metric.
  - `min_features_to_select`: Minimum number of features to retain (default 1).
  - `n_jobs`: Number of parallel jobs (default -1, use all CPUs).
  - `random_state`: Random seed.
  - `visualize`: Whether to generate a plot of feature count vs. performance.
  - `before_z_score`: Defaults to false.

### Univariate Logistic Regression Method
- `method: 'univariate_logistic'`: Selects features based on p-values from univariate logistic regression.
- Parameters:
  - `threshold`: Maximum p-value threshold for feature selection.
  - `before_z_score`: Defaults to false.

### Stepwise Feature Selection Method
- `method: 'stepwise'`: Uses AIC/BIC criteria for stepwise feature selection.
- Parameters:
  - `direction`: Stepwise direction: 'forward', 'backward', or 'both'.
  - `criterion`: Selection criterion: 'aic', 'bic', or 'pvalue'.
  - `before_z_score`: Defaults to false.

## Supported Machine Learning Models

### Classification Models
- `LogisticRegression`
- `SVM` (Support Vector Machine)
- `RandomForest`
- `XGBoost`

### Regression Models
- `LinearRegression`
- `Ridge`
- `RandomForestRegressor`
- `XGBoostRegressor`

## Supported Evaluation Metrics

### Classification Metrics
- `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`, `sensitivity`, `specificity`

### Regression Metrics
- `r2`, `mae`, `mse`, `rmse`, `explained_variance`

## Complete Configuration Example

```yaml
# Basic configuration
input:
  - path: ./data/radiomics_features.csv
    name: ''
    subject_id_col: 'subjID'
    label_col: 'label'
    features: []
output: ./results/classification_results

# Data splitting
split_method: 'custom'
train_ids_file: './data/train_ids.txt'
test_ids_file: './data/test_ids.txt'

# Feature selection
feature_selection_methods:
  - method: 'variance'
    params:
      threshold: 0.1
      plot_variances: true
      before_z_score: true
      
  - method: 'univariate_logistic'
    params:
      threshold: 0.1
      before_z_score: false
      
  - method: 'stepwise'
    params:
      Rhome: 'E:/software/R'
      direction: 'backward'
      before_z_score: false

# Model configuration
models:
  LogisticRegression:
    params:
      random_state: 42
      max_iter: 1000
      C: 1.0
      penalty: "l2"
      solver: "lbfgs"

# Visualization and saving
is_visualize: true
is_save_model: true
```

## Execution Flow

### Training Mode
1.  Read configuration and data.
2.  Preprocess data.
3.  Split data (train/test sets).
4.  Feature selection (Phase 1, for `before_z_score: true` methods).
5.  Z-score feature normalization.
6.  Feature selection (Phase 2, for `before_z_score: false` or unspecified methods).
7.  Train models (supports multiple models simultaneously).
8.  Evaluate models (calculate metrics, generate plots).
9.  Interpret models (feature importance, SHAP values).
10. Save models and results.

### Prediction Mode
1.  Load the trained model package.
2.  Read new data.
3.  Apply the preprocessing and feature selection pipeline.
4.  Generate predictions using the model.
5.  Optionally, evaluate prediction performance (if true labels are provided).
6.  Save prediction results.

## Output

After execution, the script will generate the following in the specified output directory:

1.  `models/`: Saved trained model files.
2.  `feature_selection/`: Feature selection results.
3.  `evaluation/`: Model evaluation results and plots.
4.  `predictions/`: Prediction results for the test set and new data.
5.  `model_package.pkl`: A complete model package including preprocessing, feature selection, and model parameters.
6.  `results_summary.csv`: A summary of performance metrics for all models.

## Notes

1.  Ensure the input data format is correct and the label column exists.
2.  For classification tasks, labels should be categorical (can be numeric, string, or boolean).
3.  For regression tasks, labels should be continuous numeric values.
4.  When using prediction mode, the new data should contain the same feature columns as the training data (except for the label column).
5.  It is recommended to specify a random seed in the configuration file to ensure reproducibility.

## Supported Normalization/Standardization Methods

- **z_score**: StandardScaler
- **min_max**: MinMaxScaler
- **robust**: RobustScaler
- **max_abs**: MaxAbsScaler
- **normalizer**: Normalizer
- **quantile**: QuantileTransformer
- **power**: PowerTransformer
