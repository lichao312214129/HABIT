# ICC Analysis Module User Guide

## 1. Overview

The ICC (Intraclass Correlation Coefficient) Analysis module is a specialized tool for evaluating the reliability of quantitative measurements. In research fields like radiomics, it is commonly used for:

- **Test-Retest Reliability**: To assess the consistency of features extracted from the same subjects scanned or processed at different time points.
- **Inter-Observer Agreement**: To evaluate the consistency of features extracted from the same subjects but delineated or analyzed by different observers (or different software, or with different parameters).

By calculating the ICC value for each feature, this tool helps users filter for stable and reliable features, providing a high-quality data foundation for subsequent model building.

### Supported Reliability Metrics

This module supports not only ICC but also various reliability assessment metrics:

| Metric Type | Description | Use Case |
|------------|-------------|----------|
| **ICC (6 types)** | Intraclass Correlation Coefficient | Reliability for continuous data |
| **Cohen's Kappa** | Cohen's Kappa coefficient | Agreement for 2 raters with categorical data |
| **Fleiss' Kappa** | Fleiss' Kappa coefficient | Agreement for multiple raters with categorical data |
| **Krippendorff's Alpha** | Krippendorff's Alpha coefficient | Universal reliability metric, handles missing data |

## 2. Quick Start

The primary way to run the analysis is via the `habit` command-line interface (CLI).

### Using the `habit` CLI (Recommended) ✨

```bash
# Run ICC analysis using a configuration file
habit icc --config config/config_icc_analysis.yaml
```

### Using the Legacy Script Wrapper

For backward compatibility, you can also use the `app_icc_analysis.py` script, which is now a simple wrapper around the main CLI command's logic.

```bash
python scripts/app_icc_analysis.py --config config/config_icc_analysis.yaml
```

## 3. Input Data Preparation

Correctly preparing the input data is key to a successful analysis.

- **File Format**: Supports `.csv` and `.xlsx` formats.
- **Data Structure**:
    - The **first column must be the subject ID** and serves as the index.
    - Each subsequent column represents an individual feature.
    - Each row represents a subject.
- **Data Alignment**:
    - The tool automatically identifies **common subject IDs** and **common feature columns** across all files within a group.
    - The ICC analysis will **only be performed on this intersection** of common subjects and features.
    - Therefore, please ensure that subject IDs are named consistently across the files you wish to compare.

**Example `test.csv`**:
```csv
SubjectID,feature_A,feature_B,feature_C
Patient_01,10.5,0.98,150
Patient_02,11.2,0.95,165
Patient_03,9.8,0.99,140
```

**Example `retest.csv`**:
```csv
SubjectID,feature_A,feature_B,feature_D
Patient_01,10.8,0.97,5.5
Patient_02,11.1,0.96,5.8
Patient_04,12.0,0.92,6.1
```
> In this example, the tool will only calculate ICC values for `feature_A` and `feature_B` on subjects `Patient_01` and `Patient_02`.

## 4. Configuration (`config.yaml`) Explained

The tool's behavior is driven by a YAML configuration file.

### Complete Configuration Example
```yaml
# Input Configuration
input:
  # Input mode: "files" or "directories"
  type: "files"

  # File Groups: Used when type is "files"
  # Each sub-list is an independent comparison group.
  file_groups:
    - [/path/to/test_features.csv, /path/to/retest_features.csv]
    - [/path/to/observer1_features.csv, /path/to/observer2_features.csv]

  # Directory List: Used when type is "directories"
  # dir_list:
  #   - /path/to/test_data_dir
  #   - /path/to/retest_data_dir

# Output Configuration
output:
  # Full path for the output JSON file
  path: ./results/icc_analysis/icc_results.json

# Metrics to calculate (e.g., icc2, icc3, cohen, fleiss, krippendorff, all_icc)
metrics:
  - icc3
  - fleiss_kappa

# Number of parallel processes (null uses all available CPU cores)
# Note: Parallel processing is currently handled by the higher-level application if needed.
# This setting might be used in future versions.
processes: 6

# Debug mode (True/False), enables more detailed logging
debug: false
```

### Parameter Descriptions

- **`input.type`**: Defines the input mode.
  - `files`: Directly specify the groups of files to compare. This is more flexible and is the recommended method.
  - `directories`: Specify multiple directories. The tool will automatically find and pair files with the same name within these directories for comparison. This is suitable for highly structured file organizations.

- **`input.file_groups`**: Used when `type` is `files`. This is a list where each element is another list, representing an independent comparison group. For example, you can perform a test-retest analysis and an inter-observer analysis at the same time.

- **`input.dir_list`**: Used when `type` is `directories`. Provide a list of directory paths.

- **`output.path`**: Defines the result output path. The analysis results will be saved here in JSON format.

- **`processes`**: Sets the number of CPU cores to use for parallel computation. If set to `null` or omitted, it will default to using all available cores.

- **`debug`**: If set to `true`, enables debug mode, which will log more details for troubleshooting.

## 5. Interpreting the Output

### JSON Output File

The analysis result is a JSON file with the following structure:

```json
{
    "test_features_vs_retest_features": {
        "feature_A": 0.92,
        "feature_B": 0.85,
        ...
    },
    "observer1_features_vs_observer2_features": {
        "feature_A": 0.78,
        "feature_B": 0.88,
        ...
    }
}
```
- The main keys of the JSON are group names automatically generated from the filenames of the comparison groups.
- Under each group name is a dictionary containing the ICC values for all common features within that group.

### ICC Model Note

This tool uses **ICC3 (i.e., ICC(3,1))** by default. All 6 ICC types are now supported, and users can choose the appropriate type based on their research needs.

#### ICC Types Explained

There are 6 different ICC types, classified according to McGraw & Wong (1996) and Shrout & Fleiss (1979):

| ICC Type | Name | Model | Assessment | Use Case |
|----------|------|-------|------------|----------|
| **ICC1** | ICC(1,1) | One-way Random | Absolute Agreement | Each target rated by **different random raters**, raters not interchangeable |
| **ICC2** | ICC(2,1) | Two-way Random | Absolute Agreement | Raters are a **random sample from a larger population**, assessing absolute values |
| **ICC3** | ICC(3,1) | Two-way Mixed | Consistency | Raters are **fixed**, assessing **relative ranking** rather than absolute values |
| **ICC1k** | ICC(1,k) | One-way Random (Average) | Absolute Agreement | Reliability of the **average** of multiple random raters |
| **ICC2k** | ICC(2,k) | Two-way Random (Average) | Absolute Agreement | Reliability of the **average** of multiple random raters |
| **ICC3k** | ICC(3,k) | Two-way Mixed (Average) | Consistency | Reliability of the **average** of fixed raters |

#### How to Choose an ICC Type?

**Use ICC3 (Default) When:**
- Performing test-retest reliability analysis (same segmentation software/physician performs two measurements)
- Raters are fixed and not randomly selected from a larger population
- You care about whether the relative ranking of feature values is stable, not absolute values

**Use ICC2 When:**
- Raters are a random sample from a larger population (e.g., all radiologists)
- You need to assess absolute agreement (values must be identical, not just ranking)
- Research conclusions need to generalize to a larger population of raters

**Use ICC1 When:**
- Each target is rated by different raters
- Raters are not interchangeable

**Use k-type (ICC1k, ICC2k, ICC3k) When:**
- The final analysis uses the average of multiple raters
- Example: Features extracted from ROIs averaged across 3 physicians' delineations

#### Consistency vs Absolute Agreement

- **Consistency**: Focuses on whether the relative ordering of ratings is consistent. Even if there is a systematic bias (e.g., Rater A always scores 2 points higher than Rater B), ICC will still be high as long as the ranking is consistent.

- **Absolute Agreement**: Requires the absolute values of ratings to match. If there is systematic bias, ICC values will be lower.

### ICC Value Interpretation Standard

ICC values range from 0 to 1 and can generally be interpreted according to the following standards:
- **< 0.50**: Poor
- **0.50 - 0.75**: Moderate
- **0.75 - 0.90**: Good
- **> 0.90**: Excellent

When performing feature selection, it is common practice to select features with an ICC value greater than 0.75 for subsequent modeling.

### Console Summary

After the script finishes, a summary will be printed to the console. It reports the average ICC for each group and the count and percentage of features that meet the "Good" standard (ICC >= 0.75), helping you to quickly assess overall consistency.

## 6. Advanced Usage: Configuring Metrics

All advanced configuration, such as selecting which metrics to calculate, is now handled inside your `config.yaml` file. There are no extra command-line flags.

Simply add a `metrics` list to your configuration file to specify all the metrics you want to run.

**Configuration Example:**
```yaml
input:
  type: "files"
  file_groups:
    - [test.csv, retest.csv]

output:
  path: ./results/icc_results.json

# Specify a list of metrics to calculate
metrics:
  - icc2      # ICC(2,1) - Two-way Random, Absolute Agreement
  - icc3      # ICC(3,1) - Two-way Mixed, Consistency
  - all_icc   # A shortcut to run all 6 ICC types
  - cohen     # Cohen's Kappa (for 2 raters, categorical data)
  - fleiss    # Fleiss' Kappa (for multiple raters, categorical data)
  - krippendorff # Krippendorff's Alpha
```

The output JSON file will contain the results for all specified metrics. For example, if `metrics: [icc2, icc3]` is used, the output for a feature will look like this:

```json
{
    "feature_A": {
        "ICC2": {
            "value": 0.91,
            "metric_type": "ICC2",
            "ci95": [0.83, 0.95],
            ...
        },
        "ICC3": {
            "value": 0.92,
            "metric_type": "ICC3",
            "ci95": [0.85, 0.96],
            ...
        }
    }
}
```

## 7. Python API Usage

The `icc` module has been refactored into a clean, layered architecture. You can hook into different layers depending on your needs.

### 7.1 High-Level API: Running from a Config

This is the simplest way to run the entire analysis programmatically. It mimics the behavior of the `habit icc` command.

```python
from habit.utils.config_utils import load_config
from habit.core.machine_learning.feature_selectors.icc.icc import run_icc_analysis_from_config

# 1. Load configuration from a YAML file
config = load_config('path/to/your/config_icc_analysis.yaml')

# 2. Run the analysis
# This function handles everything: file parsing, calculation, and saving results.
run_icc_analysis_from_config(config)
```

### 7.2 Mid-Level API: Analyzing In-Memory Data

If you already have your data loaded as pandas DataFrames, you can use the `analyze_features` function directly.

```python
import pandas as pd
from habit.core.machine_learning.feature_selectors.icc.icc_analyzer import (
    analyze_features,
    save_results
)

# Assume df1 and df2 are pandas DataFrames with matching indices and columns
# df1 = pd.read_csv(...)
# df2 = pd.read_csv(...)

# 1. Provide the file paths (the function will load them)
file_paths = ['path/to/rater1.csv', 'path/to/rater2.csv']

# 2. Call analyze_features
results = analyze_features(
    file_paths=file_paths,
    metrics=['icc2', 'icc3']
)

# 3. Save the results
save_results(results, 'my_icc_results.json')
```

### 7.3 Low-Level API: Using Metric Classes Directly

For maximum control, you can use the individual metric calculators on your own long-format DataFrame.

```python
import pandas as pd
from habit.core.machine_learning.feature_selectors.icc.icc_analyzer import (
    create_metric,
    ICCType
)

# Prepare a long-format DataFrame
# Structure: each row must contain a subject ID, a rater ID, and a value.
data = pd.DataFrame({
    'subject_id': [1, 1, 2, 2, 3, 3],
    'rater_id': ['A', 'B', 'A', 'B', 'A', 'B'],
    'score': [4.5, 4.7, 3.2, 3.0, 5.1, 5.3]
})

# Method 1: Use the factory function to create a metric calculator
metric = create_metric('icc3')
result = metric.calculate(data, targets='subject_id', raters='rater_id', ratings='score')

print(f"ICC(3,1) = {result.value:.3f}")
print(f"95% CI = [{result.ci95_lower:.3f}, {result.ci95_upper:.3f}]")
print(f"p-value = {result.p_value:.4f}")

# Method 2: Calculate all ICC types at once
multi_icc_metric = create_metric('all_icc')
all_results = multi_icc_metric.calculate(data, targets='subject_id', raters='rater_id', ratings='score')

for icc_type, result in all_results.items():
    print(f"{icc_type}: {result.value:.3f}")
```

## 8. Other Reliability Metrics

### 8.1 Cohen's Kappa

**Use Case**: Agreement assessment for 2 raters with categorical data

**Features**:
- Accounts for chance agreement
- Supports weighted Kappa (for ordinal categories)

**Interpretation Standards**:
| Kappa Value | Agreement Level |
|-------------|-----------------|
| < 0 | Poor (less than chance) |
| 0.01 - 0.20 | Slight |
| 0.21 - 0.40 | Fair |
| 0.41 - 0.60 | Moderate |
| 0.61 - 0.80 | Substantial |
| 0.81 - 1.00 | Almost Perfect |

### 8.2 Fleiss' Kappa

**Use Case**: Agreement assessment for multiple (≥2) raters with categorical data

**Features**:
- Extension of Cohen's Kappa for multiple raters
- Assumes each target is rated by the same number of raters

### 8.3 Krippendorff's Alpha

**Use Case**: Universal reliability metric

**Features**:
- Works with any number of raters
- Supports multiple data types (nominal, ordinal, interval, ratio)
- Handles missing data
- More general than ICC and Kappa

```python
from habit.core.machine_learning.feature_selectors.icc import create_metric

# Create Krippendorff's Alpha metric (interval data)
metric = create_metric('krippendorff', level_of_measurement='interval')
result = metric.calculate(data, 'target', 'reader', 'score')
```

## 9. FAQ

### Q1: Which ICC type should I choose?

For most radiomics studies, we recommend:
- **ICC3 (Default)**: Suitable for test-retest reliability analysis with fixed raters
- **ICC2**: If you need to generalize conclusions to a larger population of raters, or require absolute agreement

### Q2: What does a negative ICC value mean?

ICC values theoretically range from -1 to 1, but negative values usually indicate:
- Agreement between raters is worse than chance
- Data may have issues (e.g., data entry errors)
- Sample size is too small

### Q3: How to handle missing data?

This tool uses `nan_policy='omit'` by default, which ignores data pairs containing missing values. If you have substantial missing data, consider using Krippendorff's Alpha, which is specifically designed to handle missing data.

### Q4: Which metric should I use for continuous vs categorical data?

- **Continuous data**: Use ICC
- **Categorical data (2 raters)**: Use Cohen's Kappa
- **Categorical data (multiple raters)**: Use Fleiss' Kappa
- **Ordinal categorical data**: Use weighted Kappa or Krippendorff's Alpha (ordinal)