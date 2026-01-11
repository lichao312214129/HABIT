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

### Using CLI (Recommended) ✨

```bash
# Run ICC analysis using a configuration file
habit icc --config config/config_icc_analysis.yaml
```

### Using Traditional Script

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
  # Each sub-list is an independent comparison group, typically containing 2 or more files.
  file_groups:
    - [/path/to/test_features.csv, /path/to/retest_features.csv]
    - [/path/to/observer1_features.csv, /path/to/observer2_features.csv]

  # Directory List: Used when type is "directories"
  # The tool will automatically match files with the same name in these directories and group them for comparison.
  # dir_list:
  #   - /path/to/test_data_dir
  #   - /path/to/retest_data_dir

# Output Configuration
output:
  # Full path for the output JSON file
  path: ./results/icc_analysis/icc_results.json

# Number of parallel processes (null uses all available CPU cores)
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

## 6. Advanced Usage

### 6.1 Specifying ICC Type

You can specify the ICC type to calculate via configuration file or command-line arguments:

**Configuration File:**
```yaml
input:
  type: "files"
  file_groups:
    - [test.csv, retest.csv]

output:
  path: ./results/icc_results.json

# Specify metrics to calculate
metrics:
  - icc2      # ICC(2,1) - Two-way Random, Absolute Agreement
  - icc3      # ICC(3,1) - Two-way Mixed, Consistency (default)

# Return full results (including CI and p-value)
full_results: true
```

**Command Line:**
```bash
# Calculate ICC(2,1)
python -m habit.core.machine_learning.feature_selectors.icc.icc \
    --files "test.csv,retest.csv" \
    --metrics "icc2" \
    --output results.json

# Calculate multiple ICC types
python -m habit.core.machine_learning.feature_selectors.icc.icc \
    --files "test.csv,retest.csv" \
    --metrics "icc2,icc3,icc2k,icc3k" \
    --full \
    --output results.json

# Calculate all 6 ICC types
python -m habit.core.machine_learning.feature_selectors.icc.icc \
    --files "test.csv,retest.csv" \
    --metrics "multi_icc" \
    --full \
    --output results.json
```

### 6.2 Calculating Kappa Coefficients

For categorical data (e.g., grading, staging), use Kappa coefficients:

```bash
# Cohen's Kappa (2 raters)
python -m habit.core.machine_learning.feature_selectors.icc.icc \
    --files "rater1.csv,rater2.csv" \
    --metrics "cohen_kappa" \
    --output kappa_results.json

# Fleiss' Kappa (multiple raters)
python -m habit.core.machine_learning.feature_selectors.icc.icc \
    --files "rater1.csv,rater2.csv,rater3.csv" \
    --metrics "fleiss_kappa" \
    --output kappa_results.json
```

### 6.3 Returning Full Results

Use the `--full` flag to get complete results including confidence intervals and p-values:

```bash
python -m habit.core.machine_learning.feature_selectors.icc.icc \
    --files "test.csv,retest.csv" \
    --metrics "icc3" \
    --full \
    --output results.json
```

Output format:
```json
{
    "test_vs_retest": {
        "feature_A": {
            "value": 0.92,
            "metric_type": "ICC3",
            "ci95": [0.85, 0.96],
            "p_value": 0.0001,
            "additional_info": {
                "F": 24.5,
                "df1": 49,
                "df2": 49
            }
        }
    }
}
```

## 7. Python API Usage

### 7.1 Basic Usage

```python
from habit.core.machine_learning.feature_selectors.icc import (
    calculate_icc,
    calculate_reliability_metrics,
    configure_logger
)

# Configure logger
logger = configure_logger('./output/results.json')

# Basic ICC calculation (default ICC3, backward compatible)
results = calculate_icc(
    files_list=['rater1.csv', 'rater2.csv'],
    logger=logger
)

# Specify ICC type
results = calculate_icc(
    files_list=['rater1.csv', 'rater2.csv'],
    logger=logger,
    icc_type='icc2'  # Use ICC(2,1)
)
```

### 7.2 Calculating Multiple Metrics

```python
from habit.core.machine_learning.feature_selectors.icc import (
    calculate_reliability_metrics,
    configure_logger
)

logger = configure_logger('./output/results.json')

# Calculate multiple metrics with full results
results = calculate_reliability_metrics(
    files_list=['rater1.csv', 'rater2.csv'],
    logger=logger,
    metrics=['icc2', 'icc3', 'fleiss_kappa'],
    return_full_results=True
)

# Iterate through results
for group_name, features in results.items():
    print(f"\n=== {group_name} ===")
    for feature_name, metric_results in features.items():
        print(f"\n{feature_name}:")
        for metric_name, result in metric_results.items():
            if isinstance(result, dict):
                print(f"  {metric_name}: {result['value']:.3f}")
                if 'ci95' in result:
                    print(f"    95% CI: [{result['ci95'][0]:.3f}, {result['ci95'][1]:.3f}]")
            else:
                print(f"  {metric_name}: {result:.3f}")
```

### 7.3 Using Metric Classes Directly

```python
import pandas as pd
from habit.core.machine_learning.feature_selectors.icc import (
    ICCMetric,
    ICCType,
    MultiICCMetric,
    CohenKappaMetric,
    FleissKappaMetric,
    create_metric
)

# Prepare long-format data
# Structure: each row contains target (subject), reader (rater), rating (value)
data = pd.DataFrame({
    'target': [1, 1, 2, 2, 3, 3],
    'reader': [1, 2, 1, 2, 1, 2],
    'score': [4.5, 4.7, 3.2, 3.0, 5.1, 5.3]
})

# Method 1: Use factory function
metric = create_metric('icc3')
result = metric.calculate(data, targets='target', raters='reader', ratings='score')
print(f"ICC(3,1) = {result.value:.3f}")
print(f"95% CI = [{result.ci95_lower:.3f}, {result.ci95_upper:.3f}]")
print(f"p-value = {result.p_value:.4f}")

# Method 2: Instantiate metric class directly
icc_metric = ICCMetric(icc_type=ICCType.ICC2)
result = icc_metric.calculate(data, 'target', 'reader', 'score')

# Method 3: Calculate all ICC types
multi_icc = MultiICCMetric()
results = multi_icc.calculate(data, 'target', 'reader', 'score')
for icc_type, result in results.items():
    print(f"{icc_type}: {result.value:.3f}")

# Method 4: Cohen's Kappa (for categorical data)
kappa_metric = CohenKappaMetric(weights='quadratic')  # Weighted Kappa
result = kappa_metric.calculate(data, 'target', 'reader', 'category')
```

### 7.4 Custom Metrics

You can create custom metrics by extending the `BaseReliabilityMetric` class:

```python
from habit.core.machine_learning.feature_selectors.icc import (
    BaseReliabilityMetric,
    MetricResult,
    register_metric
)
import pandas as pd

@register_metric("my_custom_metric")
class MyCustomMetric(BaseReliabilityMetric):
    """Custom reliability metric"""
    
    @property
    def name(self) -> str:
        return "MyCustomMetric"
    
    def validate_data(self, data, targets, raters, ratings) -> bool:
        # Validate data format
        return True
    
    def calculate(self, data, targets, raters, ratings, **kwargs) -> MetricResult:
        # Implement your calculation logic
        value = 0.85  # Example value
        return MetricResult(
            value=value,
            ci95_lower=0.80,
            ci95_upper=0.90,
            metric_type=self.name
        )

# Use custom metric
from habit.core.machine_learning.feature_selectors.icc import create_metric
metric = create_metric("my_custom_metric")
result = metric.calculate(data, 'target', 'reader', 'score')
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