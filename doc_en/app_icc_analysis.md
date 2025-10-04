# Documentation for app_icc_analysis.py

## Overview

`app_icc_analysis.py` is a dedicated tool in the HABIT toolkit for calculating the Intraclass Correlation Coefficient (ICC). This module supports the analysis of test-retest reliability, inter-observer agreement, and various other reliability assessments for habitat features. ICC is a standard statistical method for evaluating the reliability of quantitative measurements and is of great importance in radiomics research.

## Usage

```bash
python scripts/app_icc_analysis.py --config <config_file_path>
```

## Command-Line Arguments

| Argument | Description |
|---|---|
| `--config` | Path to the YAML configuration file (required) |

## Configuration File Format

`app_icc_analysis.py` uses a YAML configuration file with the following main sections:

### Basic Configuration

```yaml
# Data path
input:
  type: <input_type>  # "files" or "directory"
  path: <input_path>  # List of files or a directory
  pattern: <file_matching_pattern>  # Used when type is "directory"

# Output configuration
output:
  dir: <output_directory>
  report_name: <report_name>

# ICC analysis configuration
icc:
  type: <icc_type>
  confidence_level: <confidence_level>
  outlier_removal: <outlier_handling_method>
```

### ICC Analysis Type Configuration

```yaml
icc:
  # ICC type, one of the following:
  # - "test_retest": Test-retest reliability
  # - "inter_observer": Inter-observer agreement
  # - "intra_observer": Intra-observer agreement
  # - "multi_reader": Multi-reader multi-case
  type: "test_retest"
  
  # ICC model configuration
  model: <model_type>  # "oneway", "twoway"
  unit: <unit>  # "single", "average"
  effect: <effect>  # "random", "fixed", "mixed"
  
  # Consistency/absolute agreement configuration
  definition: <definition>  # "consistency", "absolute_agreement"
  
  # Confidence level
  confidence_level: 0.95
  
  # Outlier removal
  outlier_removal:
    method: <method>  # "none", "zscore", "iqr", "modified_zscore"
    threshold: <threshold>  # Method-specific threshold value
```

### Grouping and Feature Configuration

```yaml
# Data grouping configuration
grouping:
  method: <grouping_method>  # "filename_pattern", "explicit_mapping", "column"
  pattern: <filename_pattern>  # For filename_pattern method
  mapping_file: <mapping_file>  # For explicit_mapping method
  id_column: <id_column_name>  # For column method
  group_column: <group_column_name>  # For column method

# Feature configuration
features:
  # Feature columns to include
  include: <list_of_features_to_include>  # Can be a list or "*" for all
  
  # Feature columns to exclude
  exclude: <list_of_features_to_exclude>
  
  # Feature categorization
  categories:
    - name: <category_1>
      features: <list_of_features_in_category_1>
    - name: <category_2>
      features: <list_of_features_in_category_2>
```

## Supported ICC Types

The ICC analysis supports the following types:

1.  **Test-Retest Reliability (test_retest)**: Assesses the consistency of measurements on the same subject at different time points.
2.  **Inter-Observer Agreement (inter_observer)**: Assesses the consistency of measurements of the same object by different observers.
3.  **Intra-Observer Agreement (intra_observer)**: Assesses the consistency of measurements of the same object by the same observer at different time points.
4.  **Multi-Reader Multi-Case (multi_reader)**: Agreement analysis for multiple readers evaluating multiple cases.

## ICC Model Parameters

### Model Type (model)

-   **oneway**: One-way random-effects model, suitable for situations where each subject is rated by a different set of raters.
-   **twoway**: Two-way model, suitable for situations where the same set of raters evaluates all subjects.

### Unit (unit)

-   **single**: Assesses the reliability of a single rating.
-   **average**: Assesses the reliability of an average rating.

### Effect (effect)

-   **random**: Raters are considered a random sample.
-   **fixed**: Raters are considered a fixed factor.
-   **mixed**: Mixed-effects model.

### Definition (definition)

-   **consistency**: Assesses the relative consistency of ratings.
-   **absolute_agreement**: Assesses the absolute agreement of ratings.

## Outlier Handling Methods

-   **none**: No outlier handling.
-   **zscore**: Identifies and handles outliers based on Z-scores.
-   **iqr**: Identifies and handles outliers based on the interquartile range.
-   **modified_zscore**: Uses the modified Z-score method.

## Execution Flow

1.  Load the configuration file.
2.  Read the input data.
3.  Group the data according to the configured grouping method.
4.  Calculate ICC for the selected features.
5.  Generate an ICC analysis report and visualization results.
6.  Save the results to the output directory.

## Output

After execution, the script will generate the following in the specified output directory:

1.  `icc_results.csv`: ICC values and their confidence intervals for all features.
2.  `icc_summary.csv`: ICC results summarized by feature category.
3.  `icc_plots/`: ICC visualization charts, including:
    -   ICC bar plots
    -   Bland-Altman plots
    -   Scatter correlation plots
    -   Heatmaps
4.  `icc_report.pdf`: A complete ICC analysis report.

## Complete Configuration Examples

### Test-Retest ICC Analysis

```yaml
# Basic configuration
input:
  type: "directory"
  path: "./data/test_retest_features"
  pattern: "*.csv"

output:
  dir: "./results/icc_analysis"
  report_name: "test_retest_icc_report"

# ICC analysis configuration
icc:
  type: "test_retest"
  model: "twoway"
  unit: "single"
  effect: "random"
  definition: "absolute_agreement"
  confidence_level: 0.95
  outlier_removal:
    method: "iqr"
    threshold: 1.5

# Grouping configuration
grouping:
  method: "filename_pattern"
  pattern: "features_{subject_id}_{timepoint}.csv"

# Feature configuration
features:
  include: "*"
  exclude: ["patient_id", "scan_date", "study_id"]
  categories:
    - name: "Shape Features"
      features: ["shape_volume", "shape_surface_area", "shape_sphericity"]
    - name: "First-Order Features"
      features: ["firstorder_mean", "firstorder_std", "firstorder_entropy"]
    - name: "Texture Features"
      features: ["glcm_*", "glrlm_*", "glszm_*"]
```

### Inter-Observer ICC Analysis

```yaml
# Basic configuration
input:
  type: "files"
  path: 
    - "./data/observer1/features.csv"
    - "./data/observer2/features.csv"
    - "./data/observer3/features.csv"

output:
  dir: "./results/inter_observer_icc"
  report_name: "inter_observer_icc_report"

# ICC analysis configuration
icc:
  type: "inter_observer"
  model: "twoway"
  unit: "single"
  effect: "random"
  definition: "absolute_agreement"
  confidence_level: 0.95
  outlier_removal:
    method: "none"

# Grouping configuration
grouping:
  method: "column"
  id_column: "patient_id"
  group_column: "observer"

# Feature configuration
features:
  include: ["intensity_*", "texture_*", "shape_*"]
  exclude: []
```

## Interpretation of Results

The interpretation of ICC values typically follows these standards:

-   **< 0.50**: Poor reliability
-   **0.50-0.75**: Moderate reliability
-   **0.75-0.90**: Good reliability
-   **> 0.90**: Excellent reliability

## Notes

1.  Ensure the data format is correct, especially the pairing of test-retest or observer data.
2.  Choose the ICC type and model parameters that are appropriate for your study design.
3.  For datasets with a large number of features, consider analyzing them by feature category.
4.  ICC analysis results should be interpreted in the context of clinical significance and research objectives.
5.  When severe outliers are present, using an appropriate outlier handling method can improve the reliability of the results.
