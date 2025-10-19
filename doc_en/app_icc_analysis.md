# ICC Analysis Module User Guide

## 1. Overview

The ICC (Intraclass Correlation Coefficient) Analysis module is a specialized tool for evaluating the reliability of quantitative measurements. In research fields like radiomics, it is commonly used for:

- **Test-Retest Reliability**: To assess the consistency of features extracted from the same subjects scanned or processed at different time points.
- **Inter-Observer Agreement**: To evaluate the consistency of features extracted from the same subjects but delineated or analyzed by different observers (or different software, or with different parameters).

By calculating the ICC value for each feature, this tool helps users filter for stable and reliable features, providing a high-quality data foundation for subsequent model building.

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

- This tool exclusively uses the **ICC3** model for calculations. This corresponds to a **Two-Way Random effects model assessing Absolute Agreement**. This is a common and strict standard for evaluating reliability across different time points or observers.

### ICC Value Interpretation Standard

ICC values range from 0 to 1 and can generally be interpreted according to the following standards:
- **< 0.50**: Poor
- **0.50 - 0.75**: Moderate
- **0.75 - 0.90**: Good
- **> 0.90**: Excellent

When performing feature selection, it is common practice to select features with an ICC value greater than 0.75 for subsequent modeling.

### Console Summary

After the script finishes, a summary will be printed to the console. It reports the average ICC for each group and the count and percentage of features that meet the "Good" standard (ICC >= 0.75), helping you to quickly assess overall consistency.