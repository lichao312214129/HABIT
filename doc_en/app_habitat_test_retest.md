# Documentation for app_habitat_test_retest_mapper.py

## Overview

`app_habitat_test_retest_mapper.py` is a specialized tool in the HABIT toolkit for evaluating the reproducibility of habitat analysis. This module assesses the stability and reliability of habitat segmentation by analyzing test-retest data (multiple scans of the same patient). This is crucial for establishing the clinical applicability of habitat analysis and validating its potential as a biomarker.

## Usage

```bash
python scripts/app_habitat_test_retest_mapper.py --config <config_file_path>
```

## Command-Line Arguments

| Argument | Description |
|---|---|
| `--config` | Path to the YAML configuration file (required) |

## Configuration File Format

`app_habitat_test_retest_mapper.py` uses a YAML configuration file with the following main sections:

### Basic Configuration

```yaml
# Data paths
test_dir: <path_to_test_data_directory>
retest_dir: <path_to_retest_data_directory>
out_dir: <path_to_output_directory>

# File matching
file_patterns:
  test: <test_file_matching_pattern>
  retest: <retest_file_matching_pattern>
  mapping: <test-retest_mapping_file>

# Analysis configuration
analysis:
  metrics: <list_of_evaluation_metrics>
  visualization: <visualization_settings>
  statistics: <statistical_analysis_settings>
```

### File Matching Configuration

The `file_patterns` section in the configuration file allows for flexible definition of how test and retest files are matched:

```yaml
file_patterns:
  # File matching patterns
  test: "*.nrrd"  # Test data file matching pattern
  retest: "*.nrrd"  # Retest data file matching pattern
  
  # Test-retest mapping method, can be one of the following:
  # 1. Path to a mapping file
  mapping: "path/to/mapping.csv"  
  
  # 2. Filename matching rule
  prefix_pattern:
    test: "patient_{id}_scan1"
    retest: "patient_{id}_scan2"
```

### Analysis Configuration

The analysis configuration section defines the specifics of the test-retest evaluation:

```yaml
analysis:
  # Evaluation metrics
  metrics:
    - "dice"  # Dice coefficient
    - "jaccard"  # Jaccard index
    - "hausdorff"  # Hausdorff distance
    - "volume_ratio"  # Volume ratio
    - "habitat_stability"  # Habitat stability index
  
  # Visualization settings
  visualization:
    overlay_images: true  # Whether to generate overlay images
    difference_maps: true  # Whether to generate difference maps
    colormap: "jet"  # Color map
  
  # Statistical analysis settings
  statistics:
    confidence_level: 0.95  # Confidence level
    permutation_tests: 1000  # Number of permutation tests
```

## Supported Evaluation Metrics

The test-retest analysis supports the following evaluation metrics:

1.  **Dice Coefficient (dice)**: Measures the spatial overlap between two habitat segmentations.
2.  **Jaccard Index (jaccard)**: Another metric for measuring spatial overlap.
3.  **Hausdorff Distance (hausdorff)**: Measures the maximum distance between the boundaries of two habitats.
4.  **Volume Ratio (volume_ratio)**: The ratio of the test volume to the retest volume.
5.  **Volume Difference Percentage (volume_diff_percent)**: The percentage difference in volume.
6.  **Centroid Distance (centroid_distance)**: The distance between the centroids of the habitats.
7.  **Habitat Stability Index (habitat_stability)**: A composite index for evaluating habitat stability.
8.  **Surface Distance (surface_distance)**: The average surface distance.
9.  **Overlap Fraction (overlap_fraction)**: The proportion of the total volume that is overlapping.

## Execution Flow

1.  Load the configuration file and test-retest mapping information.
2.  Read the test and retest data.
3.  Pair the test and retest samples.
4.  Calculate the various evaluation metrics.
5.  Generate visualization results.
6.  Perform statistical analysis.
7.  Generate a report.

## Output

After execution, the script will generate the following in the specified output directory:

1.  `metrics/`: CSV files storing each evaluation metric.
2.  `visualization/`: Habitat overlay images and difference maps.
3.  `statistics/`: Statistical analysis results.
4.  `summary_report.pdf`: A summary report of the test-retest analysis.

## Complete Configuration Example

```yaml
# Basic configuration
test_dir: ./data/test_scans
retest_dir: ./data/retest_scans
out_dir: ./results/test_retest_analysis

# File matching configuration
file_patterns:
  test: "*_habitats.nrrd"
  retest: "*_habitats.nrrd"
  mapping: "./data/test_retest_mapping.csv"

# Analysis configuration
analysis:
  metrics:
    - "dice"
    - "jaccard"
    - "hausdorff"
    - "volume_ratio"
    - "volume_diff_percent"
    - "centroid_distance"
    - "habitat_stability"
    - "surface_distance"
    - "overlap_fraction"
  
  visualization:
    overlay_images: true
    difference_maps: true
    colormap: "jet"
    save_formats: ["png", "pdf"]
    slice_view: "axial"
  
  statistics:
    confidence_level: 0.95
    permutation_tests: 1000
    bland_altman: true
    icc_analysis: true
```

## Mapping File Format

If a CSV file is used to define the test-retest mapping, the format should be:

```csv
test_id,retest_id
patient001_scan1,patient001_scan2
patient002_scan1,patient002_scan2
...
```

## Notes

1.  Ensure that the test and retest data use the same preprocessing and habitat analysis methods.
2.  It is recommended to use test-retest data acquired with the same scanner and scanning parameters.
3.  Habitats with smaller volumes may exhibit greater variability.
4.  The evaluation results should be interpreted in conjunction with their clinical significance.
5.  Habitat stability can be affected by patient status, scanning conditions, and analysis parameters.
