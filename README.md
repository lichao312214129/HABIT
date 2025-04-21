# Habitat Analysis Tools Documentation

This document provides detailed instructions for using the habitat analysis tools. The tools are designed to work together in a pipeline for analyzing medical image habitats.

## Table of Contents
1. [Habitat Map Generation](#habitat-map-generation)
2. [Habitat Feature Extraction](#habitat-feature-extraction)
3. [Test-Retest Habitat Mapping](#test-retest-habitat-mapping)
4. [ICC Analysis](#icc-analysis)
5. [Machine Learning Modeling](#machine-learning-modeling)
6. [Model Comparison and Visualization](#model-comparison-and-visualization)

## Habitat Map Generation

The `app_getting_habitat_map.py` tool generates habitat maps from medical images.

### Usage
```bash
python app_getting_habitat_map.py --config <config_file>
```

### Configuration File (YAML/JSON) Parameters
```yaml
# Basic parameters
data_dir: <path_to_data_directory>  # Root directory containing input images
out_dir: <path_to_output_directory>  # Directory to save results
processes: 4  # Number of parallel processes
plot_curves: true  # Whether to generate visualization plots
random_state: 42  # Random seed for reproducibility
images_dir: "images"  # Directory name for input images
masks_dir: "masks"  # Directory name for input masks

# Feature construction parameters
FeatureConstruction:
  method: simple  # Available methods:
    # - "kinetic": Kinetic features
    # - "simple": Original features
    # - "custom": User-defined feature extraction, please refer to the custom_feature_extractor_template.py
  image_names: [<image1>, <image2>]  # List of image names to process
  params: {}  # Additional parameters for feature extraction

# Habitat segmentation parameters
HabitatsSegmention:
  supervoxel:
    algorithm: kmeans  # Available algorithms:
      # - "kmeans": K-means clustering
      # - "gmm": Gaussian Mixture Model
    n_clusters: 50  # Number of supervoxel clusters
  
  habitat:
    algorithm: kmeans  # Available algorithms (same as supervoxel)
    max_clusters: 10  # Maximum number of habitat clusters
    min_clusters: 2  # Minimum number of habitat clusters
    habitat_cluster_selection_method: <method_name>  # Available methods:
      # - "silhouette": Silhouette score
      # - "calinski_harabasz": Calinski-Harabasz index
      # - "elbow": Elbow method
    best_n_clusters: <number>  # Optional: Predefined number of clusters
```

### Parameters
- `--config`: Path to configuration file (required)
- `--debug`: Enable debug mode (optional)

## Habitat Feature Extraction

The `app_extracting_habitat_features.py` tool extracts various types of features from habitat maps.

### Usage
```bash
python app_extracting_habitat_features.py --config <config_file>
```

### Configuration File (YAML/JSON) Parameters
```yaml
# File paths
params_file_of_non_habitat: <path_to_radiomics_params>  # Radiomics parameters for original images
params_file_of_habitat: <path_to_habitat_params>  # Radiomics parameters for habitat maps
raw_img_folder: <path_to_original_images>  # Directory containing original images
habitats_map_folder: <path_to_habitat_maps>  # Directory containing habitat maps
out_dir: <path_to_output_directory>  # Directory to save extracted features

# Processing parameters
n_processes: 4  # Number of parallel processes
habitat_pattern: "*_habitats.nrrd"  # File pattern for habitat maps:
  # - "*_habitats.nrrd": Original habitat maps
  # - "*_habitats_remapped.nrrd": Remapped habitat maps

# Feature extraction parameters
feature_types: ["traditional", "non_radiomics", "whole_habitat", "each_habitat", "msi"]
  # Available feature types:
  # - "traditional": Traditional radiomics features
  # - "non_radiomics": Non-radiomics features
  # - "whole_habitat": Features from whole habitat map
  # - "each_habitat": Features from individual habitats
  # - "msi": Multi-Scale Image features

n_habitats: 0  # Number of habitats to process:
  # - 0: Auto-detect from habitat maps
  # - >0: Process specified number of habitats

mode: "both"  # Operation mode:
  # - "extract": Extract features only
  # - "parse": Parse features only
  # - "both": Extract and parse features
```

### Parameters
- `--config`: Path to configuration file (required)
- `--debug`: Enable debug mode (optional)

## Test-Retest Habitat Mapping

The `app_habitat_test_retest_mapper.py` tool aligns habitat labels between test and retest scans.

### Usage
```bash
python app_habitat_test_retest_mapper.py --test-habitat-table <test_table> --retest-habitat-table <retest_table> --input-dir <input_dir> --out-dir <output_dir>
```

### Parameters
- `--test-habitat-table`: Path to test group habitat feature table (CSV/Excel)
- `--retest-habitat-table`: Path to retest group habitat feature table (CSV/Excel)
- `--features`: List of feature names for similarity calculation (optional)
- `--similarity-method`: Similarity calculation method (default: "pearson")
  - Available methods:
    - "pearson": Pearson correlation
    - "spearman": Spearman correlation
    - "kendall": Kendall's tau
    - "euclidean": Euclidean distance
    - "cosine": Cosine similarity
    - "manhattan": Manhattan distance
    - "chebyshev": Chebyshev distance
- `--input-dir`: Directory containing retest NRRD files
- `--out-dir`: Output directory for processed files
- `--processes`: Number of processes to use (default: 4)
- `--debug`: Enable debug logging

## ICC Analysis

The `app_icc_analysis.py` tool calculates Intraclass Correlation Coefficient (ICC) values between test-retest features.

### Usage
```bash
python app_icc_analysis.py --files "file1.csv,file2.csv,file3.csv;file4.csv,file5.csv,file6.csv" --output icc_results.json
# OR
python app_icc_analysis.py --dirs "dir1,dir2,dir3" --output icc_results.json
```

### Parameters
- `--files`: List of files to analyze (format: "file1.csv,file2.csv;file3.csv,file4.csv")
- `--dirs`: List of directories to analyze (format: "dir1,dir2,dir3")
- `--processes`: Number of processes (default: all available CPUs)
- `--output`: Output JSON file path (default: "icc_results.json")
- `--debug`: Enable debug mode

## Machine Learning Modeling

The `app_of_machine_learning.py` tool handles training and prediction using machine learning models.

### Usage
```bash
# Training mode
python app_of_machine_learning.py --config <config_file> --mode train

# Prediction mode
python app_of_machine_learning.py --config <config_file> --mode predict --model <model_file> --data <data_file> --output <output_dir>
```

### Configuration File (YAML) Parameters
```yaml
# Data configuration
data:
  path: <path_to_data_file>  # Path to input data file (CSV)
  label_col: <label_column>  # Name of the label column
  subject_id_col: <id_column>  # Name of the subject ID column
  feature_cols: [<feature1>, <feature2>]  # List of feature columns to use

# Environment configuration
environment:
  n_processes: 4  # Number of parallel processes
  random_state: 42  # Random seed for reproducibility
  debug: false  # Enable debug mode

# Data splitting configuration
split:
  method: "train_test_split"  # Available methods:
    # - "train_test_split": Simple train-test split
    # - "kfold": K-fold cross-validation
    # - "stratified_kfold": Stratified K-fold cross-validation
  test_size: 0.2  # Proportion of test set
  n_splits: 5  # Number of splits for cross-validation

# Feature selection configuration
feature_selection:
  method: "variance_threshold"  # Available methods:
    # - "variance_threshold": Remove low variance features
    # - "select_k_best": Select K best features
    # - "rfe": Recursive Feature Elimination
    # - "lasso": Lasso regression
  params:  # Parameters specific to the selected method
    threshold: 0.01  # For variance_threshold
    k: 10  # For select_k_best

# Model configuration
models:
  - name: "logistic_regression"  # Available models:
    # - "logistic_regression": Logistic Regression
    # - "random_forest": Random Forest
    # - "xgboost": XGBoost
    # - "svm": Support Vector Machine
    params:  # Model-specific parameters
      C: 1.0  # For logistic_regression
      max_depth: 3  # For random_forest and xgboost
      n_estimators: 100  # For random_forest and xgboost
```

### Parameters
- `--config`: Path to YAML config file (required)
- `--mode`: Operation mode ("train" or "predict")
- `--model`: Path to model package file (.pkl) for prediction
- `--data`: Path to data file (.csv) for prediction
- `--output`: Path to save prediction results
- `--model_name`: Name of specific model to use for prediction
- `--evaluate`: Whether to evaluate model performance and generate plots

## Model Comparison and Visualization

The `app_model_comparison_plots.py` tool compares and visualizes multiple machine learning models.

### Usage
```bash
python app_model_comparison_plots.py --config <config_file>
```

### Configuration File (YAML) Parameters
```yaml
# Output configuration
output_dir: <path_to_output_directory>  # Directory to save results
plot_dpi: 300  # DPI for saved plots
save_format: "png"  # Available formats: "png", "pdf", "svg"

# File configuration
files_config:
  - path: <path_to_prediction_file1>
    model_name: <model1_name>
    pred_col: <prediction_column1>
    subject_id_col: <id_column>
    split_col: <split_column>  # Optional: Column for data splitting

# Split configuration
split:
  enabled: true  # Whether to split data by groups
  groups: [<group1>, <group2>]  # Optional: Specific groups to analyze
  method: "stratified"  # Available methods:
    # - "stratified": Stratified splitting
    # - "random": Random splitting
    # - "custom": Custom splitting

# Visualization configuration
visualization:
  roc_curve: true  # Generate ROC curves
  pr_curve: true  # Generate Precision-Recall curves
  calibration_curve: true  # Generate calibration curves
  confusion_matrix: true  # Generate confusion matrices
  dca_curve: true  # Generate Decision Curve Analysis curves

# Statistical analysis configuration
statistics:
  delong_test: true  # Perform DeLong test for AUC comparison
  confidence_intervals: true  # Calculate confidence intervals
  significance_level: 0.05  # Significance level for tests

# Merged data configuration
merged_data:
  enabled: true  # Whether to save merged data
  save_name: "combined_predictions.csv"  # Name for merged data file
```

### Parameters
- `--config`: Path to configuration file (required)

## Notes
1. All tools support debug mode for detailed logging
2. Configuration files can be in either YAML or JSON format
3. Make sure to have all required dependencies installed
4. For large datasets, consider adjusting the number of processes based on available system resources
5. Output directories will be created automatically if they don't exist 