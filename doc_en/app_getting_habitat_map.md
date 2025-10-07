# Documentation for app_getting_habitat_map.py

## Overview

`app_getting_habitat_map.py` is the main entry point of the HABIT toolkit for performing tumor habitat analysis. This module implements the complete workflow for medical image habitat analysis through feature extraction, supervoxel clustering, and habitat clustering. The script supports both command-line and graphical user interface interactions and can execute the habitat analysis process based on a user-specified configuration file.

## Usage

```bash
python scripts/app_getting_habitat_map.py --config <config_file_path>
```

## Command-Line Arguments

| Argument | Description |
|---|---|
| `--config` | Path to the configuration file |
| `--debug` | Enable debug mode (optional) |

## Configuration File Format

`app_getting_habitat_map.py` uses a YAML configuration file with the following main sections:

### Basic Configuration

```yaml
# Data paths
data_dir: <path_to_data_directory>
out_dir: <path_to_output_directory>

# General settings
processes: <number_of_parallel_processes>
plot_curves: <true_or_false_to_generate_curve_plots>
random_state: <random_seed>
debug: <true_or_false_to_enable_debug_mode>
```

### Feature Extraction Configuration (FeatureConstruction)

The feature extraction section is divided into voxel-level and supervoxel-level parts:

```yaml
FeatureConstruction:
  voxel_level:
    method: <feature_extraction_method_expression>
    params:
      <method_specific_parameters>
  
  supervoxel_level:
    supervoxel_file_keyword: <supervoxel_file_keyword>
    method: <supervoxel_level_feature_extraction_method>
    params:
      <method_specific_parameters>

  preprocessing:
    methods:
      - method: <preprocessing_method_1>
        <method_1_parameters>
      - method: <preprocessing_method_2>
        <method_2_parameters>
```

### Habitat Segmentation Configuration (HabitatsSegmention)

```yaml
HabitatsSegmention:
  # Supervoxel clustering settings
  supervoxel:
    algorithm: <clustering_algorithm>
    n_clusters: <number_of_supervoxels>
    random_state: <random_seed>
    max_iter: <maximum_iterations>
    n_init: <number_of_initializations>
  
  # Habitat clustering settings
  habitat:
    mode: <mode>  # 'training' or 'testing'
    algorithm: <clustering_algorithm>
    max_clusters: <maximum_number_of_habitats>
    min_clusters: <minimum_number_of_habitats>  # Optional, defaults to 2
    habitat_cluster_selection_method: <cluster_number_selection_method>
    best_n_clusters: <number_of_habitats>  # Set to null for automatic selection
    random_state: <random_seed>
    max_iter: <maximum_iterations>
    n_init: <number_of_initializations>
```

## Supported Feature Extraction Methods

### Voxel-Level Feature Extraction (voxel_level)

Voxel-level feature extraction supports the following methods:

#### 1. kinetic - Dynamic Enhancement Features

Extracts dynamic enhancement features based on time-series data, such as wash-in slope, wash-out slope, etc.

```yaml
method: kinetic(raw(pre_contrast), raw(LAP), raw(PVP), raw(delay_3min), timestamps)
params:
  timestamps: <path_to_timestamps_file>
```

**Parameters:**
- `timestamps`: Path to an Excel file containing the scan times for each patient at various phases.

**Output Features:**
- `wash_in_slope`: Wash-in slope
- `wash_out_slope_lap_pvp`: Wash-out slope from the arterial phase to the portal venous phase
- `wash_out_slope_pvp_dp`: Wash-out slope from the portal venous phase to the delayed phase

#### 2. voxel_radiomics - Voxel-Level Radiomics Features

Uses PyRadiomics to extract voxel-level radiomics features.

```yaml
method: concat(voxel_radiomics(image_name))
params:
  params_voxel_radiomics: <path_to_parameter_file>
  kernelRadius: <kernel_radius>
```

**Parameters:**
- `params_voxel_radiomics`: Path to the PyRadiomics parameter file.
- `kernelRadius`: The radius of the kernel used for extracting local features, defaults to 1.

**Output Features:**
Depending on the settings in the PyRadiomics parameter file, may include:
- First-order statistics (e.g., mean, standard deviation)
- Shape features
- Gray Level Co-occurrence Matrix (GLCM) features
- Gray Level Run Length Matrix (GLRLM) features
- Gray Level Size Zone Matrix (GLSZM) features
- etc.

#### 3. local_entropy - Local Entropy Features

Calculates the local entropy in the region around each voxel as a measure of tissue heterogeneity.

```yaml
method: local_entropy(image_name)
params:
  kernel_size: <local_region_size>
  bins: <number_of_histogram_bins>
```

**Parameters:**
- `kernel_size`: The size of the local region, representing the side length of the cube around the voxel (default is 3).
- `bins`: The number of histogram bins used for calculating entropy (default is 32).

**Output Features:**
- `local_entropy`: The local entropy value for each voxel.

**Example:**
```yaml
method: concat(local_entropy(PVP), voxel_radiomics(PVP))
params:
  kernel_size: 5
  bins: 32
```

**Use Cases:**
- Quantifying tumor heterogeneity
- Analyzing microenvironment complexity
- Identifying tissue boundaries and transition zones

#### 4. concat - Feature Concatenation

Concatenates the results of multiple feature extraction methods.

```yaml
method: concat(method1(params), method2(params), ...)
```

**Example:**
```yaml
method: concat(voxel_radiomics(pre_contrast), voxel_radiomics(PVP))
```

#### 5. raw - Raw Image Data

Extracts raw image data, typically used as input for other methods.

```yaml
method: raw(image_name)
```

### Supervoxel-Level Feature Extraction (supervoxel_level)

Supervoxel-level feature extraction supports the following methods:

#### 1. supervoxel_radiomics - Supervoxel-Level Radiomics Features

Extracts radiomics features for each supervoxel.

```yaml
method: supervoxel_radiomics(image_name, params_file)
params:
  params_file: <path_to_parameter_file>
```

**Parameters:**
- `params_file`: Path to the PyRadiomics parameter file.

#### 2. mean_voxel_features - Mean of Voxel Features

Calculates the average of voxel features within each supervoxel.

```yaml
method: mean_voxel_features()
```

### Feature Preprocessing Methods (preprocessing)

Supported preprocessing methods include:

#### 1. minmax - Min-Max Normalization

```yaml
method: minmax
global_normalize: <true_or_false_for_global_normalization>
```

#### 2. standard - Standardization

```yaml
method: standard
global_normalize: <true_or_false_for_global_normalization>
```

#### 3. robust - Robust Normalization

```yaml
method: robust
global_normalize: <true_or_false_for_global_normalization>
```

#### 4. winsorize - Winsorize Transformation

```yaml
method: winsorize
winsor_limits: [<lower_limit>, <upper_limit>]
global_normalize: <true_or_false_for_global_normalization>
```

#### 5. binning - Binning Discretization

Discretizes continuous feature values into a specified number of bins, which helps reduce noise and the impact of outliers.

```yaml
method: binning
n_bins: <number_of_bins>
strategy: <binning_strategy>
global_normalize: <true_or_false_for_global_binning>
```

**Parameters:**
- `n_bins`: Number of bins, defaults to 5.
- `strategy`: Binning strategy, supports the following options:
  - `uniform`: Equal-width binning (default).
  - `quantile`: Equal-frequency binning (quantile-based).
  - `kmeans`: Binning based on K-means clustering.
- `global_normalize`: Whether to perform global binning across all samples, defaults to true.

**Use Cases:**
- Reducing feature noise
- Handling outliers
- Simplifying feature distributions
- Improving model stability

**Example:**
```yaml
method: binning
n_bins: 8
strategy: quantile
global_normalize: true
```

## Supported Clustering Algorithms

### Supervoxel Clustering (supervoxel)

- `kmeans`: K-means clustering
- `gmm`: Gaussian Mixture Model
- `spectral`: Spectral clustering
- `hierarchical`: Hierarchical clustering
- `mean_shift`: Mean Shift clustering
- `dbscan`: DBSCAN density clustering
- `affinity_propagation`: Affinity Propagation clustering

### Habitat Clustering (habitat)

- `kmeans`: K-means clustering (most common)
- `gmm`: Gaussian Mixture Model

### Cluster Number Selection Methods (habitat_cluster_selection_method)

- `inertia`: Inertia (for K-means)
- `aic`: Akaike Information Criterion (for GMM)
- `bic`: Bayesian Information Criterion (for GMM)
- `silhouette`: Silhouette Coefficient (for all algorithms)
- `calinski_harabasz`: Calinski-Harabasz Index (for all algorithms)
- `davies_bouldin`: Davies-Bouldin Index (for all algorithms)

## Complete Configuration Example

```yaml
# Data paths
data_dir: your_data_dir
out_dir: your_output_dir

# Feature extraction settings
FeatureConstruction:
  voxel_level:
    method: concat(raw(pre_contrast), raw(LAP), raw(PVP), raw(delay_3min))
    params:
      params_voxel_radiomics: ./config/params_voxel_radiomics.yaml
      kernelRadius: 2
      timestamps: F:\work\research\radiomics_TLSs\data\scan_time_of_phases.xlsx
      kernel_size: 5
      bins: 32

  supervoxel_level:
    supervoxel_file_keyword: '*_supervoxel.nrrd'
    method: mean_voxel_features()
    params:
      params_file: ./config/parameter.yaml

  preprocessing:
    methods:
      - method: minmax
        global_normalize: true
      - method: winsorize
        winsor_limits: [0.05, 0.05]
        global_normalize: true
      - method: binning
        n_bins: 8
        strategy: quantile
        global_normalize: true

# Habitat segmentation settings
HabitatsSegmention:
  supervoxel:
    algorithm: kmeans
    n_clusters: 50
    random_state: 42
    max_iter: 300
    n_init: 10
  
  habitat:
    mode: training  # 'training' for new model, 'testing' for existing model
    algorithm: kmeans
    max_clusters: 10
    min_clusters: 2  # Optional, minimum number of clusters, defaults to 2
    habitat_cluster_selection_method: inertia
    best_n_clusters: null  # Set to a number to specify, or null for auto-selection
    random_state: 42
    max_iter: 300
    n_init: 10

# General settings
processes: 2
plot_curves: true
random_state: 42
debug: false

## Execution Flow

1. Parse command-line arguments or select a configuration file via the GUI.
2. Load the configuration file and data.
3. Initialize the feature extractor and clustering algorithms.
4. Perform voxel-level feature extraction.
5. Perform supervoxel clustering.
6. Perform habitat clustering.
7. Save the results and analysis plots.

## Output

After execution, the script will generate the following in the specified output directory:

1. Supervoxel clustering results (supervoxel segmentation map for each sample).
2. Habitat clustering results (common habitats for all samples).
3. Feature data tables.
4. Analysis reports and clustering evaluation plots.

## Notes

1. Ensure the data directory structure is correct.
2. If using dynamic enhancement feature extraction, a correct `timestamps` file must be provided.
3. Parameter file paths can be relative to the script's running directory.
4. It is recommended to adjust clustering algorithms and parameters based on the characteristics of your data.
