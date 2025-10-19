# Habitat Feature Extraction Module User Guide

## Overview

The Habitat Feature Extraction module extracts features from segmented habitat maps. This module supports the extraction of various feature types from medical images, including traditional radiomics features, non-radiomics features, whole habitat features, individual habitat features, multiregional spatial interaction (MSI) features, and the Intratumoral Heterogeneity Index (IHI index).

## 🚀 Quick Start

### Using CLI (Recommended) ✨

```bash
# Use default configuration
habit extract-features

# Use specified configuration file
habit extract-features --config config/config_extract_features.yaml

# Short form
habit extract-features -c config/config_extract_features.yaml
```

### Using Traditional Scripts (Legacy Compatible)

```bash
python scripts/app_extracting_habitat_features.py --config <config_file_path>
```

## 📋 Configuration File

**📖 Configuration File Links**:
- 📄 [Current Configuration](../config/config_extract_features.yaml) - Concise configuration for actual use
- 📖 Annotated Version (Coming Soon) - Complete detailed comments `config_extract_features_annotated.yaml`

> 💡 **Tip**: Annotated config files are being prepared. See [Habitat config example](../config/config_getting_habitat_annotated.yaml) for reference format.

## Configuration File Format

`app_extracting_habitat_features.py` uses a YAML configuration file with the following main sections:

```yaml
# Feature extraction settings
params_file_of_non_habitat: <path_to_radiomics_params_for_original_image>
params_file_of_habitat: <path_to_radiomics_params_for_habitat_map>
raw_img_folder: <path_to_original_image_directory>
habitats_map_folder: <path_to_habitat_map_directory>
out_dir: <path_to_output_directory>

# Feature types and processing settings
feature_types: [<list_of_feature_types>]
n_habitats: <number_of_habitats, auto-detects if null>
mode: <operation_mode>

# Program settings
n_processes: <number_of_parallel_processes>
habitat_pattern: <habitat_file_matching_pattern>
debug: <true_or_false_for_debug_mode>
```

## Supported Feature Types

### 1. traditional

Traditional radiomics features extracted from the original image, calculated based on the entire tumor region.

Includes the following main feature classes:
- First Order Statistics
- Shape Features
- Gray Level Co-occurrence Matrix (GLCM)
- Gray Level Run Length Matrix (GLRLM)
- Gray Level Size Zone Matrix (GLSZM)
- Gray Level Dependence Matrix (GLDM)
- Neighboring Gray Tone Difference Matrix (NGTDM)

### 2. non_radiomics

Non-radiomics features (basic statistics) extracted from the habitat segmentation map, including:

- Number of habitats (num_habitats)
- Volume ratio of each habitat (label_X.volume_ratio)
- Number of connected regions for each habitat (label_X.num_regions)

### 3. whole_habitat

Radiomics features extracted from the entire habitat segmentation map, treating the habitat image as a whole. The feature classes are the same as traditional radiomics but are calculated on the habitat map instead of the original image.

### 4. each_habitat

Radiomics features extracted from each habitat individually. Each habitat will have a complete set of radiomics features, named in the format "label_X_feature_name".

### 5. msi (Multiregional Spatial Interaction)

Features that describe the spatial relationships between different habitats, including:
- Number of boundaries between habitats (firstorder_i_and_j)
- Number of adjacency relationships within a habitat (firstorder_i_and_i)
- Normalized spatial interaction features
- Texture features of the MSI matrix (contrast, homogeneity, correlation, energy, etc.)

### 6. ith_score (Intratumoral Heterogeneity Index)

Calculates the Intratumoral Heterogeneity (ITH) Score, ranging from 0 to 1. A higher value indicates greater intratumoral heterogeneity.

## Execution Flow

1. Parse command-line arguments or load the configuration file.
2. Validate the configuration parameters.
3. Create a `HabitatFeatureExtractor` instance.
4. Execute feature extraction based on the specified mode and feature types:
   - If `mode` is 'extract' or 'both', extract features from images.
   - If `mode` is 'parse' or 'both', parse features and generate summaries.
5. Save the extracted features to the output directory.

## Output

After execution, the script will generate the following in the specified output directory:

1. A `features_{timestamp}/` directory containing:
   - CSV files and tables for different feature types.
   - Extracted feature data and statistics.
   - If 'parse' is included in the `mode`, a feature summary and analysis report will be generated.

## Detailed Feature Output Description

Below are the features included in the output CSV files for each type:

### Traditional Radiomics Features (traditional)
(Standard PyRadiomics output)

### Non-Radiomics Features (non_radiomics)
- **num_habitats**: The total number of habitats.
- **label_X.num_regions**: The number of connected regions for the habitat with label X.
- **label_X.volume_ratio**: The proportion of the total habitat volume occupied by the habitat with label X.

### Whole Habitat Features (whole_habitat)
Radiomics features extracted from the entire habitat segmentation map. The feature classes are the same as traditional radiomics but are based on the habitat image.

### Individual Habitat Features (each_habitat)
Radiomics features extracted separately for each habitat region. Each habitat will have a full set of radiomics features, named in the format "label_X_feature_name".

### Multiregional Spatial Interaction (MSI) Features
- **firstorder_i_and_j**: The number of boundaries between habitat `i` and habitat `j`, indicating the strength of their spatial relationship.
- **firstorder_i_and_i**: The number of internal adjacency relationships within habitat `i`, indicating its internal connectivity.
- **firstorder_normalized_i_and_j**: The normalized spatial relationship between habitat `i` and `j`.
- **firstorder_normalized_i_and_i**: The normalized internal connectivity of habitat `i`, indicating its spatial coherence. A higher value suggests the habitat is more spatially contiguous.
- **contrast**: The contrast of the MSI matrix, representing the degree of difference between different habitats.
- **homogeneity**: The homogeneity of the MSI matrix, representing the uniformity of the spatial distribution of habitats.
- **correlation**: The correlation of the MSI matrix, representing the degree of correlation in the spatial distribution of habitats.
- **energy**: The energy of the MSI matrix, representing the regularity and simplicity of the spatial distribution of habitats.

### Intratumoral Heterogeneity (ITH) Index Feature
- **ith_score**: A score from 0 to 1, where a higher value indicates greater intratumoral heterogeneity.

## Complete Configuration Example

```yaml
# Feature extraction settings
params_file_of_non_habitat: ./config/parameter.yaml
params_file_of_habitat: ./config/parameter_habitat.yaml
raw_img_folder: /data/processed_images
habitats_map_folder: /data/habitats_output
out_dir: /data/features_output

# Feature types and processing settings
feature_types: 
  - traditional
  - non_radiomics
  - whole_habitat
  - each_habitat
n_habitats: 5
mode: both

# Program settings
n_processes: 4
habitat_pattern: '*_habitats.nrrd'
debug: false
```

## Feature Extraction Workflow

1. **Image Preparation**:
   - Read original medical images and habitat segmentation maps.
   - Perform necessary preprocessing and normalization.

2. **Feature Calculation**:
   - For `traditional` features: Use PyRadiomics to extract radiomics features from the original image.
   - For `non_radiomics` features: Calculate statistical properties of the habitat segmentation map.
   - For `whole_habitat` features: Use the entire habitat map as input for feature extraction.
   - For `each_habitat` features: Extract features for each habitat individually.
   - For `msi` features: Extract features at multiple spatial scales.

3. **Feature Integration**:
   - Combine all extracted features.
   - Create a feature matrix and labels.

4. **Feature Analysis** (if `mode` includes 'parse'):
   - Calculate feature statistics.
   - Generate feature summaries.
   - Visualize important features.

## Notes

1. Ensure the directory structure for original images and habitat maps is correct, and that habitat map filenames match the original images.
2. The PyRadiomics parameter file should be configured appropriately for your image characteristics.
3. Extracting complex features (like MSI and `each_habitat`) may require more computational resources and time.
4. If `n_habitats` is set to null or 0, the program will automatically detect the number of habitats.
5. Ensure all paths are specified correctly in the configuration, especially when switching between operating systems.
6. The habitat file matching pattern (`habitat_pattern`) defaults to `'*_habitats.nrrd'`. Adjust it according to your actual file naming.
7. Feature types can be selected individually or combined. It is recommended to choose the feature types based on your research objectives.
