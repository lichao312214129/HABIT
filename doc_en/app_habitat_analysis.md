# Habitat Analysis User Guide

## Overview

The Habitat Analysis module identifies and characterizes tumor sub-regions with distinct imaging phenotypes ("habitats"). This module supports two clustering strategies:

### 🎯 Clustering Mode Comparison

| Feature | One-Step | Two-Step |
|---------|----------|----------|
| **Clustering Process** | Direct voxel-to-habitat clustering | First cluster to supervoxels, then to habitats |
| **Clustering Hierarchy** | Single-level (individual) | Two-level (individual + population) |
| **Cluster Numbers** | Optimal number auto-determined per tumor | Fixed supervoxel count, optimizable habitat count |
| **Habitat Labels** | Independent labels per patient | Unified habitat labeling across all patients |
| **Cross-Patient Comparison** | Requires additional correspondence analysis | Direct comparison of same-numbered habitats |
| **Computational Cost** | Lower (individual clustering only) | Higher (individual + population clustering) |
| **Use Cases** | Individual heterogeneity analysis, small studies | Cohort studies, cross-patient habitat pattern recognition |

---

## 🚀 Quick Start

### Using CLI (Recommended)

```bash
# Two-step method (default)
habit habitat --config config/config_getting_habitat.yaml

# One-step method
# First modify clustering_mode: one_step in config file
habit habitat --config config/config_getting_habitat.yaml
```

### Using Traditional Script

```bash
python scripts/app_getting_habitat_map.py --config config/config_getting_habitat.yaml
```

---

## 📋 Configuration File

**📖 Configuration File Links**:
- 📄 [Current Configuration](../config/config_getting_habitat.yaml) - Concise configuration for actual use
- 📖 [Annotated Template](../config_templates/config_getting_habitat_annotated.yaml) - Complete English comments and instructions

### Key Configuration Items

```yaml
HabitatsSegmention:
  # Clustering strategy selection
  clustering_mode: two_step  # one_step or two_step
  
  # Step 1: Individual-level clustering
  supervoxel:
    algorithm: kmeans  # kmeans or gmm
    n_clusters: 50     # Fixed number for two_step
    
    # One-step specific settings
    one_step_settings:
      min_clusters: 2               # Minimum clusters to test
      max_clusters: 10              # Maximum clusters to test
      selection_method: silhouette  # Validation method
      plot_validation_curves: true  # Plot validation curves
  
  # Step 2: Population-level clustering (only for two_step)
  habitat:
    mode: training  # training or testing
    algorithm: kmeans
    max_clusters: 10
    habitat_cluster_selection_method: inertia
    best_n_clusters: 4  # Specify number or null for auto
```

---

## 🎨 One-Step Method Details

### How It Works

1. **Voxel Feature Extraction**: Calculate radiomics features for each voxel
2. **Individual Clustering**: Cluster each patient's tumor independently
3. **Auto Cluster Selection**: Use validation metrics (e.g., silhouette score) to determine optimal number
4. **Generate Personalized Habitat Maps**: Each patient gets unique habitat segmentation

### Cluster Selection Methods

| Method | Description | Optimization |
|--------|-------------|-------------|
| `silhouette` | Silhouette coefficient, measures cluster cohesion and separation | Higher is better |
| `calinski_harabasz` | Variance ratio, between/within cluster variance | Higher is better |
| `davies_bouldin` | Average similarity between clusters | Lower is better |
| `inertia` | Within-cluster sum of squares | Lower is better |

### Output Files

```
output_dir/
├── {subject}_supervoxel.nrrd           # Habitat map (per patient)
├── {subject}_validation_plots/         # Validation curves (if enabled)
│   └── {subject}_cluster_validation.png
├── results_all_samples.csv             # Clustering results for all patients
└── clustering_summary.csv              # Clustering summary statistics
```

### Example Configuration (One-Step)

```yaml
HabitatsSegmention:
  clustering_mode: one_step
  
  supervoxel:
    algorithm: kmeans
    random_state: 42
    
    one_step_settings:
      min_clusters: 3              # Test 3-8 clusters
      max_clusters: 8
      selection_method: silhouette  # Use silhouette score
      plot_validation_curves: true  # Plot validation curves for each patient
```

---

## 📊 Two-Step Method Details

### How It Works

1. **Voxel→Supervoxel**: Cluster each patient's tumor into supervoxels
2. **Supervoxel→Habitat**: Cross-patient clustering to identify common habitat patterns
3. **Population Consistency**: All patients use the same habitat definitions

### Advantages

- ✅ Cross-patient comparability
- ✅ Identify common patterns
- ✅ Suitable for cohort studies
- ✅ Easier statistical analysis

### Output Files

```
output_dir/
├── {subject}_supervoxel.nrrd              # Supervoxel map
├── {subject}_habitat.nrrd                 # Habitat map
├── mean_values_of_all_supervoxels_features.csv  # Supervoxel feature means
├── results_all_samples.csv                # Final results
├── supervoxel2habitat_clustering_model.pkl  # Clustering model
└── habitat_clustering_scores.png          # Clustering evaluation curves
```

### Example Configuration (Two-Step)

```yaml
HabitatsSegmention:
  clustering_mode: two_step
  
  supervoxel:
    algorithm: kmeans
    n_clusters: 50  # Fixed 50 supervoxels per patient
    random_state: 42
  
  habitat:
    mode: training
    algorithm: kmeans
    max_clusters: 10
    habitat_cluster_selection_method: silhouette
    best_n_clusters: null  # Auto-select
```

---

## 🔧 Advanced Usage

### Using Pre-trained Model (Two-Step)

For new test data, use previously trained model:

```yaml
habitat:
  mode: testing  # Switch to testing mode
  # Model automatically loaded from out_dir/supervoxel2habitat_clustering_model.pkl
```

### Multi-processing Acceleration

```yaml
processes: 4  # Use 4 processes for parallel processing
```

### Custom Feature Extraction

`HABIT` provides a flexible feature extraction framework, allowing you to combine multiple features at the voxel level for subsequent habitat clustering analysis. All feature-related configurations are done in the `FeatureConstruction` section of the configuration file.

### Syntax Introduction

The feature extraction syntax is designed as an expression with the format `method(arguments)`.

- **Single Feature**: Directly use one method, e.g., `raw(pre_contrast)`.
- **Multi-feature Combination**: Use multiple features as input to another extractor (like `concat`), for example, `concat(raw(pre_contrast), gabor(pre_contrast))`.
- **Cross-modality Features**: Some feature extractors (like `kinetic`) can accept multiple images from different modalities as input.

All parameters used in a method (like `timestamps` or `params_file`) must be defined in the `params` field.

### Voxel-level Features (`voxel_level`)

This is the first step of habitat analysis, used to extract one or more feature values for each voxel from the original images, forming a feature vector.

The following are all available voxel-level feature extraction methods:

| Method (`method`) | Description | Key Parameters |
| :--- | :--- | :--- |
| `raw` | **Raw Intensity**<br>Directly extracts the original signal intensity value of each voxel in the specified image. This is the most basic and direct feature. | None |
| `kinetic` | **Kinetic Features**<br>Calculates kinetic features, such as wash-in/wash-out slope, from a time-series of images (e.g., multi-phase contrast-enhanced scans). | `timestamps`: A path to an `.xlsx` file that maps each subject to the scan times of the multi-phase images. |
| `voxel_radiomics` | **Voxel-level Radiomics**<br>Uses PyRadiomics to calculate radiomics features in the local neighborhood of each voxel. Can extract various advanced features like texture (e.g., GLCM, Gabor) and intensity distribution. | `params_file`: A path to a PyRadiomics `.yaml` parameter file, used to precisely control the feature classes, filters, and settings to be extracted. |
| `local_entropy` | **Local Entropy**<br>Calculates the local information entropy within the neighborhood of each voxel. This is a measure of local texture complexity and randomness. | `kernel_size`: (Optional, default: 3) Defines the size of the neighborhood for entropy calculation, e.g., `3` for a 3x3x3 cube.<br>`bins`: (Optional, default: 32) The number of bins for histogram calculation. |

---

### Configuration Examples

Here are some specific `yaml` configuration examples showing how to use these methods.

#### Example 1: Using Single Raw Image Intensity

This is the simplest configuration, using only the raw pixel values from the `pre_contrast` image as features.

```yaml
FeatureConstruction:
  voxel_level:
    method: 'raw(pre_contrast)'
    image_names: ['pre_contrast']
    params: {}
```

#### Example 2: Combining Multiple Raw Image Intensities

You can concatenate the raw intensity values from multiple modalities into a single feature vector. Here, we use the `concat` method to combine images from three different phases.

```yaml
FeatureConstruction:
  voxel_level:
    method: 'concat(raw(pre_contrast), raw(LAP), raw(PVP))'
    image_names: ['pre_contrast', 'LAP', 'PVP']
    params: {}
```

#### Example 3: Calculating Kinetic Features

Using the `kinetic` method requires providing a `timestamps` file. The method automatically processes multiple images wrapped in `raw()`.

```yaml
FeatureConstruction:
  voxel_level:
    method: 'kinetic(raw(pre_contrast), raw(LAP), raw(PVP), raw(delay_3min), timestamps)'
    image_names: ['pre_contrast', 'LAP', 'PVP', 'delay_3min']
    params:
      timestamps: './config/scan_times.xlsx' # Path to your timestamps file
```

#### Example 4: Extracting Voxel-level Radiomics Features

Using `voxel_radiomics` requires providing a PyRadiomics parameter file.

```yaml
FeatureConstruction:
  voxel_level:
    method: 'voxel_radiomics(pre_contrast, radiomics_params)'
    image_names: ['pre_contrast']
    params:
      radiomics_params: './config/radiomics_params.yaml' # Path to your radiomics parameter file
```

#### Example 5: Extracting Local Entropy Features

Using `local_entropy` with custom neighborhood size and bin count.

```yaml
FeatureConstruction:
  voxel_level:
    method: 'local_entropy(pre_contrast, kernel_size, bins)'
    image_names: ['pre_contrast']
    params:
      kernel_size: 5
      bins: 64
```

---

## 🎯 Usage Recommendations

### Choose One-Step When...

✅ Focused on individual tumor heterogeneity  
✅ No need for cross-patient comparison  
✅ Sufficient sample size per patient (enough voxels)  
✅ Exploratory study to understand individual differences  

### Choose Two-Step When...

✅ Need cross-patient statistical analysis  
✅ Identify common habitat types across population  
✅ Build reusable habitat model  
✅ Conduct cohort studies or clinical prediction  

---

## 🐛 FAQ

### Q1: In one-step mode, each patient has different number of clusters. How to compare?

**A**: One-step focuses on intra-tumor heterogeneity, not inter-patient comparison. If comparison needed:
- Compare cluster numbers (as heterogeneity indicator)
- Extract features from each habitat for statistics
- Use two-step method to get unified habitat definitions

### Q2: How to choose appropriate cluster number range?

**A**: Recommendations:
- Minimum: 2-3 (need clear separation)
- Maximum: 10-15 (avoid over-segmentation)
- Consider tumor size (smaller tumors use fewer clusters)

### Q3: Validation curves look unstable?

**A**: Possible reasons:
- Insufficient sample size (too few voxels)
- Inappropriate feature selection
- Try different validation methods
- Increase `n_init` parameter of clustering algorithm

---

## 📚 Related Documentation

- [Feature Extraction Configuration](./app_extracting_habitat_features.md)
- [ICC Reproducibility Analysis](./app_icc_analysis.md)
- [CLI Usage Guide](../HABIT_CLI.md)

---

## 📖 References

**Two-Step Method (Classic Habitat)**:
- Wu J, et al. "Intratumoral spatial heterogeneity at perfusion MR imaging predicts recurrence-free survival in locally advanced breast cancer treated with neoadjuvant chemotherapy." Radiology, 2018.

**One-Step Method (Personalized Analysis)**:
- Nomogram for Predicting Neoadjuvant Chemotherapy  Response in Breast Cancer Using MRI-based Intratumoral  Heterogeneity Quantification

---

*Last updated: 2025-10-19*

