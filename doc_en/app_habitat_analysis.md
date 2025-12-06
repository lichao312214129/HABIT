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

### Cluster Selection Strategy

The code employs different strategies to select the optimal number of clusters based on the evaluation metric:

| Method | Description | Selection Logic |
|--------|-------------|-----------------|
| `silhouette` | Silhouette Coefficient | **Max Principle**: Higher is better. Selects the number of clusters with the highest score. |
| `calinski_harabasz` | Variance Ratio | **Max Principle**: Higher is better. Selects the number of clusters with the highest score. |
| `inertia` | Sum of Squared Errors (SSE) | **Elbow Method**: Lower is better. The code calculates the **second-order difference** of the score curve to find the "elbow point" (maximum curvature). |
| `bic` / `aic` | Information Criteria (GMM) | **Elbow Method**: Lower is better. Uses the same second-order difference logic to find the elbow point. |
| `davies_bouldin` | Davies-Bouldin Index | **Min Principle**: Lower is better. |

> **💡 Why use Elbow Method for Inertia but Min Principle for Davies-Bouldin?**
>
> *   **Inertia (SSE)**: Monotonically decreases as the number of clusters increases (reaching 0 when clusters = samples). Finding the absolute minimum leads to overfitting. Therefore, we look for the "elbow point" where the gain diminishes.
> *   **Davies-Bouldin**: Considers both intra-cluster compactness (numerator) and inter-cluster separation (denominator). As clusters increase, while compactness improves, separation might decrease (smaller denominator). Thus, it typically has a distinct global minimum and does not require the Elbow Method.

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

The configuration for feature extraction uses nested function calls. The core logic is **"Single-modality Extraction -> Multi-modality Combination"**.

1.  **Single-modality Extractors**:
    *   Purpose: Extract features from a single image modality.
    *   Examples: `raw(...)`, `voxel_radiomics(...)`, `local_entropy(...)`.
    *   **Note**: The output of these functions cannot be used directly as the final feature; it must be passed to a multi-modality combiner.

2.  **Multi-modality Combiners**:
    *   Purpose: Receive outputs from one or more single-modality extractors and integrate them into the final feature vector.
    *   Examples: `concat(...)` (most common, direct concatenation), `kinetic(...)` (for time-series).
    *   **Core Principle**: Even if extracting features from only one modality, it must be wrapped by a multi-modality combiner.

**✨ Extensibility**: The `HABIT` framework supports custom single-modality extractors and multi-modality combiners, allowing you to write your own functions as needed.

### Voxel-level Features (`voxel_level`)

This is the first step of habitat analysis, used to extract one or more feature values for each voxel from the original images, forming a feature vector.

Here are the currently built-in functions:

**Single-modality Extractors**:

| Method | Description | Key Parameters |
| :--- | :--- | :--- |
| `raw` | **Raw Intensity**<br>Directly extracts the original signal intensity value of each voxel in the specified image. | None |
| `voxel_radiomics` | **Voxel-level Radiomics**<br>Uses PyRadiomics to calculate radiomics features in the local neighborhood of each voxel. | `params_file`: Path to PyRadiomics parameter file |
| `local_entropy` | **Local Entropy**<br>Calculates the local information entropy within the neighborhood of each voxel. | `kernel_size`: Neighborhood size<br>`bins`: Number of bins |

**Multi-modality Combiners**:

| Method | Description | Note |
| :--- | :--- | :--- |
| `concat` | **Concatenation**<br>Concatenates input features directly along the channel dimension. | Most generic combiner. Must be used even for a single input to establish feature format. |
| `kinetic` | **Kinetic Features**<br>Specialized for multi-phase time-series data to calculate perfusion parameters. | **Note**: This is a specific example implementation (for reference only), showing how to handle time-series data. Requires `timestamps` parameter. |

---

### Configuration Examples

Here are some specific `yaml` configuration examples. Please note: **All results from single-modality feature extraction must be wrapped by a multi-modality combiner (like `concat`).**

#### Example 1: Using Single Raw Image Intensity

This is the simplest configuration. Even for single modality, it needs to be wrapped by a combiner (using `concat` here).

```yaml
FeatureConstruction:
  voxel_level:
    method: concat(raw(pre_contrast))
    params: {}
```

#### Example 2: Combining Multiple Raw Image Intensities

You can concatenate the raw intensity values from multiple modalities into a single feature vector. Here, we use the `concat` method to combine images from three different phases.

```yaml
FeatureConstruction:
  voxel_level:
    method: concat(raw(pre_contrast), raw(LAP), raw(PVP))
    params: {}
```

#### Example 3: Calculating Kinetic Features

Using the `kinetic` method requires providing a `timestamps` file. The method automatically processes multiple images wrapped in `raw()`.
**Note**: The `kinetic` function is provided as a **reference example** for hemodynamic feature processing, demonstrating the framework's capability to handle multi-modality time-series data. You can refer to its source code to implement your own specific logic.

```yaml
FeatureConstruction:
  voxel_level:
    method: kinetic(raw(pre_contrast), raw(LAP), raw(PVP), raw(delay_3min), timestamps)
    params:
      timestamps: './config/scan_times.xlsx' # Path to your timestamps file
```

#### Example 4: Extracting Voxel-level Radiomics Features

Using `voxel_radiomics` requires providing a PyRadiomics parameter file. Note: Even for a single feature extraction result, it must be wrapped by a multi-modality combiner (like `concat`).

```yaml
FeatureConstruction:
  voxel_level:
    method: concat(voxel_radiomics(pre_contrast, radiomics_params))
    params:
      radiomics_params: './config/radiomics_params.yaml' # Path to your radiomics parameter file
```

#### Example 5: Extracting Local Entropy Features

Using `local_entropy` with custom neighborhood size and bin count. Also requires wrapping by a combiner.

```yaml
FeatureConstruction:
  voxel_level:
    method: concat(local_entropy(pre_contrast, kernel_size, bins))
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

