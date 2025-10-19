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

```yaml
FeatureConstruction:
  voxel_level:
    # Use kinetic features
    method: kinetic(raw(pre_contrast), raw(LAP), raw(PVP), timestamps)
    params:
      timestamps: ./scan_times.xlsx
  
  # Individual-level preprocessing (optional)
  preprocessing_for_subject_level:
    methods:
      - method: winsorize
        winsor_limits: [0.05, 0.05]
      - method: minmax
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

