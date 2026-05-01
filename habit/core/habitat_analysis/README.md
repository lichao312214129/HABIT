# Habitat Analysis Module

This module implements tumor habitat clustering analysis using a two-step (or one-step) approach for medical image analysis.

## Overview

Habitat analysis divides tumors into distinct sub-regions (habitats) based on imaging features, revealing intra-tumor heterogeneity that may have prognostic or predictive value.

## Architecture

```
habitat_analysis/
├── habitat_analysis.py           # HabitatAnalysis (deep module: build / fit / predict / run)
├── config_schemas.py             # Pydantic configuration models (HabitatAnalysisConfig, ...)
├── managers/                     # FeatureManager / ClusteringManager / ResultManager
├── pipelines/
│   ├── base_pipeline.py          # HabitatPipeline + BasePipelineStep
│   └── steps/                    # Concrete pipeline steps
├── algorithms/                   # Clustering algorithms (KMeans, GMM, ...)
├── extractors/                   # Voxel / supervoxel feature extractors
└── analyzers/                    # HabitatMapAnalyzer (post-clustering features)
```

> V1 note: the legacy `strategies/` subpackage and `pipelines/pipeline_builder.py`
> have been removed. Mode dispatch (`two_step` / `one_step` / `direct_pooling`)
> now happens through the `_PIPELINE_RECIPES` dictionary inside
> `habitat_analysis.py`. See `ARCHITECTURE.md` for details.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HABITAT ANALYSIS PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Input     │    │   Feature   │    │  Supervoxel │    │   Habitat   │
│   Images    │───▶│  Extraction │───▶│  Clustering │───▶│  Clustering │
│   + Masks   │    │  (Voxel)    │    │ (Individual)│    │(Population) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                              │                  │
                                              │                  │
                                              ▼                  ▼
                                       ┌─────────────┐    ┌─────────────┐
                                       │  Supervoxel │    │   Habitat   │
                                       │    Maps     │    │    Maps     │
                                       └─────────────┘    └─────────────┘
```

## Clustering Strategies

### Two-Step Clustering (Default)

1. **Step 1: Individual-Level Clustering (Voxel → Supervoxel)**
   - For each tumor, cluster voxels into supervoxels
   - Reduces dimensionality while preserving spatial information
   - Default: 50 supervoxels per tumor

2. **Step 2: Population-Level Clustering (Supervoxel → Habitat)**
   - Pool supervoxels from all patients
   - Cluster to find common habitat patterns
   - Optimal cluster number determined automatically

```
Patient 1 Tumor    Patient 2 Tumor    Patient N Tumor
      │                  │                  │
      ▼                  ▼                  ▼
┌──────────┐       ┌──────────┐       ┌──────────┐
│ 50 super │       │ 50 super │       │ 50 super │
│  voxels  │       │  voxels  │       │  voxels  │
└────┬─────┘       └────┬─────┘       └────┬─────┘
     │                  │                  │
     └──────────────────┼──────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  Pool All       │
              │  Supervoxels    │
              │  (N × 50)       │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Population     │
              │  Clustering     │
              │  → K Habitats   │
              └─────────────────┘
```

### One-Step Clustering

- Individual-level clustering only
- Each tumor gets its own habitat assignment
- Supervoxels = Habitats (no population-level step)
- Useful when patient-specific patterns are desired

## Configuration

### Using Configuration Classes (Recommended)

```python
from habit.core.habitat_analysis import (
    HabitatAnalysis,
    HabitatConfig,
    ClusteringConfig,
    IOConfig,
    RuntimeConfig
)

config = HabitatConfig(
    clustering=ClusteringConfig(
        strategy='two_step',           # 'one_step' or 'two_step'
        supervoxel_method='kmeans',    # Clustering algorithm
        habitat_method='kmeans',
        n_clusters_supervoxel=50,      # Supervoxels per tumor
        n_clusters_habitats_min=2,     # Min habitats to test
        n_clusters_habitats_max=10,    # Max habitats to test
        random_state=42
    ),
    io=IOConfig(
        root_folder='/path/to/data',
        out_folder='/path/to/output',
        images_dir='images',
        masks_dir='masks'
    ),
    runtime=RuntimeConfig(
        n_processes=4,                 # Parallel processing
        verbose=True,
        plot_curves=True
    ),
    feature_config={
        'voxel_level': {
            'method': 'concat(raw(img1), raw(img2))',
            'image_names': ['T1', 'T2', 'FLAIR']
        },
        'supervoxel_level': {
            'method': 'mean_voxel_features()',
            'supervoxel_file_keyword': '*_supervoxel.nrrd'
        }
    }
)

analyzer = HabitatAnalysis(config=config)
results = analyzer.run()
```

## Feature Configuration

### Voxel-Level Features

Define how to extract features from each voxel:

```yaml
voxel_level:
  method: "concat(raw(pre_contrast), raw(LAP), raw(PVP))"
  image_names:
    - pre_contrast
    - LAP
    - PVP
  params:
    normalize: true
```

### Supervoxel-Level Features

Define how to aggregate voxel features into supervoxel features:

```yaml
supervoxel_level:
  method: "mean_voxel_features()"
  supervoxel_file_keyword: "*_supervoxel.nrrd"
```

## Outputs

| File | Description |
|------|-------------|
| `habitats.csv` | Results with habitat labels per supervoxel |
| `{subject}_supervoxel.nrrd` | Supervoxel label maps |
| `{subject}_habitat.nrrd` | Final habitat label maps |
| `config.yaml` / `config.json` | Saved configuration |
| `supervoxel2habitat_clustering_strategy.pkl` | Trained clustering model |
| `mean_values_of_all_supervoxels_features.csv` | Feature statistics |
| `visualizations/` | Clustering visualizations |

## Training vs Inference (sklearn-style)

### Training (fit/fit_transform)
1. Extract features and cluster supervoxels
2. Find optimal number of habitats
3. Save clustering model and statistics
4. Generate habitat maps

### Inference (load + transform)
1. Load a saved pipeline
2. Apply the fitted model to new data
3. Generate habitat maps

## Supported Clustering Algorithms

| Algorithm | Key | Description |
|-----------|-----|-------------|
| K-Means | `kmeans` | Fast, spherical clusters |
| GMM | `gmm` | Probabilistic, elliptical clusters |
| Hierarchical | `hierarchical` | Dendrogram-based |
| Spectral | `spectral` | Graph-based, non-convex |
| DBSCAN | `dbscan` | Density-based, auto clusters |
| Mean Shift | `mean_shift` | Density-based, auto clusters |
| Affinity Propagation | `affinity_propagation` | Exemplar-based |

## Validation Methods

Methods to determine optimal number of clusters:

| Method | Description | Selection Logic |
|--------|-------------|-----------------|
| `silhouette` | Silhouette coefficient (-1 to 1, higher is better) | **Max Principle**: Selects cluster number with highest score |
| `calinski_harabasz` | Variance ratio (higher is better) | **Max Principle**: Selects cluster number with highest score |
| `davies_bouldin` | Cluster separation (lower is better) | **Min Principle**: Selects cluster number with lowest score |
| `inertia` | Sum of squared distances to the nearest cluster center | **Kneedle Elbow**: Evaluates the inertia curve and selects its knee |
| `kneedle` | Automatic knee detection on the inertia curve | **Kneedle Elbow**: Same underlying inertia curve as `inertia`, explicit selector name |
| `bic` / `aic` | Information criteria for GMM (lower is better) | **Min Principle**: Selects cluster number with lowest score |
| `gap_statistic` | Gap to random reference (higher is better) | **Max Principle**: Selects cluster number with highest score |

### Combined Methods: Voting System

When multiple methods are configured as a YAML list (for example, `['silhouette', 'calinski_harabasz', 'davies_bouldin']`), the system uses a **voting system**:

1. Each method independently selects its optimal cluster number
2. Votes are counted for each cluster number
3. The cluster number with the most votes is selected
4. In case of ties, the smallest cluster number is chosen (more conservative)

**Example**: If `silhouette` and `calinski_harabasz` both select 5, and `davies_bouldin` selects 4, then 5 wins with 2 votes.

Use list syntax for combined methods. Method names are not concatenated with underscores, because names such as `calinski_harabasz` and `davies_bouldin` contain underscores themselves.

## Module Components

### `config.py` - Configuration Classes

- `HabitatConfig`: Master configuration container
- `ClusteringConfig`: Clustering parameters
- `IOConfig`: Input/output paths
- `RuntimeConfig`: Runtime behavior
- `OneStepConfig`: One-step mode settings
- `ResultColumns`: Column name constants

### `pipelines/` - Pipeline Core

- `HabitatPipeline`: sklearn-style pipeline (`fit` / `transform` / `save` / `load`)
- `BasePipelineStep`: pipeline-step interface

In V1, mode dispatch is handled by the `_PIPELINE_RECIPES` dictionary inside
`habitat_analysis.py`. There is no separate `pipeline_builder` factory.

### `habitat_analysis.py` - Main Class

Public entry points on `HabitatAnalysis`:

- `fit(subjects=None, save_results_csv=None)`: train + persist
- `predict(pipeline_path, subjects=None, save_results_csv=None)`: load + transform
- `run(subjects=None, save_results_csv=None, load_from=None)`: dispatcher

## Example YAML Configuration

```yaml
# config_habitat.yaml
root_folder: ./demo_data/preprocessed
out_dir: ./demo_data/habitat_output

clustering:
  strategy: two_step
  supervoxel_method: kmeans
  habitat_method: kmeans
  n_clusters_supervoxel: 50
  n_clusters_habitats_min: 2
  n_clusters_habitats_max: 10

runtime:
  n_processes: 4
  verbose: true
  plot_curves: true

feature_config:
  voxel_level:
    method: "concat(raw(pre_contrast), raw(LAP), raw(PVP))"
    image_names:
      - pre_contrast
      - LAP
      - PVP
  supervoxel_level:
    method: "mean_voxel_features()"
```

## References

- Gatenby RA, et al. "Quantitative imaging in cancer evolution and ecology." Radiology, 2013.
- Wu J, et al. "Intratumoral Spatial Heterogeneity at Perfusion MR Imaging." Radiology, 2017.
