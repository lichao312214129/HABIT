# HABIT API Reference

This document provides a reference for the main classes and functions in the HABIT (Habitat Analysis Based on Imaging Traits) package.

## Core Module

### HabitatAnalysis

```python
from habit.core.habitat_analysis import HabitatAnalysis
```

The main class for performing habitat analysis.

#### Constructor

```python
HabitatAnalysis(
    root_folder,
    out_folder=None,
    feature_config=None,
    supervoxel_clustering_method="kmeans",
    habitat_clustering_method="kmeans",
    n_clusters_supervoxel=50,
    n_clusters_habitats_max=10,
    n_clusters_habitats_min=2,
    habitat_cluster_selection_method=None,
    best_n_clusters=None,
    n_processes=1,
    random_state=42,
    verbose=True,
    images_dir="images",
    masks_dir="masks",
    plot_curves=True,
    progress_callback=None,
    save_intermediate_results=False
)
```

**Parameters:**

- `root_folder` (str): Root directory of data
- `out_folder` (str, optional): Output directory
- `feature_config` (dict, optional): Feature configuration dictionary
- `supervoxel_clustering_method` (str, optional): Supervoxel clustering method, default is "kmeans"
- `habitat_clustering_method` (str, optional): Habitat clustering method, default is "kmeans"
- `n_clusters_supervoxel` (int, optional): Number of supervoxel clusters, default is 50
- `n_clusters_habitats_max` (int, optional): Maximum number of habitat clusters, default is 10
- `n_clusters_habitats_min` (int, optional): Minimum number of habitat clusters, default is 2
- `habitat_cluster_selection_method` (str or list, optional): Method for selecting number of habitat clusters
- `best_n_clusters` (int, optional): Directly specify the best number of clusters
- `n_processes` (int, optional): Number of parallel processes, default is 1
- `random_state` (int, optional): Random seed, default is 42
- `verbose` (bool, optional): Whether to output detailed information, default is True
- `images_dir` (str, optional): Image directory name, default is "images"
- `masks_dir` (str, optional): Mask directory name, default is "masks"
- `plot_curves` (bool, optional): Whether to plot evaluation curves, default is True
- `progress_callback` (callable, optional): Progress callback function
- `save_intermediate_results` (bool, optional): Whether to save intermediate results, default is False

#### Methods

##### run

```python
run(subjects=None, save_results_csv=True)
```

Run the habitat clustering pipeline.

**Parameters:**

- `subjects` (list, optional): List of subjects to process. If None, all subjects will be processed.
- `save_results_csv` (bool, optional): Whether to save results as CSV files. Defaults to True.

**Returns:**

- `pd.DataFrame`: Habitat clustering results

##### extract_voxel_features

```python
extract_voxel_features(subject)
```

Extract features for a single subject.

**Parameters:**

- `subject` (str): Subject ID

**Returns:**

- `tuple`: Tuple containing subject ID, feature dataframe, original image dataframe, and mask information

##### extract_supervoxel_features

```python
extract_supervoxel_features()
```

Extract supervoxel-level features from supervoxel maps and original images.

**Returns:**

- `np.ndarray`: Supervoxel features for clustering

## Feature Extraction

### BaseFeatureExtractor

```python
from habit.core.habitat_analysis.features.base_feature_extractor import BaseFeatureExtractor
```

Base class for feature extractors.

#### Methods

##### extract_features

```python
extract_features(image_data, **kwargs)
```

Extract features from image data.

**Parameters:**

- `image_data`: Image data (format depends on the specific extractor)
- `**kwargs`: Additional parameters for feature extraction

**Returns:**

- `pd.DataFrame`: Extracted features

##### get_feature_names

```python
get_feature_names()
```

Get feature names.

**Returns:**

- `list`: List of feature names

### Feature Extractors

HABIT includes several built-in feature extractors:

- `RawFeatureExtractor`: Extract raw intensity values from images
- `KineticFeatureExtractor`: Extract kinetic features from dynamic images
- `VoxelRadiomicsExtractor`: Extract radiomics features at the voxel level
- `SupervoxelRadiomicsExtractor`: Extract radiomics features for each supervoxel
- `ConcatImageFeatureExtractor`: Concatenate features from multiple images
- `MeanVoxelFeaturesExtractor`: Calculate mean values of voxel features for each supervoxel

## Clustering

### BaseClusteringAlgorithm

```python
from habit.core.habitat_analysis.clustering.base_clustering import BaseClusteringAlgorithm
```

Base class for clustering algorithms.

#### Methods

##### fit

```python
fit(X)
```

Fit the clustering algorithm to the data.

**Parameters:**

- `X` (np.ndarray): Feature matrix

**Returns:**

- `self`: The fitted estimator

##### predict

```python
predict(X)
```

Predict cluster labels for the data.

**Parameters:**

- `X` (np.ndarray): Feature matrix

**Returns:**

- `np.ndarray`: Cluster labels

##### find_optimal_clusters

```python
find_optimal_clusters(X, min_clusters, max_clusters, methods=None, show_progress=False)
```

Find the optimal number of clusters.

**Parameters:**

- `X` (np.ndarray): Feature matrix
- `min_clusters` (int): Minimum number of clusters to consider
- `max_clusters` (int): Maximum number of clusters to consider
- `methods` (list, optional): List of evaluation methods
- `show_progress` (bool, optional): Whether to show progress

**Returns:**

- `tuple`: Tuple containing optimal number of clusters and evaluation scores

### Clustering Algorithms

HABIT includes several built-in clustering algorithms:

- `KMeansClustering`: K-means clustering
- `AgglomerativeClustering`: Hierarchical agglomerative clustering
- `SpectralClustering`: Spectral clustering
- `GMMClustering`: Gaussian Mixture Model clustering

## Utilities

### IO Utilities

```python
from habit.utils.io_utils import load_config, save_results, get_image_and_mask_paths
```

#### load_config

```python
load_config(config_file)
```

Load configuration from a config file (JSON or YAML format).

**Parameters:**

- `config_file` (str): Path to the configuration file

**Returns:**

- `dict`: Configuration dictionary

#### save_results

```python
save_results(results, output_dir, prefix="")
```

Save analysis results to disk.

**Parameters:**

- `results` (dict): Dictionary containing analysis results
- `output_dir` (str): Directory to save results
- `prefix` (str, optional): Prefix for output files

#### get_image_and_mask_paths

```python
get_image_and_mask_paths(root_dir, keyword_of_raw_folder="images", keyword_of_mask_folder="masks")
```

Get paths of images and masks.

**Parameters:**

- `root_dir` (str): Root directory containing images and masks
- `keyword_of_raw_folder` (str, optional): Keyword for raw image folder
- `keyword_of_mask_folder` (str, optional): Keyword for mask folder

**Returns:**

- `tuple`: Tuple containing dictionaries mapping subject IDs to image and mask paths

### Visualization Utilities

```python
from habit.utils.visualization import plot_feature_importance, plot_confusion_matrix, plot_roc_curve, plot_habitat_map
```

#### plot_feature_importance

```python
plot_feature_importance(feature_importances, feature_names, output_dir, top_n=20, figsize=(12, 8))
```

Plot feature importance.

**Parameters:**

- `feature_importances` (dict): Dictionary mapping feature selector names to importance values
- `feature_names` (list): List of feature names
- `output_dir` (str): Directory to save plots
- `top_n` (int, optional): Number of top features to show
- `figsize` (tuple, optional): Figure size

#### plot_confusion_matrix

```python
plot_confusion_matrix(confusion_matrix, class_names, output_path, normalize=True, figsize=(8, 6), cmap="Blues")
```

Plot confusion matrix.

**Parameters:**

- `confusion_matrix` (np.ndarray): Confusion matrix array
- `class_names` (list): List of class names
- `output_path` (str): Path to save the plot
- `normalize` (bool, optional): Whether to normalize the confusion matrix
- `figsize` (tuple, optional): Figure size
- `cmap` (str, optional): Colormap

#### plot_roc_curve

```python
plot_roc_curve(fpr, tpr, auc, output_path, figsize=(10, 8))
```

Plot ROC curve.

**Parameters:**

- `fpr` (dict): Dictionary mapping model names to false positive rates
- `tpr` (dict): Dictionary mapping model names to true positive rates
- `auc` (dict): Dictionary mapping model names to AUC values
- `output_path` (str): Path to save the plot
- `figsize` (tuple, optional): Figure size

#### plot_habitat_map

```python
plot_habitat_map(habitat_map, original_image=None, output_path=None, figsize=(15, 10), cmap="viridis", alpha=0.7)
```

Plot habitat map overlaid on original image.

**Parameters:**

- `habitat_map` (np.ndarray): 3D habitat map array
- `original_image` (np.ndarray, optional): 3D original image array
- `output_path` (str, optional): Path to save the plot
- `figsize` (tuple, optional): Figure size
- `cmap` (str, optional): Colormap for habitat map
- `alpha` (float, optional): Alpha value for overlay

### Progress Utilities

```python
from habit.core.habitat_analysis.utils.progress_utils import CustomTqdm, tqdm_with_message
```

#### CustomTqdm

```python
CustomTqdm(total=None, desc="Progress")
```

Custom progress bar class.

**Parameters:**

- `total` (int, optional): Total number of iterations
- `desc` (str, optional): Progress bar description

#### tqdm_with_message

```python
tqdm_with_message(iterable, desc="Progress", total=None, unit="it")
```

Progress bar wrapper with message.

**Parameters:**

- `iterable`: Iterable object
- `desc` (str, optional): Progress bar description
- `total` (int, optional): Total number of iterations
- `unit` (str, optional): Unit label

**Returns:**

- `iterator`: Iterator with progress bar
