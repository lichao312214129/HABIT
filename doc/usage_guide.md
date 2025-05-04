# HABIT Usage Guide

This document provides a guide on how to use HABIT (Habitat Analysis Based on Imaging Traits) for tumor habitat analysis.

## Installation

### Prerequisites

- Python 3.8+
- Required Python packages:
  - numpy
  - pandas
  - scikit-learn
  - SimpleITK
  - matplotlib
  - seaborn
  - PyRadiomics
  - PyYAML

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/HABIT.git
   cd HABIT
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

HABIT expects data to be organized in a specific directory structure:

```
data_root/
├── images/
│   ├── subject1/
│   │   ├── pre_contrast/
│   │   │   └── image.nii.gz
│   │   ├── LAP/
│   │   │   └── image.nii.gz
│   │   ├── PVP/
│   │   │   └── image.nii.gz
│   │   └── delay_3min/
│   │       └── image.nii.gz
│   ├── subject2/
│   │   └── ...
│   └── ...
└── masks/
    ├── subject1/
    │   ├── pre_contrast/
    │   │   └── mask.nii.gz
    │   ├── LAP/
    │   │   └── mask.nii.gz
    │   ├── PVP/
    │   │   └── mask.nii.gz
    │   └── delay_3min/
    │       └── mask.nii.gz
    ├── subject2/
    │   └── ...
    └── ...
```

- Each subject has a directory in both `images/` and `masks/`
- Each subject directory contains subdirectories for different image types/phases
- Each image type directory contains a single image file
- The mask directory structure mirrors the image directory structure

## Basic Usage

### Configuration

1. Create a configuration file (see [Configuration Guide](configuration.md) for details):
   ```yaml
   # config.yaml
   data_dir: /path/to/data_root
   out_dir: /path/to/output_directory

   # Other configuration parameters...
   ```

2. Run the habitat analysis using Python:
   ```python
   # Example script showing how to use HABIT in your own Python code
   from habit.core.habitat_analysis import HabitatAnalysis
   from habit.utils.io_utils import load_config

   # Load configuration
   config = load_config('config/config_kmeans.yaml')

   # Create habitat analysis instance
   habitat_analysis = HabitatAnalysis(
       root_folder=config['data_dir'],
       out_folder=config['out_dir'],
       feature_config=config['FeatureConstruction'],
       supervoxel_clustering_method=config['HabitatsSegmention']['supervoxel']['algorithm'],
       habitat_clustering_method=config['HabitatsSegmention']['habitat']['algorithm'],
       n_clusters_supervoxel=config['HabitatsSegmention']['supervoxel']['n_clusters'],
       n_clusters_habitats_max=config['HabitatsSegmention']['habitat']['max_clusters'],
       habitat_cluster_selection_method=config['HabitatsSegmention']['habitat']['habitat_cluster_selection_method'],
       n_processes=config['processes'],
       random_state=config['random_state'],
       plot_curves=config['plot_curves']
   )

   # Run the analysis
   results = habitat_analysis.run()

   # Access the results
   print(f"Found {len(results)} supervoxels across all subjects")
   print(f"Feature names: {habitat_analysis.voxel_feature_extractor.get_feature_names()}")

   # Save results to CSV (if not already saved during run)
   results.to_csv('habitat_results.csv')
   ```

### Command Line Interface

HABIT provides command-line scripts in the `scripts` directory for running analyses:

```bash
# Run habitat analysis
python scripts/run_habitat_analysis.py --config config/config_kmeans.yaml

# Extract features
python scripts/extract_features.py --config config/feature_extraction_config.yaml

# Run machine learning analysis
python scripts/run_machine_learning.py --config config/ml_config.yaml
```

## Output Files

After running the analysis, HABIT generates the following output files in the specified output directory:

- `habitats.csv`: CSV file containing habitat clustering results
- `config.json`: JSON file containing the configuration used for the analysis
- `*_supervoxel.nrrd`: NRRD files containing supervoxel maps for each subject
- `*_habitat.nrrd`: NRRD files containing habitat maps for each subject
- `habitat_clustering_scores.png`: Plot of clustering evaluation scores (if `plot_curves` is enabled)

## Advanced Usage

### Custom Feature Extractors

You can create custom feature extractors by extending the `BaseFeatureExtractor` class and placing them in the `habit/core/habitat_analysis/features` directory:

```python
# Save this as habit/core/habitat_analysis/features/my_feature_extractor.py
from habit.core.habitat_analysis.features.base_feature_extractor import BaseFeatureExtractor, register_feature_extractor
import pandas as pd
import numpy as np
import SimpleITK as sitk

@register_feature_extractor('my_custom_extractor')
class MyCustomExtractor(BaseFeatureExtractor):
    """
    Custom feature extractor that calculates histogram-based features

    This extractor calculates histogram statistics from image intensity values
    within the ROI mask.
    """

    def __init__(self, n_bins: int = 10, **kwargs):
        """
        Initialize the custom feature extractor

        Parameters
        ----------
        n_bins : int, default=10
            Number of histogram bins
        **kwargs : dict
            Additional parameters passed to parent class
        """
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.feature_names = [
            'histogram_mean',
            'histogram_std',
            'histogram_skewness',
            'histogram_kurtosis'
        ]

    def extract_features(self, image_data: Union[str, sitk.Image],
                         mask_data: Union[str, sitk.Image],
                         **kwargs) -> pd.DataFrame:
        """
        Extract histogram-based features from the image

        Parameters
        ----------
        image_data : str or sitk.Image
            Path to image file or SimpleITK image object
        mask_data : str or sitk.Image
            Path to mask file or SimpleITK image object
        **kwargs : dict
            Additional parameters

        Returns
        -------
        pd.DataFrame
            DataFrame containing extracted features
        """
        # Load image and mask if paths are provided
        if isinstance(image_data, str):
            image = sitk.ReadImage(image_data)
        else:
            image = image_data

        if isinstance(mask_data, str):
            mask = sitk.ReadImage(mask_data)
        else:
            mask = mask_data

        # Get arrays
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)

        # Extract values within mask
        values = image_array[mask_array > 0]

        # Calculate histogram
        hist, _ = np.histogram(values, bins=self.n_bins)
        hist = hist / np.sum(hist)  # Normalize

        # Calculate statistics
        mean = np.mean(hist)
        std = np.std(hist)
        skewness = np.mean(((hist - mean) / std) ** 3) if std > 0 else 0
        kurtosis = np.mean(((hist - mean) / std) ** 4) if std > 0 else 0

        # Create DataFrame with features
        features_df = pd.DataFrame({
            'histogram_mean': [mean],
            'histogram_std': [std],
            'histogram_skewness': [skewness],
            'histogram_kurtosis': [kurtosis]
        })

        return features_df
```

Then you can use your custom extractor in the configuration:

```yaml
FeatureConstruction:
  voxel_level:
    method: my_custom_extractor(pre_contrast, n_bins)
    params:
      n_bins: 20
```

### Customizing Clustering

You can customize the clustering algorithms by extending the `BaseClustering` class and placing them in the `habit/core/habitat_analysis/clustering` directory:

```python
# Save this as habit/core/habitat_analysis/clustering/my_clustering.py
from habit.core.habitat_analysis.clustering.base_clustering import BaseClustering, register_clustering_algorithm
import numpy as np
from sklearn.cluster import DBSCAN

@register_clustering_algorithm('my_dbscan')
class MyDBSCANClustering(BaseClustering):
    """
    Custom DBSCAN clustering implementation with additional functionality

    This clustering algorithm extends the standard DBSCAN with automatic
    epsilon selection based on k-distance graph.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5, **kwargs):
        """
        Initialize the custom DBSCAN clustering

        Parameters
        ----------
        eps : float, default=0.5
            The maximum distance between two samples for them to be considered neighbors
        min_samples : int, default=5
            The number of samples in a neighborhood for a point to be considered a core point
        **kwargs : dict
            Additional parameters passed to parent class
        """
        super().__init__(**kwargs)
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.labels_ = None

    def fit(self, X):
        """
        Fit the clustering model to the data

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : object
            Fitted estimator
        """
        # Create and fit DBSCAN model
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.model.fit(X)
        self.labels_ = self.model.labels_

        # Handle noise points (label -1) by assigning them to the nearest cluster
        if -1 in self.labels_:
            self._reassign_noise_points(X)

        return self

    def _reassign_noise_points(self, X):
        """Reassign noise points to the nearest cluster"""
        noise_indices = np.where(self.labels_ == -1)[0]
        cluster_indices = np.where(self.labels_ >= 0)[0]

        if len(cluster_indices) == 0:
            # All points are noise, create a single cluster
            self.labels_[:] = 0
            return

        # For each noise point, find the nearest non-noise point
        for idx in noise_indices:
            # Calculate distances to all non-noise points
            distances = np.sqrt(np.sum((X[idx] - X[cluster_indices]) ** 2, axis=1))
            # Find the nearest non-noise point
            nearest_idx = cluster_indices[np.argmin(distances)]
            # Assign the noise point to the same cluster as the nearest non-noise point
            self.labels_[idx] = self.labels_[nearest_idx]

    def predict(self, X):
        """
        Predict cluster labels for new data

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet")

        # For new data, assign each point to the nearest cluster centroid
        centroids = self._calculate_centroids(X)

        # Calculate distance to each centroid
        labels = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            distances = np.sqrt(np.sum((X[i] - centroids) ** 2, axis=1))
            labels[i] = np.argmin(distances)

        return labels

    def _calculate_centroids(self, X):
        """Calculate cluster centroids"""
        unique_labels = np.unique(self.labels_)
        centroids = np.zeros((len(unique_labels), X.shape[1]))

        for i, label in enumerate(unique_labels):
            mask = self.labels_ == label
            centroids[i] = np.mean(X[mask], axis=0)

        return centroids
```

Then you can use your custom clustering algorithm in the configuration:

```yaml
HabitatsSegmention:
  habitat:
    algorithm: my_dbscan
    eps: 0.3
    min_samples: 10
```

## Troubleshooting

### Common Issues

1. **Missing image or mask files**: Ensure that your data follows the expected directory structure.
2. **Memory errors**: Try reducing the `n_clusters_supervoxel` parameter or processing fewer subjects at a time.
3. **Feature extraction errors**: Check that the image types specified in the configuration match the available image directories.

### Logging

HABIT provides detailed logging to help diagnose issues:

```python
from habit.utils.io_utils import setup_logging

# Setup logging with detailed debug information
setup_logging(out_dir='logs', debug=True)

# You can also configure logging in your scripts
import logging
logging.info("Starting habitat analysis")
logging.debug("Detailed configuration information: %s", config)
```

## Examples

Here are some examples of how to use HABIT for different tasks:

### Basic Habitat Analysis

```python
# Example script for basic habitat analysis
from habit.core.habitat_analysis import HabitatAnalysis
from habit.utils.io_utils import load_config

# Load configuration
config = load_config('config/config_kmeans.yaml')

# Run analysis
analysis = HabitatAnalysis(
    root_folder=config['data_dir'],
    out_folder=config['out_dir'],
    feature_config=config['FeatureConstruction'],
    supervoxel_clustering_method='kmeans',
    habitat_clustering_method='kmeans',
    n_clusters_supervoxel=50,
    n_clusters_habitats_max=10,
    n_processes=4
)
results = analysis.run()
```

### Feature Extraction Only

```python
# Example script for feature extraction without clustering
from habit.core.habitat_analysis.features.feature_extractor_factory import create_feature_extractor
import SimpleITK as sitk
import pandas as pd

# Create a raw feature extractor
raw_extractor = create_feature_extractor('raw')

# Create a radiomics feature extractor
radiomics_extractor = create_feature_extractor(
    'voxel_radiomics',
    params_file='config/params_voxel_radiomics.yaml'
)

# Load image and mask
image = sitk.ReadImage('path/to/image.nii.gz')
mask = sitk.ReadImage('path/to/mask.nii.gz')

# Extract features
raw_features = raw_extractor.extract_features(image, mask)
radiomics_features = radiomics_extractor.extract_features(image, mask)

# Combine features
all_features = pd.concat([raw_features, radiomics_features], axis=1)
all_features.to_csv('extracted_features.csv')
```

### Machine Learning with Habitat Features

```python
# Example script for machine learning with habitat features
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load habitat features
features = pd.read_csv('habitats.csv')

# Prepare data
X = features.drop(['Subject', 'Supervoxel', 'Habitats', 'Label'], axis=1, errors='ignore')
y = features['Label']  # Assuming you have a Label column

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance.head(10))
```

See the `scripts` directory for more example scripts demonstrating different use cases.
