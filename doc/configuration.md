# HABIT Configuration Guide

This document provides a detailed explanation of the configuration parameters used in HABIT (Habitat Analysis Based on Imaging Traits).

## Configuration File Structure

HABIT uses YAML configuration files to define the analysis pipeline. The main configuration file is typically named `config_kmeans.yaml` or similar, and contains several sections:

1. **Data paths**: Define input and output directories for the analysis
2. **Preprocessing**: Image preprocessing settings for habitat segmentation
3. **FeatureConstruction**: Feature extraction for habitat segmentation
4. **HabitatsSegmentation**: Clustering settings for habitat segmentation
5. **General settings**: Global parameters for the analysis

## Data Paths

```yaml
data_dir: /path/to/input/data
out_dir: /path/to/output/directory
```

- `data_dir`: Root directory containing the input data. The data should be organized in a specific structure with subdirectories for images and masks.
- `out_dir`: Directory where all output files will be saved.

## Preprocessing Settings

```yaml
Preprocessing:
  N4BiasCorrection:
    images: [pre_contrast, LAP, PVP, delay_3min]
    n_iterations: [50, 50, 30, 20]  # Number of iterations at each scale level
    convergence_threshold: 0.001
    shrink_factor: 4
    spline_order: 3
    n_fitting_levels: 4
    bias_field_fwhm: 0.15

  resample:
    images: [pre_contrast, LAP, PVP, delay_3min]
    target_spacing: [1.0, 1.0, 1.0]  # Target voxel spacing in mm
    mode: bilinear  # Interpolation mode for images
    mask_mode: nearest  # Interpolation mode for masks
    align_corners: false
    anti_aliasing: true
    preserve_range: true

  registration:
    images: [pre_contrast, LAP, PVP, delay_3min]
    fixed_image: PVP
    moving_images: [pre_contrast, LAP, delay_3min]
    type_of_transform: Rigid  # Rigid or Affine
    use_mask: true  # Whether to use mask for registration
    optimizer_type: gradient_descent
    optimizer_params:
      learning_rate: 0.01
      number_of_iterations: 100
    metric_type: mutual_information
    metric_params:
      number_of_histogram_bins: 50
    interpolator_type: linear
```

### N4BiasCorrection

N4偏置场校正用于校正MRI图像中的强度不均匀性。

- `images`: 需要进行校正的图像列表
- `n_iterations`: 每个尺度级别的迭代次数列表，长度表示尺度级别数
- `convergence_threshold`: 收敛阈值，当两次迭代之间的变化小于此值时停止
- `shrink_factor`: 图像缩小因子，用于加速计算
- `spline_order`: B样条阶数 (默认: 3)
- `n_fitting_levels`: 拟合层级数 (默认: 4)
- `bias_field_fwhm`: 偏置场的半高全宽，控制平滑程度 (默认: 0.15)

### Resample

重采样用于统一不同图像的体素大小。

- `images`: 需要进行重采样的图像列表
- `target_spacing`: 目标体素间距 [x, y, z]，单位为毫米
- `mode`: 图像插值模式 (默认: "bilinear")
  - "nearest": 最近邻插值
  - "linear"/"bilinear": 线性插值
  - "bicubic": 三次插值
- `mask_mode`: mask插值模式 (默认: "nearest")
- `align_corners`: 是否对齐角点 (默认: false)
- `anti_aliasing`: 是否使用抗锯齿 (默认: true)
- `preserve_range`: 是否保持像素值范围 (默认: true)

### Registration

图像配准用于将不同时相的图像对齐到同一空间。

- `images`: 所有参与配准的图像列表
- `fixed_image`: 固定图像（参考图像）
- `moving_images`: 需要配准的移动图像列表
- `type_of_transform`: 变换类型
  - "Rigid": 刚体变换（平移和旋转）
  - "Affine": 仿射变换（平移、旋转、缩放和剪切）
- `use_mask`: 是否在配准过程中使用mask
- `optimizer_type`: 优化器类型 (默认: "gradient_descent")
  - "gradient_descent": 梯度下降
  - "conjugate_gradient": 共轭梯度
  - "lbfgs": L-BFGS优化器
- `optimizer_params`: 优化器参数
  - `learning_rate`: 学习率
  - `number_of_iterations`: 最大迭代次数
  - `convergence_minimum_value`: 收敛最小值
  - `convergence_window_size`: 收敛窗口大小
- `metric_type`: 相似度度量类型 (默认: "mutual_information")
  - "mutual_information": 互信息
  - "normalized_mutual_information": 归一化互信息
  - "mean_squares": 均方误差
  - "normalized_correlation": 归一化相关系数
- `metric_params`: 相似度度量参数
  - `number_of_histogram_bins`: 直方图bin数（用于互信息）
  - `sampling_percentage`: 采样百分比
- `interpolator_type`: 插值器类型 (默认: "linear")
  - "nearest": 最近邻插值
  - "linear": 线性插值
  - "bspline": B样条插值

## Feature Construction

```yaml
FeatureConstruction:
  voxel_level:
    method: kinetic(raw(pre_contrast), raw(LAP), raw(PVP), raw(delay_3min), timestamps)
    params:
      params_voxel_radiomics: ./config/params_voxel_radiomics.yaml
      kernelRadius: 2
      timestamps: /path/to/timestamps.xlsx

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
```

### Voxel Level

The `voxel_level` section defines how features are extracted at the voxel level.

- `method`: Feature extraction method expression. This can be a complex expression that combines multiple methods.
- `params`: Parameters for the feature extraction methods.

#### Method Expression Syntax

The method expression follows a functional syntax where methods can be nested. The general format is:

```
method_name(arg1, arg2, ..., paramName1, paramName2, ...)
```

Available methods include:

- `raw(image_name)`: Extract raw intensity values from the specified image
- `kinetic(image1, image2, ..., timestamps)`: Extract kinetic features from multiple images using timestamps
- `voxel_radiomics(image_name, params_file)`: Extract radiomics features from each voxel in the mask for the specified image
- `supervoxel_radiomics(image_name, params_file)`: Extract radiomics features for each supervoxel, the mean values derived from the voxel level step are used as the input of the supervoxel level step
- `concat(method1(image1), method2(image2), ...)`: Concatenate features from multiple methods
- `mean_voxel_features()`: Calculate mean values of voxel features for each supervoxel, the mean values derived from the voxel level step are used as the input of the supervoxel level step

#### Parameters

- `params_voxel_radiomics`: Path to the PyRadiomics parameter file for voxel-based radiomics feature extraction
- `kernelRadius`: Radius of the kernel used for voxel-based radiomics feature extraction (for methods that support it)
- `timestamps`: Path to an Excel file containing scan time information for each phase (used by kinetic method)

### Supervoxel Level

The `supervoxel_level` section defines how features are extracted at the supervoxel level.

- `supervoxel_file_keyword`: Pattern to match supervoxel files in output directory (automatically detected)
- `method`: Feature extraction method expression for supervoxel level
- `params`: Parameters for the supervoxel feature extraction methods

#### Example Methods

1. `mean_voxel_features()`: Calculate mean values of voxel features for each supervoxel, the mean values derived from the voxel level step are used as the input of the supervoxel level step
2. `concat(supervoxel_radiomics(pre_contrast, params_file), ...)`: Concatenate radiomics features from multiple images

### Feature Preprocessing

The `preprocessing` section defines how features are preprocessed before clustering.

- `methods`: List of preprocessing methods to apply in sequence
  - `method`: Preprocessing method name (e.g., "minmax", "winsorize", "standardize")
  - Method-specific parameters (e.g., `winsor_limits`, `global_normalize`)

## Habitat Segmentation

```yaml
HabitatsSegmention:
  # Supervoxel clustering settings
  supervoxel:
    algorithm: kmeans
    n_clusters: 50  # number of supervoxels to create
    random_state: 42
    max_iter: 300
    n_init: 10
  
  # Habitat clustering settings
  habitat:
    algorithm: kmeans
    max_clusters: 10  # maximum number of habitats to consider
    habitat_cluster_selection_method: inertia  # method to determine optimal number of clusters
    best_n_clusters: null  # set to null for automatic selection
    random_state: 42
    max_iter: 300
    n_init: 10
```

### Supervoxel Clustering

- `algorithm`: Clustering algorithm for supervoxel segmentation (currently supports "kmeans")
- `n_clusters`: Number of supervoxels to create (fixed value)
- `random_state`: Random seed for reproducibility
- `max_iter`: Maximum number of iterations for the clustering algorithm
- `n_init`: Number of times the algorithm will be run with different centroid seeds

### Habitat Clustering

- `algorithm`: Clustering algorithm for habitat segmentation (currently supports "kmeans")
- `max_clusters`: Maximum number of habitats to consider during automatic selection
- `habitat_cluster_selection_method`: Method to determine the optimal number of clusters
  - Options: 
    - "inertia": Use elbow method based on inertia
    - "silhouette": Use silhouette score
    - "calinski_harabasz": Use Calinski-Harabasz index
- `best_n_clusters`: Directly specify the number of clusters (if null, will be determined automatically using the specified method)
- `random_state`: Random seed for reproducibility
- `max_iter`: Maximum number of iterations for the clustering algorithm
- `n_init`: Number of times the algorithm will be run with different centroid seeds

## General Settings

```yaml
processes: 2  # Number of parallel processes
plot_curves: true  # Whether to generate and save plots
random_state: 42  # Random seed for reproducibility
debug: false  # Enable debug mode for detailed logging
```

- `processes`: Number of parallel processes to use for computation
- `plot_curves`: Whether to generate and save evaluation curves
- `random_state`: Global random seed for reproducibility
- `debug`: Enable debug mode for detailed logging
