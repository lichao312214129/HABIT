# Image Preprocessing Module User Guide

## Overview

The Image Preprocessing module implements a series of medical image preprocessing steps, including resampling, registration, and N4 bias field correction, to provide standardized and optimized image data for subsequent habitat analysis.

## 🚀 Quick Start

### Using CLI (Recommended) ✨

```bash
# Use default configuration
habit preprocess

# Use specified configuration file
habit preprocess --config config/config_image_preprocessing.yaml

# Short form
habit preprocess -c config/config_image_preprocessing.yaml
```

### Using Traditional Scripts (Legacy Compatible)

```bash
# Use specified configuration file
python scripts/app_image_preprocessing.py --config ./config/config_image_preprocessing.yaml 

# Short form
python scripts/app_image_preprocessing.py -c ./config/config_image_preprocessing.yaml 

# If no configuration file is specified, the default will be used
python scripts/app_image_preprocessing.py
```

## 📋 Configuration File

**📖 Configuration File Links**:
- 📄 [Current Configuration](../config/config_image_preprocessing.yaml) - Actual configuration file in use
- 🇬🇧 Detailed English Configuration (Coming Soon) - Complete English comments and instructions
- 🇨🇳 详细中文配置 (Coming Soon) - Includes complete Chinese comments and instructions

> 💡 **Tip**: Detailed annotated configuration files are being prepared. Please refer to the configuration instructions below for now.

## Configuration File Format

`app_image_preprocessing.py` uses a YAML configuration file with the following main sections:

### Basic Configuration

```yaml
# Data paths
data_dir: <path_to_raw_data_directory>
out_dir: <path_to_output_directory>

# Preprocessing settings
Preprocessing:
  # Configuration for preprocessing steps
```

### Preprocessing Steps Configuration

```yaml
Preprocessing:
  # Resampling configuration
  resample:
    images: [<image1>, <image2>, ...]  # List of images to resample
    target_spacing: [x, y, z]  # Target voxel spacing
    img_mode: <interpolation_mode>  # Optional, defaults to linear
    padding_mode: <padding_mode>  # Optional, defaults to border
    align_corners: <true_or_false>  # Optional, defaults to false

  # Registration configuration
  registration:
    images: [<image1>, <image2>, ...]  # List of images to register
    fixed_image: <reference_image>  # Static reference image
    moving_images: [<image1>, <image2>, ...]  # List of images to be moved
    type_of_transform: <transform_type>  # e.g., Rigid, Affine, SyNRA
    use_mask: <true_or_false>  # Optional, defaults to false

  # N4 bias field correction configuration
  n4_correction:
    images: [<image1>, <image2>, ...]  # List of images to correct
    num_fitting_levels: <number_of_fitting_levels>  # Optional, defaults to 4
    num_iterations: [<iter1>, <iter2>, ...]  # Optional, defaults to 50 per level
    convergence_threshold: <threshold>  # Optional, defaults to 0.001
```

## Supported Preprocessing Steps

### 1. Resampling

Adjusts the voxel spacing of an image to a specified target resolution.

#### Parameters

| Parameter | Type | Description | Default |
|---|---|---|---|
| images | List[str] | List of images to resample | Required |
| target_spacing | List[float] | Target voxel spacing (in mm) | Required |
| img_mode | str | Image interpolation mode | "linear" |
| padding_mode | str | Padding mode for values outside boundaries | "border" |
| align_corners | bool | Whether to align corners | False |

#### Supported Interpolation Modes

- `nearest`: Nearest-neighbor interpolation
- `linear`: Linear interpolation
- `bilinear`: Bilinear interpolation (equivalent to linear)
- `bspline`: B-spline interpolation
- `bicubic`: Bicubic interpolation (equivalent to bspline)
- `gaussian`: Gaussian interpolation
- `lanczos`: Lanczos window-corrected sinc interpolation
- `hamming`: Hamming window-corrected sinc interpolation
- `cosine`: Cosine window-corrected sinc interpolation
- `welch`: Welch window-corrected sinc interpolation
- `blackman`: Blackman window-corrected sinc interpolation

### 2. Registration

Aligns images to a reference image using ANTs (Advanced Normalization Tools).

#### Parameters

| Parameter | Type | Description | Default |
|---|---|---|---|
| images | List[str] | List of images to process | Required |
| fixed_image | str | Keyword for the reference image | Required |
| moving_images | List[str] | List of images to be moved | Required |
| type_of_transform | str | Type of transformation | "Rigid" |
| metric | str | Similarity metric | "MI" |
| optimizer | str | Optimization method | "gradient_descent" |
| use_mask | bool | Whether to use a mask | False |

#### Supported Transformation Types

1.  **Rigid**: Translation and rotation only
2.  **Affine**: Translation, rotation, scaling, and shearing
3.  **SyN**: Symmetric Normalization (deformable registration)
4.  **SyNRA**: SyN + Rigid + Affine (most commonly used)
5.  **SyNOnly**: SyN without initial rigid/affine
6.  **TRSAA**: Translation + Rotation + Scaling + Affine
7.  **Elastic**: Elastic transformation
8.  **SyNCC**: SyN with cross-correlation metric
9.  **SyNabp**: SyN with mutual information metric
10. **SyNBold**: SyN optimized for BOLD images
11. **SyNBoldAff**: SyN + Affine for BOLD images
12. **SyNAggro**: SyN with aggressive optimization
13. **TVMSQ**: Time-Varying Diffeomorphic with Mean Squared Metric

### 3. N4 Bias Field Correction

Corrects intensity non-uniformity in medical images, often caused by magnetic field inhomogeneities in MRI scanners.

#### Parameters

| Parameter | Type | Description | Default |
|---|---|---|---|
| images | List[str] | List of images to correct | Required |
| num_fitting_levels | int | Number of fitting levels | 4 |
| num_iterations | List[int] | Iterations per level | [50] * num_fitting_levels |
| convergence_threshold | float | Convergence threshold | 0.001 |

### 4. Histogram Standardization

Matches the histogram of an image to the histogram of a reference image, making the intensity distributions across different images more consistent. This is very useful for standardizing images across different scanners or acquisition sequences.

#### Parameters

| Parameter | Type | Description | Default |
|---|---|---|---|
| images | List[str] | List of images to standardize | Required |
| reference_key | str | Key name of the reference image | Required |

#### Configuration Example

```yaml
histogram_standardization:
  images: [pre_contrast, LAP, delay_3min]  # Image sequences to process
  reference_key: PVP  # Key name of the reference image
```

## Complete Configuration Example

```yaml
# Data paths
data_dir: F:\work\research\radiomics_TLSs\data\raw_data
out_dir: F:\work\research\radiomics_TLSs\data\results

# Preprocessing settings
Preprocessing:
  # Resampling (optional)
  resample:
    images: [T2WI, ADC]
    target_spacing: [1.0, 1.0, 1.0]

  # Registration (optional)
  registration:
    images: [T2WI, ADC]
    fixed_image: T2WI
    moving_images: [ADC]
    type_of_transform: SyNRA  # Supports all ANTs registration methods
    use_mask: false

  # N4 bias field correction (optional)
  # n4_correction:
  #   images: [T2WI, ADC]
  #   num_fitting_levels: 2
    
  # Z-Score normalization (optional)
  # zscore_normalization:
  #   images: [T2WI, ADC]
  #   only_inmask: false
    
  # Histogram standardization (optional)
  # histogram_standardization:
  #   images: [T2WI]
  #   reference_key: ADC

# General settings
processes: 1  # Number of parallel processes
random_state: 42  # Random seed
```

## Execution Flow

1.  Load the configuration file.
2.  Initialize the `BatchProcessor`.
3.  Execute batch processing for all specified images.
4.  Generate a detailed processing log.

## Output Structure

```
out_dir/
├── processed_images/
│   ├── images/
│   │   └── subject_id/
│   │       ├── modality1/
│   │       │   └── modality1.nii.gz
│   │       └── modality2/
│   │           └── modality2.nii.gz
│   └── masks/
│       └── subject_id/
│           ├── modality1/
│           │   └── mask_modality1.nii.gz
│           └── modality2/
│               └── mask_modality2.nii.gz
└── logs/
    └── processing.log
```

## Logging

The script maintains a detailed log file `preprocessing_debug.log`, which includes:
- Processing progress
- Error messages
- Parameter settings
- Performance metrics

## Best Practice Recommendations

1.  **Resampling**:
    -   Choose an appropriate target spacing based on analysis needs.
    -   Use linear interpolation for images.
    -   Use nearest-neighbor interpolation for masks.

2.  **Registration**:
    -   Use SyN for deformable registration.
    -   Use the MI (Mutual Information) metric for multi-modal registration.
    -   Always use a mask if available.
    -   Consider using SyNRA for better initial alignment.

3.  **N4 Correction**:
    -   Use 4 fitting levels in most cases.
    -   Adjust the number of iterations based on image complexity.
    -   Use a mask if available for better correction.

## Common Issues and Solutions

1.  **Registration Failure**:
    -   Try different transformation types.
    -   Adjust the similarity metric.
    -   Enable `use_mask` if a mask is available.
    -   Check image orientations.

2.  **N4 Correction Issues**:
    -   Increase the number of iterations.
    -   Adjust the convergence threshold.
    -   Use a mask for better correction.

3.  **Memory Issues**:
    -   Reduce the number of parallel processes.
    -   Process images in smaller batches.
    -   Use a lower resolution for initial registration.

## Performance Considerations

1.  **Parallel Processing**:
    -   Use multiple processes to speed up processing.
    -   Adjust the number of processes based on available memory.
    -   Consider using fewer processes for memory-intensive operations.

2.  **Memory Usage**:
    -   Monitor memory usage during processing.
    -   Adjust batch size if necessary.
    -   Use appropriate image formats.

3.  **Disk Space**:
    -   Processed images are saved in NIfTI format.
    -   Consider compression for long-term storage.
    -   Periodically clean up temporary files.
