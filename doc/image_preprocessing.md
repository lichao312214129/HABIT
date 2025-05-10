# Image Preprocessing Documentation

## Overview

The HABIT image preprocessing pipeline provides a comprehensive set of tools for medical image preprocessing. This document provides detailed information about each preprocessing step and its parameters.

## Preprocessing Steps

### 1. Resampling

The resampling step adjusts the voxel spacing of images to a target resolution. This is often necessary when working with images from different scanners or when standardizing the resolution for analysis.

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| target_spacing | List[float] | Target voxel spacing in mm | Required |
| img_mode | str | Interpolation mode for images | "linear" |
| padding_mode | str | Padding mode for out-of-bound values | "border" |
| align_corners | bool | Whether to align corners | False |

#### Interpolation Modes

- `nearest`: Nearest neighbor interpolation
- `linear`: Linear interpolation
- `bilinear`: Bilinear interpolation (same as linear)
- `bspline`: B-spline interpolation
- `bicubic`: Bicubic interpolation (same as bspline)
- `gaussian`: Gaussian interpolation
- `lanczos`: Lanczos windowed sinc interpolation
- `hamming`: Hamming windowed sinc interpolation
- `cosine`: Cosine windowed sinc interpolation
- `welch`: Welch windowed sinc interpolation
- `blackman`: Blackman windowed sinc interpolation

#### Example Configuration

```yaml
resample:
  images: [t1, t2, flair]
  target_spacing: [1.0, 1.0, 1.0]
  img_mode: linear
  padding_mode: border
  align_corners: false
```

### 2. Registration

The registration step aligns images to a reference image using ANTs (Advanced Normalization Tools). This is crucial for multi-modal analysis and longitudinal studies.

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| fixed_image | str | Key of the reference image | Required |
| type_of_transform | str | Type of transformation | "SyN" |
| metric | str | Similarity metric | "MI" |
| optimizer | str | Optimization method | "gradient_descent" |
| use_mask | bool | Whether to use masks | False |

#### Transformation Types

1. **Rigid**: Translation and rotation only
2. **Affine**: Translation, rotation, scaling, and shearing
3. **SyN**: Symmetric normalization (deformable)
4. **SyNRA**: SyN + Rigid + Affine
5. **SyNOnly**: SyN without initial rigid/affine
6. **TRSAA**: Translation + Rotation + Scaling + Affine
7. **Elastic**: Elastic transformation
8. **SyNCC**: SyN with cross-correlation metric
9. **SyNabp**: SyN with mutual information metric
10. **SyNBold**: SyN optimized for BOLD images
11. **SyNBoldAff**: SyN + Affine for BOLD images
12. **SyNAggro**: SyN with aggressive optimization
13. **TVMSQ**: Time-varying diffeomorphism with mean square metric

#### Similarity Metrics

- `CC`: Cross-correlation
- `MI`: Mutual information
- `MeanSquares`: Mean squares
- `Demons`: Demons metric

#### Optimizers

- `gradient_descent`: Gradient descent optimization
- `lbfgsb`: L-BFGS-B optimization
- `amoeba`: Amoeba optimization

#### Example Configuration

```yaml
registration:
  images: [t2, flair]
  fixed_image: t1
  type_of_transform: SyN
  metric: MI
  optimizer: gradient_descent
  use_mask: true
```

### 3. N4 Bias Field Correction

The N4 bias field correction step corrects intensity inhomogeneity in medical images, which is often caused by magnetic field inhomogeneity in MRI scanners.

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| num_fitting_levels | int | Number of fitting levels | 4 |
| num_iterations | List[int] | Iterations per level | [50] * num_fitting_levels |
| convergence_threshold | float | Convergence threshold | 0.001 |

#### Example Configuration

```yaml
n4_correction:
  images: [t1, t2, flair]
  num_fitting_levels: 4
  num_iterations: [50, 50, 50, 50]
  convergence_threshold: 0.001
```

## Best Practices

1. **Resampling**:
   - Choose target spacing based on your analysis needs
   - Use linear interpolation for images
   - Use nearest neighbor for masks

2. **Registration**:
   - Use SyN for deformable registration
   - Use MI metric for multi-modal registration
   - Always use masks when available
   - Consider using SyNRA for better initial alignment

3. **N4 Correction**:
   - Use 4 fitting levels for most cases
   - Adjust iterations based on image complexity
   - Use masks if available for better correction

## Common Issues and Solutions

1. **Registration Failures**:
   - Try different transformation types
   - Adjust similarity metric
   - Use masks if available
   - Check image orientation

2. **N4 Correction Issues**:
   - Increase number of iterations
   - Adjust convergence threshold
   - Use masks for better correction

3. **Memory Issues**:
   - Reduce number of parallel processes
   - Process images in smaller batches
   - Use lower resolution for initial registration

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

The pipeline maintains detailed logs in the `logs` directory. The log file includes:
- Processing progress
- Error messages
- Parameter settings
- Performance metrics

## Performance Considerations

1. **Parallel Processing**:
   - Use multiple processes for faster processing
   - Adjust number of processes based on available memory
   - Consider using fewer processes for memory-intensive operations

2. **Memory Usage**:
   - Monitor memory usage during processing
   - Adjust batch size if needed
   - Use appropriate image formats

3. **Disk Space**:
   - Processed images are saved in NIfTI format
   - Consider compression for long-term storage
   - Clean up temporary files regularly 