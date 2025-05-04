# MONAI-based Image Preprocessing

This document provides a guide on how to use the MONAI-based image preprocessing functionality in HABIT.

## Overview

The MONAI-based preprocessing module leverages the powerful [MONAI](https://monai.io/) library for medical image preprocessing. MONAI provides a comprehensive set of features for medical image processing, including:

- Data loading and augmentation
- Preprocessing transforms
- GPU acceleration
- Caching mechanisms for improved performance

## Installation

To use the MONAI-based preprocessing, you need to install MONAI and its dependencies:

```bash
pip install monai torch
```

## Data Structure

The preprocessing module expects data to be organized in the following structure:

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

## Configuration

The preprocessing configuration is specified in a YAML file. Here's an example configuration:

```yaml
# Preprocessing settings
Preprocessing:
  # N4 Bias Field Correction
  n4biascorrection:
    images: [pre_contrast, LAP, PVP, delay_3min]
    n_iterations: [50, 50, 30, 20]
    convergence_threshold: 0.001
    
  # Resampling to uniform spacing
  spacing:
    images: [pre_contrast, LAP, PVP, delay_3min]
    target_spacing: [1.0, 1.0, 1.0]  # Target voxel spacing in mm
    mode: bilinear
    
  # Orientation standardization
  orientation:
    images: [pre_contrast, LAP, PVP, delay_3min]
    axcodes: "RAS"  # Right, Anterior, Superior orientation
    
  # Intensity normalization
  normalize:
    images: [pre_contrast, LAP, PVP, delay_3min]
    nonzero: True  # Only normalize non-zero values
    channel_wise: True
```

### Configuration Options

Each preprocessor in the configuration has the following structure:

```yaml
preprocessor_name:
  images: [image_type1, image_type2, ...]  # Which images to process
  param1: value1  # Preprocessor-specific parameters
  param2: value2
```

The `images` field specifies which image types to process. The other fields are specific to each preprocessor.

## Available Preprocessors

The following preprocessors are available:

### N4 Bias Field Correction

Corrects intensity non-uniformity in MR images.

```yaml
n4biascorrection:
  images: [pre_contrast, LAP, PVP, delay_3min]
  n_iterations: [50, 50, 30, 20]
  convergence_threshold: 0.001
```

### Spacing

Resamples images to a uniform spacing.

```yaml
spacing:
  images: [pre_contrast, LAP, PVP, delay_3min]
  target_spacing: [1.0, 1.0, 1.0]  # Target voxel spacing in mm
  mode: bilinear
```

### Orientation

Standardizes image orientation.

```yaml
orientation:
  images: [pre_contrast, LAP, PVP, delay_3min]
  axcodes: "RAS"  # Right, Anterior, Superior orientation
```

### Normalize

Normalizes image intensity.

```yaml
normalize:
  images: [pre_contrast, LAP, PVP, delay_3min]
  nonzero: True  # Only normalize non-zero values
  channel_wise: True
```

### Scale Intensity

Scales image intensity to a specific range.

```yaml
scaleintensity:
  images: [pre_contrast, LAP, PVP, delay_3min]
  minv: 0.0
  maxv: 1.0
```

### Crop Foreground

Crops the image to the foreground region.

```yaml
cropforeground:
  images: [pre_contrast, LAP, PVP, delay_3min]
  source_key: PVP  # Use this image to determine foreground
  margin: 10  # Margin around foreground in pixels
```

### Registration

Registers images to a reference image.

```yaml
registration:
  images: [pre_contrast, LAP, PVP, delay_3min]
  fixedImage: PVP
  movingImage: [pre_contrast, LAP, delay_3min]
  method: rigid  # rigid or affine
```

## Usage

### Command Line

You can run the preprocessing from the command line using the `run_monai_preprocessing.py` script:

```bash
python scripts/run_monai_preprocessing.py --config config/monai_preprocessing_config.yaml --data_dir /path/to/data --out_dir /path/to/output --verbose
```

### Python API

You can also use the preprocessing module in your Python code:

```python
from habit.core.preprocessing import MonaiPreprocessor, load_config

# Load configuration
config = load_config('config/monai_preprocessing_config.yaml')

# Create preprocessor
preprocessor = MonaiPreprocessor(
    config=config,
    root_folder='/path/to/data',
    out_folder='/path/to/output',
    verbose=True
)

# Run preprocessing
preprocessor.run()
```

## Advanced Usage

### Custom Transforms

You can extend the `MonaiPreprocessor` class to add custom transforms:

```python
from habit.core.preprocessing import MonaiPreprocessor
from monai.transforms import Compose, RandRotated

class CustomPreprocessor(MonaiPreprocessor):
    def _create_transforms(self):
        # Get base transforms
        transforms = super()._create_transforms()
        
        # Add custom transforms
        custom_transforms = [
            RandRotated(
                keys=['pre_contrast', 'LAP', 'PVP', 'delay_3min'],
                range_x=0.2,
                range_y=0.2,
                range_z=0.2,
                prob=0.5
            )
        ]
        
        # Combine transforms
        return Compose([transforms] + custom_transforms)
```

### GPU Acceleration

By default, the preprocessing will use GPU acceleration if available. You can specify the device explicitly:

```bash
python scripts/run_monai_preprocessing.py --config config/monai_preprocessing_config.yaml --data_dir /path/to/data --device cuda:0
```

### Parallel Processing

You can control the number of worker processes for data loading:

```bash
python scripts/run_monai_preprocessing.py --config config/monai_preprocessing_config.yaml --data_dir /path/to/data --num_workers 8
```

### Batch Processing

You can control the batch size for data loading:

```bash
python scripts/run_monai_preprocessing.py --config config/monai_preprocessing_config.yaml --data_dir /path/to/data --batch_size 4
```

## Troubleshooting

### Out of Memory Errors

If you encounter out of memory errors, try:

1. Reducing the batch size: `--batch_size 1`
2. Reducing the number of worker processes: `--num_workers 2`
3. Using CPU instead of GPU: `--device cpu`

### Missing Images

If some images are not being processed, check:

1. The data structure matches the expected format
2. The image types in the configuration match the actual image types in the data
3. The image files have the expected extensions (.nii.gz, .nii, .mha)

### Registration Issues

For complex registration tasks, consider using MONAI's registration module separately or a dedicated registration tool like ANTs or Elastix.
