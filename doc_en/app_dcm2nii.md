# DICOM to NIfTI Conversion Module

## Overview

This module provides the functionality to batch convert DICOM files to NIfTI format using the dcm2niix tool. It is integrated into the HABIT preprocessing pipeline and implemented through the `Dcm2niixConverter` class, supporting both single and batch conversion operations.

**Note**: This feature is not a standalone application script but is used as part of the preprocessing pipeline. To use it, please refer to `app_image_preprocessing.py`.

## Requirements

### Dependencies
- `dcm2niix` executable (must be installed and accessible in the system PATH)
- Python packages: `subprocess`, `pathlib`, `logging`
- HABIT dependencies: `BasePreprocessor`, `CustomTqdm`, `file_system_utils`

### Installing dcm2niix
1. Download dcm2niix: https://github.com/rordenlab/dcm2niix
2. Add dcm2niix.exe to the system PATH
3. Verify installation: `dcm2niix --help`

## Usage

### 1. Single DICOM Directory Conversion

```python
from habit.core.preprocessing import Dcm2niixConverter

# Initialize converter
converter = Dcm2niixConverter(
    keys=["dicom_path"],
    output_dir="/path/to/output",
    dcm2niix_executable="dcm2niix.exe",  # For Windows
    compress=True,
    anonymize=True
)

# Setup data
data = {
    "subject_id": "subject_001",
    "dicom_path": "/path/to/dicom/directory"
}

# Convert
result = converter(data)
```

### 2. Batch Conversion of Multiple Subjects

```python
from habit.core.preprocessing import batch_convert_dicom_directories

# Setup data structure
subjects_data = {
    "subject_001": {
        "T1": "/path/to/subject_001/T1_dicom",
        "T2": "/path/to/subject_001/T2_dicom"
    },
    "subject_002": {
        "T1": "/path/to/subject_002/T1_dicom"
    }
}

# Batch convert
converted_files = batch_convert_dicom_directories(
    input_mapping=subjects_data,
    output_dir="/path/to/output",
    dcm2niix_executable="dcm2niix.exe",
    compress=True,
    anonymize=True
)
```

### 3. Integration into Preprocessing Pipeline

```python
from habit.core.preprocessing import PreprocessorFactory

# Pipeline configuration
pipeline_config = [
    {
        "type": "dcm2niix_converter",
        "params": {
            "keys": ["dicom_path"],
            "output_dir": "/path/to/output",
            "dcm2niix_executable": "dcm2niix.exe",
            "compress": True
        }
    },
    {
        "type": "load_image",
        "params": {
            "keys": ["dicom_path_nifti"]
        }
    }
]

# Create and apply preprocessors
for step in pipeline_config:
    preprocessor = PreprocessorFactory.create(step["type"], **step["params"])
    data = preprocessor(data)
```

## Configuration Parameters

### Dcm2niixConverter Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keys` | Union[str, List[str]] | Required | Key name(s) for the DICOM directory path(s) |
| `output_dir` | str | Required | Output directory path |
| `dcm2niix_executable` | str | "dcm2niix" | Path to the dcm2niix executable |
| `filename_format` | Optional[str] | None | Output filename format |
| `compress` | bool | True | Whether to compress the output file |
| `anonymize` | bool | True | Whether to anonymize filenames |
| `ignore_derived` | bool | True | Whether to ignore derived images |
| `crop_images` | bool | True | Whether to crop images |
| `verbose` | bool | False | Whether to output detailed information |
| `batch_mode` | bool | True | Whether to enable batch mode |
| `allow_missing_keys` | bool | False | Whether to allow missing keys |

### dcm2niix Command Line Options

The mapping of module parameters to dcm2niix options is as follows:

- `-f`: filename_format (Output filename format)
- `-i y`: ignore_derived=True (Ignore derived images)
- `-l y`: batch_mode=True (Batch mode)
- `-p y`: anonymize=True (Anonymize filenames)
- `-x y`: crop_images=True (Crop images)
- `-v y`: verbose=True (Verbose output)
- `-z y`: compress=True (Compress output)
- `-o`: output_dir (Output directory)

## Output Structure

Converted files are organized in the following structure:

```
output_dir/
├── subject_001/
│   ├── T1/
│   │   └── subject_001_T1.nii.gz
│   └── T2/
│       └── subject_001_T2.nii.gz
├── subject_002/
│   └── T1/
│       └── subject_002_T1.nii.gz
└── conversion_summary.json
```

## Data Flow

### Input Data Format
```python
data = {
    "subject_id": "subject_001",
    "dicom_path": "/path/to/dicom/directory",
    # ... other keys
}
```

### Output Data Format
```python
data = {
    "subject_id": "subject_001",
    "dicom_path": "/path/to/dicom/directory",
    "dicom_path_nifti": "/path/to/output/subject_001/T1/subject_001_T1.nii.gz",
    "dicom_path_dcm2niix_meta": {
        "original_dicom_dir": "/path/to/dicom/directory",
        "converted_files": {"T1": "/path/to/output/subject_001/T1/subject_001_T1.nii.gz"},
        "conversion_params": {...}
    },
    # ... other keys
}
```

## Error Handling

### Common Errors

1. **dcm2niix not found**
   ```
   RuntimeError: dcm2niix executable not found: dcm2niix.exe
   ```
   - **Solution**: Ensure dcm2niix is installed and in the system PATH.

2. **DICOM directory not found**
   ```
   FileNotFoundError: Input directory does not exist: /path/to/dicom
   ```
   - **Solution**: Check if the DICOM directory path is correct.

3. **No NIfTI files created**
   ```
   RuntimeError: No NIfTI files were created for /path/to/dicom
   ```
   - **Solution**: Check if the DICOM files are valid and the dcm2niix parameters are correct.

### Error Recovery

- Setting `allow_missing_keys=True` will skip failed conversions.
- Check the log output for detailed error messages.
- Use `verbose=True` to get detailed output from dcm2niix.

## Examples

### Example 1: Use as part of the preprocessing pipeline
```bash
# Configure the dcm2niix conversion step in the config file
# Refer to config/config_image_preprocessing.yaml
python scripts/app_image_preprocessing.py --config ./config/config_image_preprocessing.yaml
```

### Example 2: Custom parameters
```python
converter = Dcm2niixConverter(
    keys=["dicom_path"],
    output_dir="./converted_nifti",
    filename_format="%p_%s",  # PatientID_SeriesName
    compress=False,  # Do not compress
    anonymize=False,  # Do not anonymize
    verbose=True  # Verbose output
)
```

### Example 3: Multi-sequence conversion
```python
subjects_data = {
    "patient_001": {
        "T1_MPRAGE": "/data/patient_001/T1_MPRAGE_dicom",
        "T2_FLAIR": "/data/patient_001/T2_FLAIR_dicom",
        "DWI": "/data/patient_001/DWI_dicom"
    }
}

converted = batch_convert_dicom_directories(
    input_mapping=subjects_data,
    output_dir="./multi_sequence_nifti",
    dcm2niix_executable="dcm2niix.exe"
)
```

## Performance Considerations

1. **Progress Tracking**: Uses CustomTqdm to display conversion progress.
2. **Memory Usage**: The conversion process does not load image data into memory.
3. **Disk Space**: Ensure the output directory has sufficient disk space.
4. **Parallel Processing**: Can process multiple subjects in parallel using multiprocessing.

## Integration with HABIT Pipeline

This module is fully integrated into the HABIT preprocessing pipeline:

1. **Factory Registration**: Registered via `@PreprocessorFactory.register("dcm2niix_converter")`.
2. **Base Class Inheritance**: Inherits from `BasePreprocessor`.
3. **Progress Reporting**: Uses the unified `CustomTqdm` progress bar.
4. **Error Handling**: Supports the `allow_missing_keys` parameter.
5. **Metadata Management**: Automatically generates conversion metadata.

## Troubleshooting

### Common Issues and Solutions

1. **Windows Path Issues**: Use forward slashes or raw strings, e.g., r"C:\path\to\dir".
2. **Permission Issues**: Ensure you have write permissions for the output directory.
3. **Filename Conflicts**: Use a unique `filename_format` to avoid overwriting files.
4. **DICOM Format Issues**: Check if the DICOM files conform to the standard.

### Debugging Tips

1. Set `verbose=True` to view dcm2niix output.
2. Check the log file for error messages.
3. Test with a small dataset first.
4. Verify that dcm2niix runs correctly standalone.

## Future Enhancements

1. **Parallel Processing**: Support for multi-process parallel conversion.
2. **Advanced Filtering**: Filter sequences based on DICOM tags.
3. **Quality Control**: Automatically check conversion quality.
4. **BIDS Compliance**: Support for BIDS format output.
5. **GUI Interface**: A graphical user interface.
