# dcm2niix 3D vs 4D Output Issue

## Problem Description

When running dcm2niix using Python's `subprocess.run` or `os.system`, you might encounter different output formats:

- **Python execution**: Outputs 4D image (e.g., 40 slices)
- **Direct terminal execution**: Outputs 3D image (e.g., 80 slices)

## Root Cause

When dcm2niix processes DICOM data containing multiple echoes, phases, or series, it decides how to organize the output based on its parameters:

1. **Missing `-m` parameter (merge slices)**:
   - By default, dcm2niix attempts to merge 2D slices
   - However, for certain data (especially multi-echo), it may split into multiple 3D volumes or 4D volumes

2. **Missing `-s` parameter (single file mode)**:
   - Without forcing single file mode, dcm2niix may output multiple files
   - Or organize data in 4D format (time dimension)

## Solution

### 1. Add Critical Parameters

Two key parameters have been added to `habit/core/preprocessing/dcm2niix_converter.py`:

#### `merge_slices` (corresponds to dcm2niix's `-m` parameter)
Options:
- **`"y"` or `"1"`**: Default merge behavior (merge 2D slices)
- **`"2"` (recommended for 3D output)**: Merge based on series (more aggressive merging)
- **`"n"` or `"0"`**: No merging, keep original structure
- **`None`**: Don't specify parameter, use dcm2niix default

#### `single_file_mode` (corresponds to dcm2niix's `-s` parameter)
Options:
- **`None` (strongly recommended)**: Don't specify parameter, let dcm2niix auto-decide best format
- **`True`**: Force single file output (`-s y`), may preserve 4D structure
- **`False`**: Allow multiple file output (`-s n`), may split volumes

### 2. Configuration File Settings

Add these two parameters to your configuration file:

```yaml
Preprocessing:
  dcm2nii:
    images: [delay2, delay3, delay5]
    dcm2niix_path: path/to/dcm2niix.exe
    compress: true
    anonymize: false
    merge_slices: "2"        # Use "2" for more aggressive merging
    single_file_mode: null   # Use null to let dcm2niix auto-decide (recommended)
```

**Important Notes**:
- For 4D→3D issues, **recommended settings**: `merge_slices: "2"` and `single_file_mode: null`
- `single_file_mode: null` means don't pass `-s` parameter to dcm2niix
- If still having issues, try setting `merge_slices` to `null` as well, using dcm2niix complete defaults

### 3. Verify Output

After conversion, use the following methods to verify the output:

#### Check Image Dimensions with Python

```python
import SimpleITK as sitk

# Read converted image
image = sitk.ReadImage('output.nii.gz')

# Get image size
size = image.GetSize()
print(f"Image size: {size}")
print(f"Dimensions: {len(size)}D")

# 3D image should be (width, height, depth)
# e.g.: (512, 512, 80)

# 4D image would be (width, height, depth, time)
# e.g.: (512, 512, 40, 2)
```

#### Use Command-line Tools

```bash
# Use ITK-SNAP or 3D Slicer to view
# Or use Python script
python -c "import SimpleITK as sitk; img=sitk.ReadImage('output.nii.gz'); print(f'Size: {img.GetSize()}, Dimensions: {len(img.GetSize())}D')"
```

## Technical Details

### dcm2niix Parameter Explanation

- **`-m y`**: Merge 2D slices from same series
- **`-m n`**: Do not merge 2D slices
- **`-m 2`**: Merge 2D slices based on series only

- **`-s y`**: Single file mode
- **`-s n`**: Multiple file mode (default)

### Why Do Terminal and Python Results Differ?

Possible reasons:
1. **Environment variable differences**: Different PATH or other environment variables between terminal and Python
2. **Default parameters**: Terminal might use different default parameters or configuration files
3. **Working directory**: Different working directories may affect dcm2niix behavior
4. **dcm2niix version**: Different versions may have different default behaviors

## FAQ

### Q: I've set the parameters but still get 4D images. What should I do?

A: Follow these troubleshooting steps:

1. **Check the actual executed command**:
   - When running Python program, the console will print the actual dcm2niix command
   - Copy that command and run it directly in terminal to see if you get the same issue
   
2. **Compare terminal command with Python command**:
   ```bash
   # Assume Python outputs this command:
   dcm2niix.exe -b n -l y -m 2 -p n -v y -z y -o output input
   
   # Try different parameter combinations directly in terminal:
   # Option 1: Without -m and -s parameters
   dcm2niix.exe -b n -l y -p n -v y -z y -o output input
   
   # Option 2: Use -m n (no merge, may output multiple 3D files)
   dcm2niix.exe -b n -l y -m n -p n -v y -z y -o output input
   
   # Option 3: Use -m y (default merge)
   dcm2niix.exe -b n -l y -m y -p n -v y -z y -o output input
   ```

3. **Try different configuration combinations**:
   ```yaml
   # Option 1: Don't specify any parameters (recommended first try)
   merge_slices: null
   single_file_mode: null
   
   # Option 2: Use -m 2
   merge_slices: "2"
   single_file_mode: null
   
   # Option 3: No merging (may output multiple 3D files)
   merge_slices: "n"
   single_file_mode: null
   ```

4. **Check DICOM data**:
   - Confirm if DICOM data actually contains multiple phases/echoes
   - Review dcm2niix output logs to understand how it's parsing the data
   - Use `verbose: true` parameter for more information

### Q: How can I test temporarily without modifying code?

A: Run dcm2niix command directly in terminal for testing:

```bash
dcm2niix.exe -m y -s y -z y -b n -o output_dir input_dicom_dir
```

Parameter explanation:
- `-m y`: Merge slices
- `-s y`: Single file mode
- `-z y`: Compress output
- `-b n`: Don't generate JSON files
- `-o output_dir`: Output directory
- `input_dicom_dir`: Input DICOM directory

### Q: Should I always use merge_slices and single_file_mode?

A: **Recommended to set True**, unless you have special requirements:
- If you need to preserve 4D structure of multi-echo or multi-phase data, set to False
- If you need to process each phase separately, set single_file_mode to False

## References

- [dcm2niix GitHub](https://github.com/rordenlab/dcm2niix)
- [dcm2niix Documentation](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage)

## Changelog

- **2025-10-29**: Added `merge_slices` and `single_file_mode` parameters to resolve 3D/4D output issue

