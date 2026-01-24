# DICOM Information Extraction Tool

## Overview

This tool provides command-line functionality to extract and view information from DICOM files. It supports batch reading of DICOM files, can extract specified DICOM tag information, and save results in CSV, Excel, or JSON formats.

**Key Features**:
- Batch reading of DICOM files (supports directories, single files, or YAML config files)
- Automatic grouping by series to avoid duplicate reading of multiple slice files from the same series
- Support for extracting specified DICOM tags or all standard tags
- Multiple output formats (CSV, Excel, JSON)
- List all available tags in DICOM files

## Requirements

### Dependencies
- `pydicom`: DICOM file reading library
- `pandas`: Data processing
- `openpyxl`: Excel file support (if using Excel format)

### Installation
```bash
pip install pydicom pandas openpyxl
```

Or install all HABIT dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Command Format

```bash
habit dicom-info --input <input_path> [options]
```

### Common Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--input` | `-i` | DICOM directory, file, or YAML config file path (required) | - |
| `--tags` | `-t` | DICOM tags to extract, comma-separated | All standard tags |
| `--output` | `-o` | Output file path | Not saved |
| `--format` | `-f` | Output format: csv, excel, json | csv |
| `--recursive` | - | Search subdirectories recursively | Enabled |
| `--no-recursive` | - | Do not search subdirectories | - |
| `--group-by-series` | - | Group by series, read only one file per series | Enabled |
| `--no-group-by-series` | - | Read all files (no grouping) | - |
| `--one-file-per-folder` | - | Only read one DICOM file per folder to significantly speed up scanning | Disabled |
| `--dicom-extensions` | - | Comma-separated list of DICOM file extensions (e.g., `.dcm,.dicom,.ima`). Only used with `--one-file-per-folder` | .dcm,.dicom |
| `--include-no-extension` | - | Also check files without extensions by reading DICOM magic bytes. Only used with `--one-file-per-folder`. Useful for some medical devices that produce DICOM files without extensions | Disabled |
| `--num-workers` | `-j` | Number of worker threads for parallel processing. Default: min(32, cpu_count + 4). Set to 1 to disable parallelism | Auto |
| `--max-depth` | `-d` | Target folder depth. Quickly locates folders at specified depth, then finds **one** DICOM per folder as representative (stops as soon as found, doesn't scan all subfolders). 0=root, 1=root+1 level. Example: For patient/study/series/*.dcm structure, use `-d 2` to get one representative per study | Unlimited |
| `--list-tags` | - | List available tags instead of extracting | - |
| `--num-samples` | - | Number of files to sample when listing tags | 1 |

## Usage Examples

### 1. List Available DICOM Tags

View which tags are available in DICOM files:

```bash
habit dicom-info -i /path/to/dicom --list-tags
```

Save tag list to file:

```bash
habit dicom-info -i /path/to/dicom --list-tags -o available_tags.txt
```

### 2. Extract Specific Tags and Save as CSV

Extract patient name, study date, and modality:

```bash
habit dicom-info -i /path/to/dicom --tags "PatientName,StudyDate,Modality" --output results.csv
```

### 3. Extract All Standard Tags and Save as Excel

```bash
habit dicom-info -i /path/to/dicom --output dicom_info.xlsx --format excel
```

### 4. Read from YAML Config File

If your data is organized in a YAML config file:

```yaml
images:
  subject_001:
    T1: /path/to/subject_001/T1_dicom
    T2: /path/to/subject_001/T2_dicom
  subject_002:
    T1: /path/to/subject_002/T1_dicom
```

You can extract information like this:

```bash
habit dicom-info -i config.yaml --tags "PatientID,SeriesDescription" --output dicom_info.csv
```

### 5. Extract Information Without Saving (Display Only)

```bash
habit dicom-info -i /path/to/dicom --tags "PatientID,SeriesNumber,Modality"
```

### 6. Read All Files (No Series Grouping)

By default, the tool groups files by `SeriesInstanceUID` and reads only one representative file per series. If you need to read all files:

```bash
habit dicom-info -i /path/to/dicom --no-group-by-series --output all_files.csv
```

### 7. Non-Recursive Search

Search only in the specified directory, not subdirectories:

```bash
habit dicom-info -i /path/to/dicom --no-recursive --output results.csv
```

## Standard DICOM Tags

If the `--tags` option is not specified, the tool extracts the following standard tags:

**Patient Information**:
- PatientID, PatientName, PatientBirthDate, PatientSex, PatientAge

**Study Information**:
- StudyInstanceUID, StudyDate, StudyTime, StudyDescription

**Series Information**:
- SeriesInstanceUID, SeriesNumber, SeriesDescription, SeriesDate, SeriesTime

**Device Information**:
- Modality, Manufacturer, ManufacturerModelName

**Image Parameters**:
- SliceThickness, SpacingBetweenSlices, PixelSpacing
- Rows, Columns, BitsAllocated, BitsStored, HighBit
- ImagePositionPatient, ImageOrientationPatient

**Scan Parameters**:
- EchoTime, RepetitionTime, FlipAngle
- InstanceNumber, SliceLocation
- AcquisitionDate, AcquisitionTime

**Contrast Agent Information**:
- ContrastBolusAgent, ContrastBolusVolume

**X-ray Parameters** (CT):
- KVP, XRayTubeCurrent, ExposureTime

**Window Settings**:
- WindowCenter, WindowWidth

**Pixel Value Conversion**:
- RescaleIntercept, RescaleSlope

## Series Grouping Explanation

**Important**: By default, the tool groups files by `SeriesInstanceUID`. This is because:

1. **One series contains multiple slices**: In DICOM, a series typically contains multiple slice files (.dcm), where each slice is one layer of a 3D volume.

2. **Avoid duplicate information**: All slice files in the same series contain the same series-level information (such as PatientID, StudyDate, SeriesDescription), with only slice-specific information (such as InstanceNumber, SliceLocation) being different.

3. **Improve efficiency**: By grouping, the tool reads only one representative file per series, greatly reducing processing time and output file size.

**Grouping Behavior**:
- The tool reads the first file of each series
- The output includes a `Files_In_Series` column showing how many files are in that series
- Files without `SeriesInstanceUID` are processed individually

**Disable Grouping**:
If you need to view detailed information for each slice file (e.g., InstanceNumber, SliceLocation), you can use the `--no-group-by-series` option.

## Output Format Description

### CSV Format
- Comma-separated values
- Can be read by Excel, text editors, or pandas
- Recommended for data analysis and processing

### Excel Format
- .xlsx format
- Can be opened directly in Excel
- Suitable for viewing and manual editing

### JSON Format
- Structured data format
- Suitable for programmatic processing
- Each DICOM file's information as a JSON object

## Frequently Asked Questions

### Q: Why is there a `Files_In_Series` column in the output?
A: When using the `--group-by-series` option (default), the tool displays how many files are in each series. This helps understand the data scale.

### Q: How to extract custom tags?
A: Use the `--tags` option to specify tag names, for example:
```bash
habit dicom-info -i /path/to/dicom --tags "CustomTag1,CustomTag2" --output results.csv
```

### Q: What if DICOM files don't have SeriesInstanceUID?
A: The tool automatically handles this. Files without SeriesInstanceUID are processed individually and not grouped with other files.

### Q: Can I process multiple directories at once?
A: Yes. Use the `--recursive` option (enabled by default) to recursively search all subdirectories. Or use a YAML config file to specify multiple paths.

### Q: What if the output file is very large?
A: If the output file is very large, it might be because:
1. Series grouping is not used (use `--group-by-series`)
2. Too many tags are extracted (extract only needed tags)
3. The data volume is indeed large (consider batch processing)

## Technical Details

### File Recognition
The tool automatically recognizes files with the following extensions:
- `.dcm`
- `.dicom`

### Series Grouping Algorithm
1. Quickly read `SeriesInstanceUID` from each file (without reading the full file)
2. Group by `SeriesInstanceUID`
3. Select the first file of each series for full reading
4. Files without `SeriesInstanceUID` are processed individually

### Error Handling
- Files that cannot be read are skipped with a warning logged
- If all files cannot be read, the tool exits with an error message
- Missing some tags will not cause the entire file read to fail

## Related Documentation

- [Image Preprocessing Documentation](app_image_preprocessing.md)
- [DICOM Conversion Documentation](app_dcm2nii.md)

