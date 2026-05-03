# DICOM Tag Cheat Sheet for `habit dicom-info`

Common tags users want to extract for cohort QC. Pass them as a comma-separated
list to `habit dicom-info -t "Tag1,Tag2,..."`.

## Patient metadata

| Tag | Meaning |
|---|---|
| `PatientID` | de-identified or original patient ID |
| `PatientName` | (often anonymized; can be sensitive) |
| `PatientSex` | M / F / O |
| `PatientAge` | format like `045Y` |
| `PatientBirthDate` | YYYYMMDD (if not stripped) |

## Study / Series

| Tag | Meaning |
|---|---|
| `StudyInstanceUID` | unique per imaging session |
| `SeriesInstanceUID` | unique per acquisition (sequence) |
| `StudyDate` | YYYYMMDD |
| `StudyTime` | HHMMSS |
| `StudyDescription` | free text (e.g. "Liver MRI Protocol") |
| `SeriesDescription` | free text (e.g. "T2W TSE", "DWI b=800", "Arterial Phase") |
| `SeriesNumber` | order within the study |
| `Modality` | CT / MR / PT / US |

## Acquisition geometry

| Tag | Meaning |
|---|---|
| `SliceThickness` | mm |
| `PixelSpacing` | "row\\col" in mm |
| `SpacingBetweenSlices` | mm; may differ from SliceThickness |
| `Rows` | image height (pixels) |
| `Columns` | image width (pixels) |
| `ImageOrientationPatient` | 6 numbers; coordinate frame |
| `ImagePositionPatient` | 3 numbers; first voxel position |

## MRI-specific

| Tag | Meaning |
|---|---|
| `MagneticFieldStrength` | Tesla, e.g. 1.5 or 3.0 |
| `RepetitionTime` | TR (ms) |
| `EchoTime` | TE (ms) |
| `FlipAngle` | degrees |
| `EchoTrainLength` | for fast spin echo |
| `ScanningSequence` | SE / GR / IR / EP |
| `SequenceVariant` | SK / SP / OSP / MTC |
| `DiffusionBValue` | for DWI |
| `ContrastBolusAgent` | name of contrast (Gd-DTPA, ...) |

## CT-specific

| Tag | Meaning |
|---|---|
| `KVP` | tube voltage |
| `XRayTubeCurrent` | mA |
| `Exposure` | mAs |
| `ConvolutionKernel` | reconstruction kernel (e.g. STANDARD, B30f) |
| `RescaleIntercept` | for HU conversion |
| `RescaleSlope` | for HU conversion |
| `ContrastBolusAgent` | (also CT) |

## Equipment

| Tag | Meaning |
|---|---|
| `Manufacturer` | SIEMENS / GE / PHILIPS / TOSHIBA |
| `ManufacturerModelName` | scanner model |
| `StationName` | which scanner physically |
| `InstitutionName` | hospital name |
| `SoftwareVersions` | scanner software version |

## Cohort QC starter set

```bash
habit dicom-info -i ./raw_dicoms \
  -t "PatientID,StudyDate,Modality,SeriesDescription,Manufacturer,ManufacturerModelName,MagneticFieldStrength,SliceThickness,PixelSpacing,KVP,Rows,Columns" \
  -o cohort_qc.csv \
  --one-file-per-folder \
  --max-depth 3
```

Open the CSV and check:
- Mixed magnetic field strengths (1.5T + 3T) → harmonization may be needed
- Mixed manufacturers → expect intensity bias; use histogram standardization
- Slice thickness > 5mm → warn user about reduced 3D shape feature reliability
- Mixed pixel spacing → resample is mandatory before any further analysis

## Listing all tags from a sample file

If you don't know which tags exist:

```bash
habit dicom-info -i ./raw_dicoms --list-tags --num-samples 3
```

Then add the interesting ones to your `-t` list.
