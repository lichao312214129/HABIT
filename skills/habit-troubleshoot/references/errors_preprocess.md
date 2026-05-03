# Preprocess Errors

## Symptom: `dcm2niix not found` / `[Errno 2] No such file or directory: 'dcm2niix.exe'`

**Cause**: `dcm2niix_path` in the dcm2nii config is wrong or the binary
isn't installed.

**Fix**:
1. Download dcm2niix: https://github.com/rordenlab/dcm2niix/releases
2. Extract somewhere stable (e.g. `./software/dcm2niix/dcm2niix.exe`)
3. Update the config to use **forward slashes** OR escaped backslashes:
   ```yaml
   dcm2niix_path: ./software/dcm2niix/dcm2niix.exe
   # NOT: D:\Tools\dcm2niix\dcm2niix.exe   <- backslashes break YAML
   ```

## Symptom: `KeyError: 'images'` or `KeyError: 'fixed_image'`

**Cause**: Required key missing under a method block.

**Fix**: Every method block under `Preprocessing:` must list `images:`.
For `registration:`, both `fixed_image:` and `moving_images:` are mandatory.

## Symptom: `RuntimeError: Inputs do not occupy the same physical space`

**Cause**: Trying to register or normalize images with mismatched grids.

**Fix**: Run `resample` BEFORE `registration` in the pipeline. All images
should be resampled to the same `target_spacing` first:

```yaml
Preprocessing:
  resample:
    images: [T1, T2, DWI, ADC]
    target_spacing: [1.0, 1.0, 1.0]

  registration:                # then register
    images: [T2, T1, DWI, ADC]
    fixed_image: T2
    moving_images: [T1, DWI, ADC]
```

## Symptom: registration takes hours per subject

**Cause**: Using `SyNAggro` or `BSplineSyN` (very slow) on large images.

**Fix**: Switch to `SyNRA` (default) or `Rigid`. SyNRA is rigid+affine+SyN
and works for >95% of cases. Use Rigid only if you don't expect deformation.

## Symptom: N4 makes the image worse

**Cause**: N4 was applied to a CT image (only valid for MRI).

**Fix**: Remove `n4_correction:` from the pipeline for CT data. CT has no
bias field — it's measured in HU.

## Symptom: `processed_images/` is empty after run

**Cause**: Preprocessing crashed silently for every subject. Check the log:

```bash
tail -100 <out_dir>/preprocess.log
```

Common subcauses:
1. Subject folder has no `images/` directory → fix layout
2. Mask file mismatch (mask doesn't overlap image) → manually fix mask
3. NaN in input → re-export from ITK-SNAP/Slicer

## Symptom: `Mask appears to be empty for subject X`

**Cause**: Mask file has all zeros OR mask label != 1.

**Fix**: Open in ITK-SNAP. If mask exists but uses label 2 (or another
integer), change downstream configs to use that label, OR re-save the mask
with label 1.

## Symptom: `MemoryError` during preprocessing

**Cause**: `processes:` set too high for available RAM. Each parallel
worker holds the full volume + intermediates.

**Fix**: Reduce `processes:` to 2 or 1. For 4D DCE volumes, `processes: 1`
is often safest.

## Symptom: dcm2nii produces 4D files but I want 3D per phase

**Cause**: Default dcm2niix merges multi-phase as 4D.

**Fix**: In the dcm2nii config:
```yaml
dcm2nii:
  merge_slices: "2"        # "2" = merge by series, gives separate 3D outputs
  single_file_mode: null   # let dcm2niix decide
```

## Symptom: pre-existing `processed_images/` got overwritten

**Cause**: Re-running preprocess with the same `out_dir` overwrites by design.

**Fix**: This is expected behavior. To preserve old results, change `out_dir`
or move/rename the existing folder before rerunning.
