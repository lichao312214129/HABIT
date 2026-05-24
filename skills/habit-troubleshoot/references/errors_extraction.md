# Feature Extraction Errors

## Symptom: `KeyError: 'params_file_of_non_habitat'` or `params_file_of_habitat`

**Cause**: HABIT requires both keys to be present in the config even if you
don't extract radiomics-based features.

**Fix**: Add both keys (point them at any existing PyRadiomics file):
```yaml
params_file_of_non_habitat: ./config/radiomics/parameter.yaml
params_file_of_habitat: ./config/radiomics/parameter_habitat.yaml
```

## Symptom: `No habitat files found matching pattern '*_habitats_remapped.nrrd'`

**Cause**: Your habitat clustering used **one_step** mode, which doesn't
produce `_remapped` files (only two_step does).

**Fix**: Change the pattern in the config:
```yaml
habitat_pattern: '*_habitats.nrrd'   # for one_step output
# or
habitat_pattern: '*_habitats_remapped.nrrd'   # for two_step output
```

## Symptom: PyRadiomics fails with `Image and mask have different geometry`

**Cause**: After preprocessing, image and mask may have slightly different
spacing/origin due to floating point rounding.

**Fix**: HABIT auto-resamples masks to match images during extraction. If
this still fails, check that preprocessing wrote both images AND masks to
`processed_images/` (not just images).

## Symptom: `voxel_radiomics` / TorchRadiomics crash on GLCM (MKL, eigvals, MCC)

**Cause**: `params_file` enables `glcm:` without an explicit feature list (all 24
GLCM features). Voxel mode uses small neighborhoods (`kernelRadius=1–3`); flat
patches yield degenerate 1×1 GLCM matrices. **MCC, Imc1, Imc2** need eigenvalue
or mutual-information statistics and crash on CUDA/MKL or return NaN.

**Fix**:
1. Use `config/radiomics/params_voxel_radiomics.yaml` (21 stable GLCM features).
2. Or list GLCM features explicitly — **exclude MCC, Imc1, Imc2** for voxel mode.
3. HABIT auto-substitutes the safe list when `glcm` is unrestricted and logs a
   warning; explicit unsafe names are not removed.

## Symptom: All output CSVs have NaN-only columns

**Cause**: PyRadiomics extraction silently failed for those features —
usually because the ROI is too small for that feature class.

**Fix**:
1. Check ROI volumes: `python skills/habit-quickstart/scripts/check_data_layout.py <data_dir>`
2. ROIs < 30 voxels can't compute GLCM/GLRLM reliably
3. Either redraw larger masks OR remove problematic feature classes from
   the params YAML

## Symptom: `MemoryError` during extraction

**Cause**: `n_processes` too high, or `each_habitat` enabled with too many
habitats.

**Fix**:
- Lower `n_processes: 2`
- Remove `each_habitat` from `feature_types` (it generates 5×N features
  for 5 habitats × N base features)

## Symptom: extraction is super slow (hours per subject)

**Cause**: `parameter.yaml` has Wavelet + LoG + all 7 feature classes.

**Fix**: Use `config/radiomics/parameter_basic.yaml` for testing:
```yaml
imageType:
  Original: {}
featureClass:
  firstorder:
  shape:
  glcm:
```

Then enable Wavelet/LoG only after the rest works.

## Symptom: `subject IDs don't match between raw_img_folder and habitats_map_folder`

**Cause**: Subject naming inconsistency between the two folders.

**Fix**:
1. List both folders:
   ```bash
   ls <raw_img_folder>
   ls <habitats_map_folder>
   ```
2. Common reasons:
   - One has `subj001`, the other has `sub001`
   - One has trailing spaces or hidden files
   - habitat map files are nested inside a per-subject folder vs flat
3. Fix the naming, then re-run.

## Symptom: `each_habitat` produces 1500+ feature columns

**Cause**: Expected behavior — 5 habitats × ~300 PyRadiomics features = 1500.

**Fix** (if undesirable):
- Remove `each_habitat` from `feature_types`
- OR use `config/radiomics/parameter_basic.yaml` to lower per-habitat feature count

## Symptom: `whole_habitat_radiomics.csv` is empty (0 rows)

**Cause**: No subject had a successful habitat extraction.

**Fix**:
1. Verify habitat files exist: `ls <habitats_map_folder>/*/*_habitats*.nrrd`
2. Verify the habitat label maps have non-zero labels (use ITK-SNAP)
3. Check the extraction log for "skipping" messages
