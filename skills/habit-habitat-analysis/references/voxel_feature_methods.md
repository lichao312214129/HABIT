# Voxel-Level Feature Methods for Habitat Clustering

`FeatureConstruction.voxel_level.method` controls **what gets clustered**.
This is the most important biological choice in the whole habitat analysis.

## Method comparison matrix

| Method | What it captures | Speed | Best for | Required extras |
|---|---|---|---|---|
| `raw(<img>)` | Single-image intensity | Fast | T2-only / single-sequence pilot | None |
| `concat(raw(A), raw(B), ...)` | Joint intensity across modalities | Fast | Multi-modal MRI (T1+T2+DWI+ADC) | All listed images present per subject |
| `kinetic(raw(p1), ..., timestamps)` | Time-intensity curve features | Medium | DCE-MRI / dynamic CT | Excel timestamp file |
| `voxel_radiomics(<img>)` | Local texture (firstorder, GLCM…) | Slow | Texture-rich tumors (HCC, glioma) | PyRadiomics params YAML |
| `local_entropy(<img>)` | Local intensity disorder | Medium | Heterogeneity-focused study | None |

## Detailed guidance

### `raw(<img>)`
The simplest baseline. Produces 1 feature per voxel. Use only when you have a
single representative sequence and you just want to demonstrate that habitat
analysis works.

### `concat(raw(...), raw(...), ...)`
The most common production choice. Each voxel becomes a vector of intensities
across all listed sequences. Make sure all sequences are **registered and
intensity-normalized** before this — otherwise voxels in unregistered images
will be clustered together by accident.

```yaml
voxel_level:
  method: concat(raw(T1), raw(T2), raw(DWI), raw(ADC))
  params: {}
```

### `kinetic(...)`
Derives time-intensity curve features (peak enhancement, wash-in slope, etc.)
from multi-phase contrast images. Phase order MUST match the column order in
your timestamps Excel file.

```yaml
voxel_level:
  method: kinetic(raw(pre_contrast), raw(LAP), raw(PVP), raw(delay_3min), timestamps)
  params:
    timestamps: ./data/scan_time_of_phases.xlsx
```

Excel layout:

| subjID  | pre_contrast | LAP | PVP | delay_3min |
|---------|--------------|-----|-----|------------|
| sub001  | 0.0          | 0.5 | 1.5 | 3.0        |
| sub002  | 0.0          | 0.5 | 1.5 | 3.0        |

Times are in **minutes from injection**. Subject IDs must match folder names.

### `voxel_radiomics(<img>)`
Computes PyRadiomics features for each voxel using its local neighborhood.
**Slow and memory-hungry.** HABIT auto-restricts GLCM to safe features
(Contrast, Correlation, JointEnergy, Idm) when `kernelRadius<=3` to prevent
crashes on small neighborhoods.

```yaml
voxel_level:
  method: voxel_radiomics(T2)
  params:
    params_file: ./config/params_voxel_radiomics.yaml
```

Recommended `params_voxel_radiomics.yaml`:

```yaml
featureClass:
  firstorder:
  glcm:                    # auto-restricted
setting:
  binWidth: 25
  kernelRadius: 1          # 1=3x3x3 ; 2=5x5x5
  maskedKernel: true
```

### `local_entropy(<img>)`
Sliding-window Shannon entropy. Cheaper than full voxel_radiomics and good
when "heterogeneity" is the only signal you care about.

```yaml
voxel_level:
  method: local_entropy(T2)
  params:
    window_size: [3, 3, 3]
```

## Decision tree

```
Single sequence available?
  -> raw(<seq>)

Multi-modal MRI/CT, no DCE?
  -> concat(raw(M1), raw(M2), ...)

DCE-MRI or dynamic CT?
  -> kinetic(raw(p1), ..., timestamps)
  Need an accurate timestamps Excel.

Want texture-driven habitats?
  -> voxel_radiomics(<seq>) if cohort small (<100) and you can afford runtime
  -> local_entropy(<seq>) for a faster compromise
```

## Sanity-checks before launching

1. Every modality referenced inside `raw(...)` must exist for every subject.
2. All modalities must be **co-registered** (use `habit-preprocess` first).
3. For `kinetic`, every subject ID in the timestamps Excel must match a
   subject folder name.
4. For `voxel_radiomics`, the params file must exist and be valid PyRadiomics
   YAML.
