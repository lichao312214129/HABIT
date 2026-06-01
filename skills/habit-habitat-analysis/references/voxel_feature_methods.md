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
**Slow and memory-hungry.** For **CT habitat** voxel texture, use the **R3B12** preset
(`kernelRadius: 3`, `binWidth: 12` HU) from Petersen et al. (*Radiol Artif Intell*
2024;6(2):e230118, doi:10.1148/ryai.230118) — better repeatability/reproducibility than R1B25.
Use `config/radiomics/params_voxel_radiomics.yaml`:
GLCM must list **21 stable features** (exclude MCC/Imc1/Imc2). Bare `glcm:`
enables all 24; on `kernelRadius=1–3` many neighborhoods are flat, GLCM
matrices degenerate, and MCC/Imc1/Imc2 trigger CUDA/MKL `eigvals` errors.
HABIT substitutes the safe list when `glcm` is unrestricted and logs a warning.

```yaml
voxel_level:
  method: voxel_radiomics(T2)
  params:
    params_file: ./config/radiomics/params_voxel_radiomics.yaml
    kernelRadius: 3          # CT R3B12 default: 3 → 7×7×7; 1=3×3×3 (habit param, not in params_file)
    voxelBatch: 1000         # habit default; -1 = no batching
    useTorchRadiomics: auto  # auto uses torch+CUDA when available
    # torchGpus: [0, 1]      # allowed GPU indices
    # torchGpuCount: 2       # optional: use first N GPUs from torchGpus
```

Recommended `params_voxel_radiomics.yaml` (PyRadiomics settings only):

```yaml
featureClass:
  firstorder:
  glcm:
    - Contrast
    - Correlation
    - JointEnergy
    - Idm
    - Autocorrelation
    - JointAverage
    - JointEntropy
    - DifferenceAverage
    - DifferenceEntropy
    - SumAverage
    - SumEntropy
    - SumSquares
    - MaximumProbability
    - Idmn
    - Id
    - Idn
    - InverseVariance
    - DifferenceVariance
    - ClusterTendency
    - ClusterShade
    - ClusterProminence
setting:
  binWidth: 12               # CT R3B12 (see params_voxel_radiomics.yaml header reference)
  voxelArrayShift: 300
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
5. For `supervoxel_radiomics`, use `params_supervoxel_radiomics.yaml` and set
   `supervoxelBatch` / torch keys under `supervoxel_level.params` (see below).

## Supervoxel-level: `supervoxel_radiomics(<img>)`

Extracts **whole-ROI** PyRadiomics texture per supervoxel label (not a local kernel).

```yaml
supervoxel_level:
  supervoxel_file_keyword: '*_supervoxel.nrrd'
  method: supervoxel_radiomics(T2)
  params:
    params_file: ./config/radiomics/params_supervoxel_radiomics.yaml
    supervoxelBatch: 64          # habit default; batch group size for label loops
    useSupervoxelCext: auto        # auto = C extension when built, else prior Torch/PyRadiomics path
    useTorchRadiomics: auto      # inherits from voxel_level.params if omitted
    # torchGpus: [0, 1]
    # torchGpuCount: 2
    # torchDtype: float32
```

**Binning semantics:** one PyRadiomics discretization on the union mask (`sv_map > 0`),
then per-label `cMatrices` ROI matrices — analogous in spirit to voxel-level whole-mask
binning, but each unit is a supervoxel ROI. Values differ from legacy per-label
`execute()` (per-label bin).

**Not used:** `kernelRadius` (voxel_radiomics only).

**Matrix backend:** `useSupervoxelCext: auto` (default) uses habit native C-extension batched
matrices when compiled; otherwise the prior Torch/PyRadiomics stacked-matrix path.

**Feature backend:** `useTorchRadiomics: auto` + CUDA → TorchRadiomics GPU; otherwise CPU
PyRadiomics with the same union-mask bin path.

Compare with `mean_voxel_features()` when you already have `voxel_level` features and
only need aggregation (faster, consistent with voxel pipeline).
