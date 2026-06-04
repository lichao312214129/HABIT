---
name: habit-habitat-analysis
description: Generate tumor habitat (sub-region) maps from medical images using voxel-level clustering. Use when the user wants to identify intra-tumor heterogeneity zones, perform supervoxel clustering, generate habitat .nrrd maps, run kinetic DCE habitat, or use voxel-radiomics texture clustering. Triggers on "生境分析", "亚区聚类", "habitat", "supervoxel", "kinetic", "DCE 分期生境", "intra-tumor heterogeneity". Runs `habit get-habitat`.
---

# HABIT Habitat Analysis (Core)

The core function of HABIT: cluster voxels inside the ROI to discover sub-regions
("habitats") that may correspond to different biological behaviors (necrosis,
proliferation, hypoxia, etc.).

## CLI

```bash
# Train mode (default)
habit get-habitat --config <config.yaml>

# Predict mode (apply pre-trained habitat model to new patients)
habit get-habitat --config <config.yaml> --mode predict --pipeline <path/to/pipeline.pkl>

# Debug mode
habit get-habitat --config <config.yaml> --debug
```

## Required Information

Before generating a config, confirm:

| Field | Notes |
|---|---|
| `data_dir` (preprocessed) | output of `habit preprocess` |
| `out_dir` | output root |
| Modalities to cluster | drives `voxel_level.method` |
| `clustering_mode` | `one_step`, `two_step`, or `direct_pooling` (see below) |
| (kinetic only) timestamps Excel | required |
| (voxel_radiomics only) PyRadiomics params YAML | required |
| (voxel_radiomics only) `kernelRadius` / `voxelBatch` | optional; CT preset **3** / 1000 (`-1` = no batching); see R3B12 ref below |
| (voxel_radiomics only) `useTorchRadiomics` / `torchGpus` / `torchGpuCount` | optional; GPU pool + count |
| `processes` / `cap_processes_to_gpu_pool` | Stage-1 parallelism; default `processes: 2`, `cap_processes_to_gpu_pool: true` |
| (supervoxel_radiomics only) PyRadiomics params YAML | `params_supervoxel_radiomics.yaml` |
| (supervoxel_radiomics only) `supervoxelBatch` | optional; default 64 |
| (supervoxel_radiomics only) `useSupervoxelCext` | optional; default auto (C extension when built, else prior path) |
| (supervoxel_radiomics only) torch keys | inherit from `voxel_level.params` if omitted |
| Expected habitat count | typical 3-5 |

## Decision 1 — one_step vs two_step

ALWAYS ask the user this first.

### one_step (recommended for beginners)
- Each tumor independently determines its optimal cluster number (silhouette / elbow)
- No population-level model — habitat labels are per-subject local meaning
- Faster, simpler
- Use template: `config/habitat/config_getting_habitat.yaml`

### two_step (publications / predictive modeling)
- **Step 1**: voxel → supervoxel (per subject; e.g. 50 supervoxels per tumor)
- **Step 2**: supervoxel → habitat (population-level model; same habitat label = same biology across all patients)
- Slower, requires train+test cohort
- Generates `supervoxel2habitat_clustering_model.pkl` for `--mode predict`
- Use template: `config/habitat/config_getting_habitat.yaml`

If unsure → start with **one_step**.

## Decision 2 — Voxel feature method

`FeatureConstruction.voxel_level.method` defines what voxel features get clustered.
This is the most important biological choice.

| Data type | Method | Template |
|---|---|---|
| Single modality | `raw(<seq>)` | one_step / two_step minimal |
| Multi-modal (most common) | `concat(raw(M1), raw(M2), ...)` | one_step / two_step minimal |
| DCE-MRI | `kinetic(raw(p1), ..., timestamps)` | `config/habitat/config_getting_habitat.yaml` |
| Texture-based | `voxel_radiomics(<seq>)` | `config/habitat/config_getting_habitat.yaml` |
| Local entropy | `local_entropy(<seq>)` | (manual) |

Detailed comparison: `references/voxel_feature_methods.md`.

For `kinetic`: user MUST provide a timestamps Excel with subject IDs and
per-phase scan times. Without this, fail fast.

For `voxel_radiomics`: use `config/radiomics/params_voxel_radiomics.yaml` for
GLCM — bare `glcm:` in a params file enables all 24 features; MCC/Imc1/Imc2
crash on small kernels. HABIT defaults unrestricted GLCM to 21 stable features.
**CT voxel texture (R3B12):** `kernelRadius: 3` in habitat YAML, `binWidth: 12` in
`params_voxel_radiomics.yaml` (Petersen et al., *Radiol Artif Intell* 2024;6(2):e230118,
doi:10.1148/ryai.230118)
and logs a warning. Optional `torchGpus` selects which CUDA devices may be used; `torchGpuCount`
limits how many of them are active. Stage-1 workers receive `gpuSlotIndex` from the parallel
pool (default `cap_processes_to_gpu_pool: true` caps workers to `len(torchGpus)`). Set
`cap_processes_to_gpu_pool: false` to keep full `processes` on 1-GPU / many-CPU hosts (workers
share GPUs via modulo mapping). See `references/voxel_feature_methods.md`.

For `supervoxel_radiomics` (two_step Step 2 input): use
`config/radiomics/params_supervoxel_radiomics.yaml`; union-mask binning + per-label
ROI extraction. **`method` must use an outer combiner** (typically `concat(...)`), even for
one modality, e.g. `concat(supervoxel_radiomics(T2, params_file))` with `params_file` path
in `supervoxel_level.params`. Set `supervoxelBatch`, `useSupervoxelCext` (default auto), and
torch keys under `supervoxel_level.params` (inherit torch keys from `voxel_level.params` when
omitted). See `references/voxel_feature_methods.md` (Supervoxel-level section).

## Population-level preprocessing (two_step only)

```yaml
FeatureConstruction:
  preprocessing_for_group_level:
    methods:
      - method: binning
        n_bins: 10
        bin_strategy: uniform
```

**Recommendation**: `binning` with `n_bins: 10` is the publication default.

## Optional postprocess (clean tiny fragments)

```yaml
HabitatSegmentation:
  postprocess_supervoxel:    # for two_step
    enabled: true
    min_component_size: 30
    connectivity: 1
  postprocess_habitat:        # for both modes
    enabled: true
    min_component_size: 30
    connectivity: 1
```

Removes tiny isolated voxel clusters (<30 voxels). Recommended for clean
visualization.

## Reading the cluster validation curves

After running with `plot_curves: true`, inspect
`<out_dir>/visualizations/.../cluster_validation_scores.png`. See
`references/cluster_validity_guide.md` for how to interpret silhouette,
inertia (elbow), Davies-Bouldin, Calinski-Harabasz.

## Validate output (MANDATORY after run)

```bash
python skills/habit-habitat-analysis/scripts/validate_habitat_output.py <out_dir>
# add --two-step if clustering_mode was two_step
```

Checks every subject has a habitat .nrrd, no degenerate (1-cluster) maps,
and habitats.csv fractions sum to ~1.

## Reference templates

Config index (canonical YAML paths): `skills/CONFIG_SOURCES.md`.

| File | Use |
|---|---|
| `config/habitat/config_getting_habitat.yaml` | beginner one_step |
| `config/habitat/config_getting_habitat.yaml` | publication two_step |
| `config/habitat/config_getting_habitat.yaml` | DCE multi-phase |
| `config/habitat/config_getting_habitat.yaml` | texture clustering |
| `references/voxel_feature_methods.md` | how to choose `voxel_level.method` |
| `references/cluster_validity_guide.md` | reading the validation plots |

Full annotated reference: `config/habitat/config_getting_habitat.yaml`.

## Output files

```
out_dir/
├── <subject_id>/
│   ├── <subject_id>_supervoxel.nrrd         # two_step only
│   ├── <subject_id>_habitats.nrrd            # final habitat map (MAIN OUTPUT)
│   └── <subject_id>_habitats_remapped.nrrd   # two_step only — consistent labels across subjects
├── habitats.csv                               # per-subject habitat fractions
├── supervoxel2habitat_clustering_model.pkl    # two_step train mode only
├── mean_values_of_all_supervoxels_features.csv
└── visualizations/
```

The `*_habitats.nrrd` is the file users open in ITK-SNAP / 3D Slicer.

## Common pitfalls

1. **Mask not found** → check `data_dir/<subject>/masks/`.
2. **Memory error during voxel_radiomics** → reduce `processes`, set `voxelBatch: 512` (or lower), use `useTorchRadiomics: auto` with `cap_processes_to_gpu_pool: true` and align `processes` with GPU count, or set `cap_processes_to_gpu_pool: false` only when you need CPU parallelism on 1 GPU (accept GPU contention), switch to `concat(raw(...))`.
3. **A few subjects fail in large batches but succeed alone** → default `individual_subject_auto_retry_rounds: 2` retries Stage 1 in the same run; reduce `processes` or raise `individual_subject_timeout_sec`; default `individual_subject_parallel_mode: persistent` (use `isolated` if pickle fails); see `errors_habitat.md`.
4. **Cluster number 1 returned** → tumor too homogeneous; add modalities or switch to `voxel_radiomics`/`kinetic`.
5. **kinetic fails** → verify Excel timestamps file IDs match folder names.
6. **Predict mode without pipeline** → require `--pipeline` or `pipeline_path` in config.

For more, see `habit-troubleshoot/references/errors_habitat.md`.

## Next step

After habitat maps are generated, use `habit-feature-extraction` to extract
per-subject quantitative features.
