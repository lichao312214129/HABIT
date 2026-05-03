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
| `clustering_mode` | one_step or two_step (see below) |
| (kinetic only) timestamps Excel | required |
| (voxel_radiomics only) PyRadiomics params YAML | required |
| Expected habitat count | typical 3-5 |

## Decision 1 — one_step vs two_step

ALWAYS ask the user this first.

### one_step (recommended for beginners)
- Each tumor independently determines its optimal cluster number (silhouette / elbow)
- No population-level model — habitat labels are per-subject local meaning
- Faster, simpler
- Use template: `config_templates/skill_scaffolds/habitat_one_step_minimal.yaml`

### two_step (publications / predictive modeling)
- **Step 1**: voxel → supervoxel (per subject; e.g. 50 supervoxels per tumor)
- **Step 2**: supervoxel → habitat (population-level model; same habitat label = same biology across all patients)
- Slower, requires train+test cohort
- Generates `supervoxel2habitat_clustering_model.pkl` for `--mode predict`
- Use template: `config_templates/skill_scaffolds/habitat_two_step_minimal.yaml`

If unsure → start with **one_step**.

## Decision 2 — Voxel feature method

`FeatureConstruction.voxel_level.method` defines what voxel features get clustered.
This is the most important biological choice.

| Data type | Method | Template |
|---|---|---|
| Single modality | `raw(<seq>)` | one_step / two_step minimal |
| Multi-modal (most common) | `concat(raw(M1), raw(M2), ...)` | one_step / two_step minimal |
| DCE-MRI | `kinetic(raw(p1), ..., timestamps)` | `config_templates/skill_scaffolds/habitat_kinetic_dce.yaml` |
| Texture-based | `voxel_radiomics(<seq>)` | `config_templates/skill_scaffolds/habitat_voxel_radiomics.yaml` |
| Local entropy | `local_entropy(<seq>)` | (manual) |

Detailed comparison: `references/voxel_feature_methods.md`.

For `kinetic`: user MUST provide a timestamps Excel with subject IDs and
per-phase scan times. Without this, fail fast.

For `voxel_radiomics`: HABIT auto-restricts GLCM to safe features
(Contrast, Correlation, JointEnergy, Idm) when `kernelRadius<=3`. Tell the
user this is automatic.

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
HabitatsSegmention:
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

Config index (scaffold → annotated → standard): `skills/CONFIG_SOURCES.md`.

| File | Use |
|---|---|
| `config_templates/skill_scaffolds/habitat_one_step_minimal.yaml` | beginner one_step |
| `config_templates/skill_scaffolds/habitat_two_step_minimal.yaml` | publication two_step |
| `config_templates/skill_scaffolds/habitat_kinetic_dce.yaml` | DCE multi-phase |
| `config_templates/skill_scaffolds/habitat_voxel_radiomics.yaml` | texture clustering |
| `references/voxel_feature_methods.md` | how to choose `voxel_level.method` |
| `references/cluster_validity_guide.md` | reading the validation plots |

Full annotated reference: `config_templates/config_getting_habitat_annotated.yaml`.

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
2. **Memory error during voxel_radiomics** → reduce `processes`, switch to `concat(raw(...))`.
3. **Cluster number 1 returned** → tumor too homogeneous; add modalities or switch to `voxel_radiomics`/`kinetic`.
4. **kinetic fails** → verify Excel timestamps file IDs match folder names.
5. **Predict mode without pipeline** → require `--pipeline` or `pipeline_path` in config.

For more, see `habit-troubleshoot/references/errors_habitat.md`.

## Next step

After habitat maps are generated, use `habit-feature-extraction` to extract
per-subject quantitative features.
