---
name: habit-habitat-analysis
description: Generate tumor habitat (sub-region) maps from medical images using voxel-level clustering. Use when the user wants to identify intra-tumor heterogeneity zones, perform supervoxel clustering, generate habitat maps (.nrrd), or run the core HABIT habitat segmentation. Triggers on phrases like "生境分析", "habitat 聚类", "肿瘤亚区", "亚区分析", "supervoxel", "kinetic features", "DCE 分期生境", "habitat segmentation", "intra-tumor heterogeneity".
---

# HABIT Habitat Analysis (Core)

This is the **core** function of HABIT: cluster voxels inside the ROI to discover sub-regions ("habitats") that may correspond to different biological behaviors (necrosis, proliferation, hypoxia, etc.).

## CLI

```bash
# Train mode (default)
habit get-habitat --config <config.yaml>

# Predict mode (apply pre-trained habitat model to new patients)
habit get-habitat --config <config.yaml> --mode predict --pipeline <path/to/pipeline.pkl>

# Debug mode
habit get-habitat --config <config.yaml> --debug
```

## Critical First Decision: one_step vs two_step

**ALWAYS ASK** the user this before generating a config:

### one_step (recommended for beginners)
- Each tumor independently determines its optimal cluster number (silhouette / elbow)
- No population-level model — habitat labels are per-subject local meaning
- Faster, simpler
- Good for: pilot studies, single-cohort exploration
- Use template: `references/config_one_step_minimal.yaml`

### two_step (for publications & predictive modeling)
- **Step 1**: voxel → supervoxel (per subject; e.g. 50 supervoxels per tumor)
- **Step 2**: supervoxel → habitat (population-level model; same habitat label = same biology across all patients)
- Slower, requires train+test cohort
- Generates `supervoxel2habitat_clustering_model.pkl` that can be applied via `--mode predict`
- Use template: `references/config_two_step_minimal.yaml`

If unsure → start with **one_step**.

## Second Decision: Voxel Feature Method

The `FeatureConstruction.voxel_level.method` defines what voxel features get clustered. Choose based on what data the user has:

| Data type | Method | Example |
|---|---|---|
| Single modality (e.g. T2 only) | `raw(T2)` | Simplest baseline |
| Multi-modal (T1+T2+DWI+ADC) | `concat(raw(T1), raw(T2), raw(DWI), raw(ADC))` | Most common |
| DCE-MRI (multi-phase contrast) | `kinetic(raw(pre), raw(LAP), raw(PVP), raw(delay), timestamps)` | Time-intensity curves; needs `timestamps` Excel |
| Texture-based clustering | `voxel_radiomics(T2)` | Slow; needs PyRadiomics params file |
| Local entropy | `local_entropy(T2)` | Texture variation |

For `kinetic`: user MUST provide an Excel file with patient IDs and per-phase scan times. Without this, fail fast and tell them.

For `voxel_radiomics`: HABIT auto-restricts GLCM features to safe ones (Contrast, Correlation, JointEnergy, Idm) when using small kernels. Tell the user this is automatic and intentional.

## Population-level Preprocessing (two_step only)

Before the population clustering step, supervoxel features are typically normalized:

```yaml
FeatureConstruction:
  preprocessing_for_group_level:
    methods:
      - method: binning           # Discretization — most stable for habitat
        n_bins: 10
        bin_strategy: uniform
      # Other options: minmax, zscore, robust, winsorize, log
```

Recommendation: **binning with n_bins=10** is the default for publication work.

## Optional: Postprocess (clean tiny fragments)

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

Removes tiny isolated voxel clusters (<30 voxels) by reassigning them to nearest neighbor. Recommended for clean visualization.

## Reference Templates

Full annotated templates in the project:
- `config_templates/config_getting_habitat_annotated.yaml` — complete reference
- `config_templates/config_habitat_one_step_example.yaml` — simple one_step example

Minimal scaffolds in `references/`:
- `config_one_step_minimal.yaml`
- `config_two_step_minimal.yaml`

## Output Files

```
out_dir/
├── <subject_id>/
│   ├── <subject_id>_supervoxel.nrrd         # supervoxel labels (two_step only)
│   ├── <subject_id>_habitats.nrrd            # final habitat map ← MAIN OUTPUT
│   └── <subject_id>_habitats_remapped.nrrd   # consistent labels across subjects
├── habitats.csv                               # per-subject habitat fractions
├── supervoxel2habitat_clustering_model.pkl    # trained model (two_step train mode)
├── mean_values_of_all_supervoxels_features.csv
└── visualizations/
    ├── cluster_centroids.png
    └── feature_heatmap.png
```

The `*_habitats.nrrd` is the file users open in ITK-SNAP / 3D Slicer to view colored habitat maps.

## Predict Mode Workflow

To apply a trained habitat model to new patients:

1. User must have already run train mode and have:
   - `supervoxel2habitat_clustering_model.pkl`
   - `mean_values_of_all_supervoxels_features.csv`
2. Set `HabitatsSegmention.habitat.mode: testing` in YAML, OR pass `--mode predict --pipeline <pkl>`
3. New `data_dir` should contain new patients with same modality structure

## Common Pitfalls

1. **Mask not found** → check `data_dir/<subject>/masks/` exists and mask file matches modality.
2. **Memory error during voxel_radiomics** → reduce `processes` in config, or switch to `concat(raw(...))`.
3. **Cluster number 1 returned** → tumor is too homogeneous OR features are degenerate. Check raw images aren't constant inside ROI.
4. **kinetic method fails** → verify Excel timestamp file has correct subject IDs matching folder names.
5. **Predict mode without pipeline** → fail. Always require `--pipeline` or `pipeline_path` in config.

## Verification

After running:
- Check `out_dir/<subject>/<subject>_habitats.nrrd` exists for every subject
- Open in ITK-SNAP, overlay on original T2 — habitats should look spatially coherent (not random pepper)
- Check `habitats.csv` — habitat fractions should sum to ~1.0 per subject

## Next Steps

After habitat maps are generated, the next typical step is **feature extraction** → see `habit-feature-extraction` skill.
