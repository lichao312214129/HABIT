# Habitat Analysis Errors

## Symptom: `subject X has no mask` / `mask not found`

**Cause**: HABIT looked for `<data_dir>/<subject>/masks/<modality>/<modality>.nii.gz`
and didn't find it.

**Fix**: Verify the mask file exists at the expected path. Common mistakes:
- Mask is named `mask.nii.gz` instead of matching modality name
- Mask is under `masks/` directly without the `<modality>/` subfolder
- Subject directory uses different IDs in `images/` vs `masks/`

Run the layout checker:
```bash
python skills/habit-quickstart/scripts/check_data_layout.py <data_dir>
```

## Symptom: `kinetic() failed: missing timestamps for subject X`

**Cause**: Subject ID in folder name doesn't match any row in the timestamps
Excel.

**Fix**: Open the Excel and confirm:
- The first column header matches `subjID` (or whatever HABIT expects)
- Subject IDs are written exactly as folder names (`sub001` not `Sub001`)
- No extra whitespace in IDs

## Symptom: `voxel_radiomics` crashes with `Failed to initialize firstorder feature class`

**Cause**: PyRadiomics params file path is wrong or YAML is malformed.

**Fix**:
1. Check the `params_file:` path inside `voxel_level.params:` exists
2. Validate YAML: `python -c "import yaml; yaml.safe_load(open('params.yaml'))"`
3. Make sure at least one feature class is enabled in the params file

## Symptom: cluster number is always 1 (degenerate)

**Cause**: ROI is too homogeneous — voxels are all very similar.

**Diagnosis**:
1. Open the original image + mask in ITK-SNAP
2. Use the histogram tool to inspect intensity inside the ROI
3. If the histogram has only one narrow peak, the tissue is homogeneous

**Fixes**:
- Add more modalities to `concat(...)` to provide more discriminating signal
- Switch to `voxel_radiomics(<seq>)` for texture-based clustering
- For DCE, use `kinetic(...)` which adds the temporal dimension
- If all options fail, accept that this tumor is not amenable to habitat
  analysis (biological reality, not a software bug)

## Symptom: `MemoryError` during voxel_radiomics

**Cause**: Voxel-radiomics is heavy; default `processes` is too high.

**Fix**:
```yaml
processes: 1   # or 2 for very small ROIs
```

Also check `kernelRadius` in the params YAML — `kernelRadius: 3` extracts
features from 7x7x7 neighborhoods which is 343 voxels per voxel.

## Symptom: predict mode fails with `pipeline file not found`

**Cause**: `--mode predict` needs an existing trained pipeline.

**Fix**: Either:
- Pass `--pipeline <path_to_pkl>` on the CLI, OR
- Set in YAML: `HabitatsSegmention.habitat.mode: testing` AND
  ensure `<out_dir>/supervoxel2habitat_clustering_model.pkl` exists from a
  prior train run.

## Symptom: habitat maps look like random pepper noise

**Cause**: Population-level clustering converged on noise rather than
biological signal. Usually means:
1. `n_clusters` (supervoxel count) too high relative to ROI size
2. Features weren't normalized at population level
3. Different scanners introduced unmodeled variance

**Fix**:
1. Reduce `supervoxel.n_clusters` to 30-50
2. Enable `preprocessing_for_group_level` with `binning` (recommended)
3. Enable `postprocess_habitat` to clean tiny fragments
4. Confirm preprocessing applied z-score / histogram standardization

## Symptom: `*_habitats_remapped.nrrd` missing for some subjects

**Cause**: Two-step mode but the remapping step crashed for those subjects.

**Fix**: Check the log:
```bash
grep -A 5 "subject X" <out_dir>/habitat_analysis.log
```

Usually the cause is missing supervoxel file. Re-run for those subjects only:
```bash
# Move other subjects out temporarily, then:
habit get-habitat --config <config>
# Move them back
```

## Symptom: `silhouette` always picks `min_clusters`

**Cause**: With `min_clusters: 2`, silhouette tends to pick 2 because
2-cluster solutions inherently have higher silhouette scores.

**Fix**: Either:
- Set `fixed_n_clusters: 4` based on biological expectation
- Switch to `inertia` and use the elbow method visually
- Read `cluster_validity_guide.md` in `habit-habitat-analysis/references/`
