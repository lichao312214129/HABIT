# Choosing a PyRadiomics `parameter.yaml`

`habit radiomics` (and the radiomics components of `habit extract`) take a
PyRadiomics parameter file. The file has three sections:

```yaml
imageType:    # which image filters to apply
featureClass: # which feature families to compute
setting:      # discretization, masks, normalization
```

## Quick decision

| Use case | Filters | Feature classes | Approx feature count |
|---|---|---|---|
| Pilot / small cohort | Original only | firstorder + shape + glcm | ~70 |
| Standard radiomics | Original + LoG | + glrlm + glszm | ~250 |
| Comprehensive | Original + LoG + Wavelet | All 7 classes | ~1500 |
| Voxel-radiomics (clustering) | Original | firstorder + glcm | ~25 |

## Templates (repo-wide)

See `skills/CONFIG_SOURCES.md`. PyRadiomics starter files:

- `config_templates/skill_scaffolds/pyradiomics_parameter_example.yaml` — generic full set
- `config_templates/skill_scaffolds/pyradiomics_parameter_basic.yaml` — minimal (~70 features)
- `config_templates/skill_scaffolds/pyradiomics_parameter_with_filters.yaml` — LoG + Wavelet (~1500 features)

## Critical settings

### `binWidth` vs `binCount`
- `binWidth: 25` — fixed bin width; intensity range divided into bins of size 25
- `binCount: 32` — fixed number of bins regardless of intensity range

For **CT** (HU): `binWidth: 25` is standard.
For **MRI**: depends on z-score normalization. After z-score, intensities are
roughly ±3, so `binCount: 32` is safer than `binWidth`.

### `resampledPixelSpacing`
- If preprocessing already resampled (recommended), set this to `null` or
  comment it out — re-resampling here changes shape features unnecessarily.
- If you skipped preprocessing resample, set it here, e.g. `[1, 1, 1]`.

### `normalize`
- `false` if you already z-scored in preprocessing
- `true` if your input is raw intensities

### `label`
- Default `1`. Change if your mask uses a different integer for the ROI.

### `force2D`
- Almost always `false` (HABIT works in 3D). Only enable for thin-slice
  data where you really want per-slice features.

## Multi-modal extraction trick

`habit radiomics` does NOT support per-modality parameter files. If you need
different binning for CT vs MRI in the same study, run `habit radiomics`
twice with separate configs and merge the CSVs with `habit merge-csv`.

## What if I add `Wavelet` and the run takes 1 hour per subject?

That's expected. Wavelet has 8 decompositions, each gets the full feature
set computed. Mitigations:
- Drop one of glcm / glrlm / glszm
- Use only Original + LoG (no Wavelet)
- Increase `n_processes` in the radiomics config

## Validating the params file

Quick sanity check: pyradiomics will tell you immediately on bad YAML:

```bash
python -c "import yaml; print(yaml.safe_load(open('parameter.yaml')))"
```

Then preview features with one subject:
```bash
python -c "
from radiomics import featureextractor
ex = featureextractor.RadiomicsFeatureExtractor('parameter.yaml')
print(list(ex.enabledFeatures.keys()))
print(list(ex.settings.keys()))
"
```
