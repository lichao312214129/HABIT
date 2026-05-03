---
name: habit-feature-extraction
description: Extract quantitative features from HABIT habitat maps — traditional radiomics, whole-habitat radiomics, per-habitat radiomics, MSI (Most Significant Intensity), and ITH (Intra-Tumor Heterogeneity) scores. Use when the user has habitat .nrrd maps and wants per-subject feature CSVs ready for ML. Triggers on "提取生境特征", "MSI 特征", "ITH 异质性", "habitat features", "extract radiomics from habitats". Runs `habit extract`.
---

# HABIT Feature Extraction

Extract per-subject quantitative features from previously generated habitat maps.
Output is a set of CSV files ready for ML.

## CLI

```bash
habit extract --config <config_extract_features.yaml>
```

## Required Information

| Field | Notes |
|---|---|
| `raw_img_folder` | preprocessed images + masks |
| `habitats_map_folder` | output of `habit get-habitat` |
| `out_dir` | where CSVs land |
| `feature_types` | what to extract (see table below) |
| `habitat_pattern` | `*_habitats.nrrd` for one_step, `*_habitats_remapped.nrrd` for two_step |
| `params_file_of_*` | required only if extracting radiomics-based features |

## Feature types

| Type | What | Output CSV | Needs PyRadiomics? |
|---|---|---|---|
| `traditional` | Whole-tumor radiomics on **original images** | `raw_image_radiomics.csv` | Yes (`params_file_of_non_habitat`) |
| `non_radiomics` | Volumes, fractions, basic stats | `habitat_basic_features.csv` | No |
| `whole_habitat` | Radiomics on the entire **habitat label map** | `whole_habitat_radiomics.csv` | Yes (`params_file_of_habitat`) |
| `each_habitat` | Radiomics for **each habitat** independently | `each_habitat_radiomics.csv` | Yes |
| `msi` | **Most Significant Intensity** stats per habitat | `msi_features.csv` | No |
| `ith_score` | **Intra-Tumor Heterogeneity** entropy scores | `ith_scores.csv` | No |

### Recommendations

- **Standard publication setup**: `[traditional, non_radiomics, whole_habitat, msi, ith_score]`
  - Use template `config_templates/skill_scaffolds/extract_features_publication.yaml`.
- **Maximum features (slow, high-dim)**: add `each_habitat`. Only do this for cohorts >200.
- **No PyRadiomics installed**: `[non_radiomics, msi, ith_score]`
  - Use template `config_templates/skill_scaffolds/extract_features_msi_ith_only.yaml`.

## PyRadiomics parameter files

When using radiomics-based feature types, two params files are required:

```yaml
params_file_of_non_habitat: ./config/parameter.yaml         # for traditional
params_file_of_habitat: ./config/parameter_habitat.yaml     # for whole/each habitat
```

PyRadiomics starter YAMLs (copy or reference by path):
- `config_templates/skill_scaffolds/pyradiomics_parameter_basic.yaml` — Original only, ~70 features
- `config_templates/skill_scaffolds/pyradiomics_parameter_with_filters.yaml` — + LoG + Wavelet, ~1500 features

Even when not using radiomics types, both keys must exist in the config
(point them at any valid YAML to satisfy parsing).

## Standard config

```yaml
params_file_of_non_habitat: ./config/parameter.yaml
params_file_of_habitat: ./config/parameter_habitat.yaml

raw_img_folder: ./data/preprocessed_images
habitats_map_folder: ./results/habitat_analysis
out_dir: ./results/features

n_processes: 4
habitat_pattern: '*_habitats_remapped.nrrd'

feature_types:
  - traditional
  - non_radiomics
  - whole_habitat
  - msi
  - ith_score

n_habitats:           # leave empty = auto-detect
debug: false
```

## Reference templates

Config index: `skills/CONFIG_SOURCES.md`.

| File | Use |
|---|---|
| `config_templates/skill_scaffolds/extract_features_minimal.yaml` | scaffold |
| `config_templates/skill_scaffolds/extract_features_publication.yaml` | full publication setup |
| `config_templates/skill_scaffolds/extract_features_msi_ith_only.yaml` | quick path, no PyRadiomics |
| `config_templates/skill_scaffolds/pyradiomics_parameter_basic.yaml` | minimal PyRadiomics params |
| `config_templates/skill_scaffolds/pyradiomics_parameter_with_filters.yaml` | full PyRadiomics with LoG+Wavelet |

Full annotated reference: `config_templates/config_extract_features_annotated.yaml`.

## Voxel-based GLCM auto-protection

When using voxel-level radiomics with small kernels (kernelRadius=1-3), GLCM
features can fail on overly homogeneous neighborhoods. HABIT silently
restricts GLCM to: Contrast, Correlation, JointEnergy, Idm. If you see GLCM
warnings, this is why — intentional and safe.

If users want full GLCM:
- Increase `kernelRadius` to 2 or 3
- OR explicitly list GLCM features in their `parameter.yaml`

## Validate output (MANDATORY after run)

```bash
python skills/habit-feature-extraction/scripts/inspect_feature_csv.py <out_dir>/whole_habitat_radiomics.csv \
  --subject-id-col subjID --label-col label
```

Checks for duplicate subject IDs, all-NaN columns, constant features,
non-numeric features, label binarity.

## Output structure

```
out_dir/
├── raw_image_radiomics.csv          # if 'traditional'
├── habitat_basic_features.csv       # if 'non_radiomics'
├── whole_habitat_radiomics.csv      # if 'whole_habitat'
├── each_habitat_radiomics.csv       # if 'each_habitat'
├── msi_features.csv                 # if 'msi'
├── ith_scores.csv                   # if 'ith_score'
└── extract_features.log
```

Each CSV has subject IDs in the first column, features in the rest. Direct
input for `habit model`.

## Common pitfalls

1. **`habitat_pattern` mismatch** → check actual file names. Use `*_habitats.nrrd` for one_step output.
2. **Missing PyRadiomics params** → fail fast; tell user which features need params.
3. **`each_habitat` produces 5×N features for 5 habitats** → can balloon to 1000+.
4. **Subject ID mismatch** between `raw_img_folder` and `habitats_map_folder` → all subjects must exist in both.
5. **MSI/ITH require habitat maps but no PyRadiomics** — fastest path for users without PyRadiomics.

For specific errors, see `habit-troubleshoot/references/errors_extraction.md`.

## Next step

To combine feature CSVs with clinical data into a single ML input file:
```bash
habit merge-csv f1.csv f2.csv clinical.csv -o ml_input.csv --index-col subjID
```

Then proceed to `habit-machine-learning`.
