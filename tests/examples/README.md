# Manual HABIT v1 recipe API examples

Standalone runnable scripts (not collected by pytest). Each script demonstrates
how to embed HABIT in a notebook or pipeline using the **public Python API only**
— no CLI, no YAML config files as the primary path.

## Philosophy

```text
import public API  →  build spec in code  →  load data  →  call recipe  →  save
```

- **Habitat / ML workflows**: build ``HabitatSpec`` or ``MLSpec`` from
  ``habit.spec.specs`` in code.
- **Imaging cohorts**: load from ``demo_data/`` via ``habit.adapters.DirectoryDataSource``.
- **Tabular data**: load CSV into ``habit.contracts.table.FeatureTable`` directly.
- **Legacy workflows** (preprocess, radiomics, DICOM sort): pass a config
  **dict** to the recipe — still no YAML file on disk.
- **Exception**: ``manual_run_from_yaml.py`` is the single YAML-bridge demo for
  v0.1 configs via ``habit.recipes.run_from_yaml``.

Path constants live in ``demo_paths.py`` (constants only, no API wrappers).

## Script style

Linear top-level scripts (no ``main()`` wrappers) so they copy-paste cleanly
into Jupyter cells. English comments, inline type hints.

## Environment

Run from the repository root inside the ``py310`` conda environment:

```powershell
cd F:\work\habit_project
& "E:\conda\mconda\envs\py310\python.exe" tests/examples/manual_habitat_two_step.py
```

Outputs go under ``demo_data/results/examples/<name>/`` so repeated runs do not
overwrite CLI golden artefacts.

## Prerequisites

- **Imaging workflows**: ``demo_data/preprocessed/``
- **Tabular workflows**: ``demo_data/ml_data/``
- **DICOM sort**: ``demo_data/dicom/`` and ``tools/bin/dcm2niix.exe``
- Some scripts depend on prior outputs (noted below)

## Scripts

| Script | API entry | Data source |
|--------|-----------|-------------|
| `manual_habitat_two_step.py` | `recipes.two_step` + `HabitatSpec` | `DirectoryDataSource` |
| `manual_habitat_one_step.py` | `recipes.one_step` + `HabitatSpec` | `DirectoryDataSource` |
| `manual_habitat_direct_pooling.py` | `recipes.direct_pooling` + `HabitatSpec` | `DirectoryDataSource` |
| `manual_habitat_apply_model.py` | `recipes.apply_habitat_model` + `HabitatModel` | `DirectoryDataSource` |
| `manual_preprocess.py` | `recipes.preprocess_images` | config dict in code |
| `manual_icc_analysis.py` | `domain.evaluation.statistics.icc_analysis` | CSV → `FeatureTable` |
| `manual_test_retest.py` | `recipes.test_retest_analysis` | config dict in code |
| `manual_extract_features.py` | `recipes.extract_habitat_features` | config dict in code |
| `manual_traditional_radiomics.py` | `recipes.traditional_radiomics` | config dict in code |
| `manual_voxel_texture_gpu.py` | `VoxelRadiomicsFeatures` (domain) | `DirectoryDataSource` |
| `manual_ml_train.py` | `recipes.train_model` + `MLSpec` | CSV → `FeatureTable` |
| `manual_ml_cross_validate.py` | `recipes.cross_validate` + `MLSpec` | CSV → `FeatureTable` |
| `manual_compare_models.py` | `recipes.compare_models` | config dict in code |
| `manual_run_from_yaml.py` | `recipes.run_from_yaml` | patched YAML (legacy bridge only) |
| `manual_sort_dicom.py` | `recipes.sort_dicom` | config dict in code |

## Dry-run flags

Several slower imaging scripts accept ``--dry-run`` to validate imports and paths
without running the full pipeline:

- `manual_habitat_apply_model.py --dry-run`
- `manual_preprocess.py --dry-run`
- `manual_traditional_radiomics.py --dry-run`
- `manual_voxel_texture_gpu.py --dry-run`
- `manual_sort_dicom.py --dry-run`

## Suggested order for dependent workflows

1. `manual_habitat_two_step.py`
2. `manual_extract_features.py` or `manual_test_retest.py`
3. CLI ML runs (for compare inputs) then `manual_compare_models.py`
