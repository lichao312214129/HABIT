# Cloud Coverage Audit Report

Synthetic-data smoke-test matrix + coverage audit for HABIT v1.0.0, produced
on branch `cloud/coverage-smoke`. The real `demo_data` dataset is gitignored,
so every pipeline run below executes on the deterministic synthetic tree in
`tests/fixtures/synthetic_data.py` (seed 42 everywhere).

## Test inventory

| Suite | Tests | Result |
|---|---|---|
| `tests/cloud_coverage` (new) | 54 | **51 passed, 3 skipped** in ~14 s |
| Audit run (see command below) | 780 | **745 passed, 34 skipped, 1 pre-existing failure** in ~37 s |

New modules: `test_habitat_matrix.py` (9), `test_machine_learning.py` (4),
`test_model_comparison.py` (3), `test_preprocessing.py` (4),
`test_radiomics.py` (3), `test_auxiliary.py` (2), `test_dicom.py` (4),
`test_feature_extraction.py` (2), `test_cli_smoke.py` (23).

### Skips (all with documented reasons)

| Test | Reason |
|---|---|
| `test_preprocessing.py::test_preprocess_elastix_registration` | elastix binary unavailable on this Linux image (only `tools/bin/elastix.exe` ships; elastix is on the do-not-install list) |
| `test_preprocessing.py::test_preprocess_dcm2nii` | dcm2niix binary unavailable on this Linux image (only `tools/bin/dcm2niix.exe` ships; dcm2niix is on the do-not-install list) |
| `test_dicom.py::test_sort_dicom_cli` | dcm2niix binary unavailable on this Linux image (only `tools/bin/dcm2niix.exe` ships; dcm2niix is on the do-not-install list) |

The remaining 31 skips in the audit run come from pre-existing suites
(`tests/api`, `tests/recipes`, `tests/spec`) and are almost all gated on the
unavailable `demo_data` tree or optional backends (autogluon/torch), per
their own markers.

### Pre-existing failure (present on the base branch, NOT a product bug)

`tests/spec/test_legacy.py::test_bundled_habitat_config_translates_and_validates[config_habitat_two_step_v1]`
fails on the unmodified base branch: the test parametrizes over every
`config/habitat/*.yaml` and asserts each is a v0 document, but
`config_habitat_two_step_v1.yaml` is intentionally a v1 document.
`detect_yaml_version` correctly returns `v1`; the test's discovery should
exclude v1 files. This is a test-side bug in a pre-existing file, which
this branch is not allowed to modify; it is excluded from the audit totals
below via the `-k` conjunction filter shown in the audit command.

## Audit command

```bash
pytest tests/cloud_coverage tests/api tests/recipes tests/contracts tests/spec tests/commands \
  --cov=habit.api --cov=habit.recipes --cov=habit.contracts --cov=habit.spec \
  --cov-report=term-missing \
  -k "not (test_bundled_habitat_config_translates_and_validates and config_habitat_two_step_v1)"
```

Verified result: **745 passed, 34 skipped, 1 deselected, 0 failed**; total
coverage 83%. (`--deselect` with the bracketed parametrized node id does not
match under pytest 9.1.1, and a bare `-k "not config_habitat_two_step_v1"`
would also exclude four unrelated tests whose ids mention that config name;
the conjunction above excludes exactly the one pre-existing failure
documented below.)

## Per-module coverage (audited packages)

Total: **3884 statements, 648 missed, 83% covered.**

| Module | Stmts | Miss | Cover |
|---|---|---|---|
| habit/api/__init__.py | 14 | 0 | 100% |
| habit/api/analysis.py | 35 | 0 | 100% |
| habit/api/clinical.py | 155 | 17 | 89% |
| habit/api/contracts.py | 46 | 5 | 89% |
| habit/api/dicom_sort.py | 20 | 0 | 100% |
| habit/api/estimators.py | 250 | 45 | 82% |
| habit/api/exceptions.py | 2 | 0 | 100% |
| habit/api/habitat.py | 57 | 1 | 98% |
| habit/api/image.py | 188 | 58 | 69% |
| habit/api/machine_learning.py | 44 | 0 | 100% |
| habit/api/plugins.py | 158 | 28 | 82% |
| habit/api/preprocessing.py | 19 | 0 | 100% |
| habit/api/provenance.py | 92 | 8 | 91% |
| habit/api/radiomics.py | 109 | 41 | 62% |
| habit/api/registry.py | 3 | 0 | 100% |
| habit/api/utils.py | 18 | 1 | 94% |
| habit/contracts/__init__.py | 11 | 0 | 100% |
| habit/contracts/geometry.py | 25 | 1 | 96% |
| habit/contracts/habitat.py | 176 | 12 | 93% |
| habit/contracts/image.py | 52 | 4 | 92% |
| habit/contracts/manifest.py | 134 | 19 | 86% |
| habit/contracts/ops.py | 39 | 1 | 97% |
| habit/contracts/outcome.py | 102 | 27 | 74% |
| habit/contracts/provenance.py | 32 | 2 | 94% |
| habit/contracts/subject.py | 132 | 11 | 92% |
| habit/contracts/table.py | 45 | 1 | 98% |
| habit/recipes/__init__.py | 14 | 0 | 100% |
| habit/recipes/auxiliary.py | 87 | 45 | 48% |
| habit/recipes/comparison.py | 11 | 0 | 100% |
| habit/recipes/features.py | 10 | 0 | 100% |
| habit/recipes/habitat.py | 155 | 3 | 98% |
| habit/recipes/icc.py | 6 | 0 | 100% |
| habit/recipes/modeling.py | 132 | 2 | 98% |
| habit/recipes/preprocess.py | 7 | 0 | 100% |
| habit/recipes/result.py | 107 | 6 | 94% |
| habit/recipes/sort_dicom.py | 7 | 0 | 100% |
| habit/recipes/study.py | 54 | 8 | 85% |
| habit/recipes/yaml_runner.py | 508 | 260 | 49% |
| habit/spec/__init__.py | 6 | 0 | 100% |
| habit/spec/legacy.py | 517 | 28 | 95% |
| habit/spec/policy.py | 63 | 3 | 95% |
| habit/spec/specs.py | 203 | 9 | 96% |
| habit/spec/yaml_io.py | 32 | 2 | 94% |
| **TOTAL** | **3884** | **648** | **83%** |

Notable gaps: `habit/recipes/yaml_runner.py` (49% -- the v1 runners for
preprocess/radiomics/model/cv/compare/icc/extract/dicom documents are
only partially exercised through `run_from_yaml`; the habitat path is fully
covered), `habit/recipes/auxiliary.py` (48% -- the `dice`/`dicom_info` recipe
functions are not what the CLI commands wire to, see below),
`habit/api/radiomics.py` (62%) and `habit/api/image.py` (69%).

## Public-symbol touch audit

Method: every symbol in `habit/_public_api.py::_PUBLIC_API_MODULES` (204
symbols) is counted as TOUCHED when an executable line inside its source
range ran during the audit run but not during a bare
`import habit.api, habit.recipes, habit.contracts, habit.spec` baseline
(this filters import-time `def`/`class`/dataclass-field execution).

Result: **90 touched, 18 untouched (in the audited packages), 92 defined
outside the audited packages, 4 unresolvable by static inspection.**

### Untouched public symbols in audited packages

| Symbol | Notes |
|---|---|
| `GeometryPolicy`, `GeometryReport`, `ImageMaskPair`, `read_image`, `read_mask` (`habit.api.image`) | v1 image-IO facade; pipeline runs go through SimpleITK helpers instead |
| `CohortFingerprint`, `CohortOperator`, `DataSource`, `ExecutionBackend`, `ImageRef`, `ResultWriter`, `SubjectOperator` (`habit.contracts`) | Protocol/ABC surface for third-party implementations; only the concrete adapters run in tests |
| `cohort_from_directory` (`habit.contracts`) | v1 convenience constructor; CLI paths build cohorts through `DirectoryDataSource` |
| `CVResult`, `ModelResult`, `PredictionResult` (`habit.recipes`) | Result dataclasses ARE produced by the runs; flagged untouched because their source range executes entirely at class-definition time -- static-inspection limitation, not a real gap |
| `dice` (`habit.recipes.auxiliary`) | CLI `habit dice` wires to `habit.utils.dice_calculator.run_dice_calculation`, not the recipe |
| `dicom_info` (`habit.recipes.auxiliary`) | CLI `habit dicom-info` wires to the command's own implementation, not the recipe |

### Unresolvable by static inspection (defined in `habit.kernels`)

`KNEE`, `MAXIMIZE`, `MINIMIZE`, `SCORE_DIRECTIONS` -- module-level
constants/enum members whose source span `inspect` cannot resolve; the
kernels package is outside the audited coverage scope. `habit.kernels`
exhaustive-selection behaviour is separately covered by
`tests/kernels/test_selection_methods.py`.

### Defined outside the audited packages (92)

These live in `habit.domain`, `habit.adapters`, `habit.compat`, `habit.datasets`,
`habit.exceptions`, `habit.kernels`, `habit.monai`/`habit.compat.monai`,
`habit.viz` and the `habit.api.*` config-schema modules; line coverage for
them was not measured by the audit command above. They are exercised heavily
through the same suites (e.g. `DirectoryDataSource`, `TablePipeline`,
`KMeansSupervoxelizer`, `make_synthetic_feature_table` all run in
`tests/api`, `tests/recipes` and `tests/cloud_coverage`).

## Known issues (owned elsewhere, not fixed on this branch)

Per project-owner directive, these three known bugs are owned by another
agent on branch `cloud/behavior-tests` and were intentionally not touched
(this suite never triggers them):

1. `get-habitat` traceback when `data_dir` does not exist;
2. `get-habitat` traceback when configured modality missing from data tree;
3. `check-config` not rejecting unknown component names in v1 documents.

## Genuine product bugs found by this suite

None. Every pipeline ran to completion on the synthetic tree; the two
surprises encountered were documented product behaviour, not bugs:

- `habit cv` / `habit model` (v1) fit ONE classifier per run -- the v1
  `MLSpec` models a single classifier; extra v0 `models` entries warn and
  are preserved under `legacy` (`cmd_ml.py:557`). The matrix therefore runs
  LogisticRegression and RandomForest as two CV configs.
- The v1 `habit model` recipe persists `model.habitpipeline` + JSON metrics
  and keeps per-row probabilities in memory (`ModelResult`); it does not
  write the v0.1 `all_prediction_results.csv`. The model-comparison test
  serialises probabilities through the v1 API instead.

Environment additions required beyond `requirements-ci.txt` (all pure-Python
optional deps that shipped features import lazily): `seaborn` + `chardet`
(tests/api, tests/recipes imports), `scikit-image` + `pyarrow` (supervoxel
features / parquet output), `kneed` (elbow cluster selection), `pingouin`
(ICC analysis path).
