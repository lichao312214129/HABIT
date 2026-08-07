# Changelog

All notable changes to the HABIT public Python API are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- **BREAKING (behaviour): ``TablePipeline`` now inherits
  ``sklearn.pipeline.Pipeline``.** The constructor signature is unchanged
  (``TablePipeline(steps=[...HABIT components...], model=...)``) and every
  HABIT verb — ``fit(table)``, ``transform(table)``, ``predict(table)``,
  ``predict_proba(table)``, ``predict_survival_function``, ``evaluate``,
  ``set_random_state``, ``spec``, ``save`` / ``load`` — behaves exactly as
  before. Two attributes changed meaning:
  - ``pipeline.steps`` is now scikit-learn's ``List[Tuple[str, estimator]]``.
    It is not overridden, because ``sklearn.pipeline.Pipeline`` reads *and
    writes* it directly.
  - **``pipeline.components`` is the new home of the HABIT transformation
    components** (the tuple ``pipeline.steps`` used to return).
    ``pipeline.model`` / ``pipeline.classifier`` are unchanged.

  In exchange, a ``TablePipeline`` works directly with ``sklearn.base.clone``,
  ``get_params`` / ``set_params``, ``cross_val_score``, ``GridSearchCV`` and
  ``RandomizedSearchCV``, and nested parameter grids such as
  ``{"model__component__C": [0.1, 1, 10]}`` address a HABIT component's own
  parameters. Rationale and the full contract are recorded in
  ``developer/api_upgrade/08_naming_decisions.md`` §8.
- ``TableClassifierEstimator.classes_`` now reports the endpoint's own dtype
  (e.g. ``[0, 1]``) instead of the probability frame's string column labels,
  matching ``predict()`` and letting label-aware scikit-learn scorers work.
  The frame column labels moved to the new ``proba_columns_`` attribute.
- ``.habitpipeline`` files are now written at ``format_version = 2`` (the
  added field is the ``FrameToTable`` head's column schema). Version 1 files
  written by earlier releases still load and predict identically.

### Added

- ``habit.domain.sklearn_interop``: the table-level scikit-learn interop
  adapters, lifted out of the frozen ``habit.compat`` layer into L3.
  ``habit.compat.sklearn`` keeps deprecated aliases of
  ``TableTransformerEstimator`` / ``TableClassifierEstimator`` /
  ``as_transformer`` / ``as_classifier`` for all of v1.x.
- ``habit.domain.sklearn_interop.FrameToTable``: rebuilds a ``FeatureTable``
  from a plain frame plus a static column schema, which is what lets
  scikit-learn's cross-validation drivers slice ``X`` by row. Every
  ``TablePipeline`` carries one as its head step (named ``"frame_to_table"``);
  it passes ``FeatureTable`` input straight through unchanged.
- ``as_regressor`` / ``as_survival_model`` / ``as_outcome_model`` factories for
  the two terminal-model families the v1.0 interop surface did not cover.
- Every built-in tabular component (classifiers, regressors, survival models,
  preprocessors, feature selectors) now implements scikit-learn's
  ``get_params`` / ``set_params`` / ``clone`` protocol, sourced from the same
  single mapping ``spec.params`` is built from, so a searched value cannot
  disagree with the recorded fingerprint.

## [1.0.4] - 2026-08-07

### Changed

- **PyRadiomics is installed separately**: removed
  ``python -m habit.install_radiomics`` and the Windows auto-install path in
  ``require_pyradiomics()``. HABIT extras no longer declare ``pyradiomics``
  (the empty ``radiomics`` extra remains as a documented alias). On Windows,
  install the matching prebuilt wheel from GitHub Release ``v1.0.2``; on
  macOS / Linux use PyPI (``pyradiomics>=3.0.1,<3.2``) or conda-forge. See
  the Installation tutorial for the wheel URL table.

### Added


- Stable top-level namespace: ``import habit`` exposes pipeline runners, config
  classes, and utilities via lazy loading (see ``habit.api.registry``).
- v1.0 layered public API registered at top level: all ``habit.kernels``
  numerical functions, the ``habit.compat`` interop entry points
  (``as_estimator`` / ``as_transformer`` / ``as_classifier``, MONAI dict
  converters, ``NnUNetDataSource``), and the ``habit.viz`` figure functions.
- ``habit.utils.deprecation``: ``HabitDeprecationWarning``,
  ``HabitPendingDeprecationWarning``, and the ``deprecated`` decorator.
- ``habit.kernels.cluster_selection``, ``habit.kernels.voxel_texture`` and
  ``habit.kernels.radiomics.voxel_maps``: the cluster-count, local-entropy and
  voxel-feature-map numerics, now shared by both engines.
- Domain voxel feature families reaching v0.1 parity: ``voxel_radiomics``,
  ``kinetic``, ``local_entropy`` and ``concat``.
- ``gap`` / ``inertia`` criteria for the K-means habitat fitter and
  ``silhouette`` / ``calinski_harabasz`` / ``davies_bouldin`` / ``gap`` for the
  GMM fitter, each recording an auditable selection report.
- ``habit.recipes``: assembly functions for the three habitat designs —
  ``two_step``, ``one_step``, ``direct_pooling`` — plus ``apply_habitat_model``
  for projecting a published ``HabitatModel`` onto a new cohort. Each takes a
  cohort and a ``HabitatSpec``, runs in memory, and returns a ``StudyResult``;
  no file path, YAML or run mode is involved.
- ``habit.adapters.DirectoryResultWriter``: the write-side counterpart of
  ``DirectoryDataSource``, implementing the ``ResultWriter`` protocol with the
  v0.1 directory layout (habitat NRRDs with geometry, ``.habitatmodel``,
  feature CSV, run manifest).
- ``ResultWriter.write_manifest``, so a run manifest can be persisted through
  the same protocol as every other artefact.
- Golden coverage for predict mode (``habitat_two_step_predict``) and for the
  habitat feature families (``habitat_features``), including MSI, ITH and
  basic feature tables.
- ``habit.datasets``: ``make_synthetic_cohort`` and ``make_synthetic_feature_table``
  for deterministic in-memory test cohorts and feature tables (no files, no
  network).
- Synthetic fast golden gate (``tests/golden/fast/``, ``baseline/fast/``)
  covering two_step, one_step, direct_pooling, predict, habitat_features, and
  ml_kfold with fixed ``n_habitats=3``; CI runs under ``-m "not slow"`` in
  about 20 seconds.
- ``scripts/make_fast_golden_baseline.py`` to regenerate fast baselines.
- ``tests/recipes/test_recipes_fast_parity.py`` synthetic recipe parity tests.

### Changed

- **BREAKING — automatic habitat-count selection**: the ``elbow``
  cluster-validation method is now an alias of ``kneedle``. Up to v0.1.x,
  ``elbow`` selected the cluster count with a second-difference (discrete
  curvature) criterion while ``kneedle`` used the normalised maximum-deviation
  (Kneedle) criterion, and the two could disagree on the same inertia curve.
  Both keys, and ``inertia``, now resolve to the single implementation in
  ``habit.kernels.cluster_selection.knee_index``, shared by the v0.1 engines
  and the v1.0 domain fitters so they can no longer drift apart.

  - Configurations stay valid: ``selection_method: elbow`` and
    ``habitat_cluster_selection_method: [elbow]`` are still accepted.
  - Results may change: re-running a study with automatic selection can yield
    a different habitat count than v0.1.x did. On the bundled demo cohort the
    one-step and direct-pooling configurations move from 3 to 4 habitats; the
    two-step configuration is unchanged at 4.
  - To reproduce a published habitat count, set ``fixed_n_clusters``
    (per-subject: ``supervoxel.one_step_settings.fixed_n_clusters``;
    cohort-level: ``habitat.fixed_n_clusters``).
  - ``get_optimization_direction()`` now reports ``"knee"`` for ``elbow``,
    ``kneedle`` and ``inertia``.
- **BREAKING — internal module moves** (deep imports only; the public API is
  unaffected): the supervoxel radiomics numerics moved out of the v0.1 engine
  into the kernel layer.
  ``habit.compat.engines.habitat_analysis.clustering_features.supervoxel_cext`` →
  ``habit.kernels.radiomics.cext``; ``...clustering_features.torchradiomics``
  → ``habit.kernels.radiomics.torchradiomics``;
  ``...clustering_features.batched_supervoxel_radiomics`` →
  ``habit.kernels.radiomics.supervoxel_batch``;
  ``...clustering_features.supervoxel_radiomics_settings`` →
  ``habit.kernels.radiomics.settings``.

- Top-level ``habit.Cohort`` now binds to the v1.0 imaging cohort contract
  (``habit.contracts.subject.Cohort``). The v0.1 clinical cohort was renamed
  ``habit.ClinicalCohort``; the old name remains importable from
  ``habit.api.clinical`` as a deprecated alias (removal planned for v1.2.0).
- ``import habit`` no longer imports sklearn/pandas/scipy at package import
  time; ``habit.exceptions.NotFittedError`` is constructed lazily on first
  access while remaining a subclass of ``sklearn.exceptions.NotFittedError``.
- Removed the empty leftover package ``habit/cli_commands/`` (source modules
  were deleted earlier; only stale ``__pycache__`` directories remained).
- Public runners: ``run_preprocess``, ``run_dicom_sort``, ``run_habitat_analysis``,
  ``run_feature_extraction``, ``run_radiomics``, ``run_ml``, ``run_kfold``,
  ``run_model_comparison``, ``run_icc_analysis``.
- Public config classes: ``PreprocessingConfig``, ``DicomSortConfig``,
  ``HabitatAnalysisConfig``, ``FeatureExtractionConfig``, ``RadiomicsConfig``,
  ``MLConfig``, ``ModelComparisonConfig``, ``TestRetestConfig``, ``ICCConfig``.
- Helpers: ``apply_habitat_cli_overrides``, ``apply_ml_mode_override``,
  ``setup_logger``, ``is_available``.
- ``habit.__version__`` sourced from ``habit._version``.
- API contract tests under ``tests/api/`` (golden MSI/ITH, pipeline smoke,
  CLI–API parity) and CI workflow ``.github/workflows/tests.yml``.
- ``StudyResult`` moved from ``habit.contracts`` to ``habit.recipes``. It is
  the recipe layer's return type; no lower layer produces or consumes one.
  ``StudyResult.habitat_model`` is now optional (``one_step`` defines habitats
  per subject, exposed in the new ``subject_models``).
- ``LegacyConfigAdapter`` no longer translates
  ``preprocessing_for_group_level`` into a cohort preprocessing chain for
  ``clustering_mode: one_step``; v0.1 silently ignored that block, and the
  adapter now reproduces that behaviour explicitly, with a warning.
- Habitat label maps written by the v1 writer use a single integer type
  (``int32``) instead of v0.1's per-code-path mix of ``uint16`` and
  ``int32``; label values are unchanged.
- **BREAKING — v0.1 pickle pipelines removed**: legacy
  ``habitat_pipeline.pkl`` raw pickles and ``*_final_pipeline.pkl`` ML
  artefacts are no longer loaded or remapped. Predict/train must use v1
  ``habitat_model.habitatmodel`` (habitat) or ``.habitpipeline`` (ML).
  Attempts to load v0.1 pickles fail with a migration message pointing at
  ``apply_habitat_model`` / re-train. The ``pickle_compat`` module and
  ``habitat_pipeline.pkl`` output shim are removed; train writes
  ``habitat_model.habitatmodel`` and ``run_manifest.json`` only. Slow golden
  baseline ``tests/golden/baseline/`` regenerated for v1 artefact layout.

### Notes

- v0.1 YAML/CLI engines now live under ``habit.compat.engines.*``; the
  ``habit.core`` package has been removed. Prefer the top-level ``habit``
  imports and ``habit.recipes`` for new integrations.
- Deep paths such as ``habit.compat.engines.preprocessing.run.run_preprocess_from_config``
  continue to work for YAML-driven workflows via ``habit.compat.legacy_core``.

## [1.0.2] - 2026-08-06

### Changed

- **Python support widened to 3.10–3.14** (``requires-python``
  ``>=3.10,<3.15``). Verified on Windows x64: full install-check suite passes
  on 3.10, 3.13 and 3.14. Windows wheels are now published for every
  supported CPython (cp310–cp314); the sdist builds on all of them.
- **numpy 2.x supported**: the base dependency is now ``numpy>=1.26,<3``.
  Verified: full check suite, fast golden tests (no numeric drift) and all
  packaging/API/architecture gates pass under numpy 2.2.6, and the published
  C extension (built against numpy 2 headers) also runs under numpy 1.26.
  Installing HABIT no longer forces a numpy 2.x environment to downgrade.
- ``pyarrow`` on CPython 3.14 requires ``>=22`` (first release with cp314
  wheels); 3.10–3.13 keep the previous ``>=15,<22`` range via environment
  markers.
- ``radiomics`` extra widened to ``pyradiomics>=3.0.1,<3.2``. PyPI ships no
  usable PyRadiomics Windows binaries (the 3.1.0 sdist is broken; 3.0.1 has
  no Windows wheels), so HABIT publishes self-built PyRadiomics 3.1.0 Windows
  wheels (cp310–cp314, numpy-2 compiled, verified on numpy 1.26 and 2.x) as
  GitHub Release assets. PyRadiomics remains an optional extra, never a hard
  dependency.
