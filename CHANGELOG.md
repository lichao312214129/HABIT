# Changelog

All notable changes to the HABIT public Python API are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [2.0.0] - 2026-09-03

**Major Release (Breaking Migration):** Complete architectural alignment with capability-based namespaces and scikit-learn style parameter contracts.

### Breaking Changes

- **Physical capability packages**: Migrated all domain implementations to explicit, capability-named subpackages:
  - `habit.voxel_features` (e.g., `RawVoxelFeatures`, `GaborVoxelFeatures`)
  - `habit.supervoxel` (e.g., `SLICSupervoxelizer`, `KMeansSupervoxelizer`)
  - `habit.habitat_features` (e.g., `GraphFeatureExtractor`, `VolumeFeatureExtractor`, `MSIFeatureExtractor`)
  - `habit.image_preprocessing` (e.g., `Resampling`, `ZScoreIntensityNormalisation`, `N4Correction`)
  - `habit.classification` (e.g., `AutoGluonClassifier`)
  - `habit.combiners` (e.g., `ConcatenateCombiner`, `ExpressionCombiner`)
  - `habit.precision` (e.g., `BilinearInterp`, `BSplineInterp`)
- **Removal of legacy packages**:
  - `habit.domain` is completely removed.
  - Legacy `habit.compat.engines` and monolithic batch engines are removed.
- **Scikit-learn style parameter contracts**:
  - Eliminated separate `*Params` Pydantic classes; `__init__` constructors are now the single source of truth for parameters, types, defaults, and validation.
- **Package root exports**:
  - `import habit` exposes only `__version__`. All classes and operators must be imported from their respective canonical capability packages or `habit.api.*`.

### Added

- Lightweight workflow recipes for preprocessing and radiomics (`habit.recipes.preprocess`, `habit.recipes.radiomics`) decoupled from legacy code.
- Dedicated I/O adapters in `habit.adapters` (`preprocessing_io`, `radiomics_io`).
- Strict packaging and architecture contract tests enforcing zero upward dependencies and clean wheel distribution.

**vs 1.1.3:** habitat-first docs and embeddable atomic API; graph-topology
defaults and several public feature / viz APIs. Upgrade with
``pip install -U "habitat-analysis[tables,viz]"``.

### Added

- Contour perturbation gallery (``morphological`` / ``gradient_weighted``
  / ``slice_extent``) with one figure per operator, plus public L0
  exports ``morphological_grow_shrink``, ``boundary_band_mask``,
  ``boundary_weighted_perturbation``, ``slice_extent_perturbation``.
- Graph topology opt-in edge method ``min_distance``: an edge exists when
  the closest-voxel Euclidean distance between two regions is within
  ``distance_threshold`` (voxel-index units). This is not centroid
  distance. Library default remains ``adjacency`` + corner +
  ``adjacency_min_voxels=10``.
- ``habitat_ith_dispersion``: per-habitat ITH
  ``d_i = 1 - (S_{i,max} / n_i) / S_i``. The global ITH score is the
  volume-weighted mean of these values.
- Habitat feature contrast API: ``to_habitat_feature_panel`` /
  ``compare_habitat_features`` melt an ``each_habitat`` table and run
  paired habitat-vs-habitat tests (Cliff's delta or Cohen's d, BH-FDR).
  Publication figures: heatmap, effect-size forest, violin, grouped bars
  (cohort mean or one subject). High-dimensional texture tables use the
  heatmap + top-k effect plot, not one violin per feature.
- SHAP figure family on ``habit.viz``: ``plot_shap_bar``,
  ``plot_shap_violin``, ``plot_shap_heatmap``, ``plot_shap_decision``,
  ``plot_shap_force`` (static matplotlib). YAML ``plot_types`` accepts
  the new names; they stay opt-in (default still includes ``shap``
  beeswarm only). ML gallery documents the full set.
- Clustering feature-chain scalers: MaxAbs, quantile, and L2
  (``feature_preprocessing_method``).
- One-step ``Report`` streaming with ``figure_layout="by_subject"`` and
  graph figure atoms.
- GPU texture-matrix path and native OpenMP C supervoxel matrix extract
  (opt-in / auto).

### Changed

- ``plot_habitat_graph_network_2d`` H1–Hk panels keep the featured
  habitat opaque and fade other habitat fills to alpha 0.2 (background 0
  unchanged). The All-habitats panel stays fully opaque. Panel titles,
  colorbar text, and the edge figlegend use larger publication-readable
  sizes.
- ``plot_ith_summary`` is a single-panel bar chart: a global ``ITH`` bar
  (Okabe–Ito reddish purple) then, when ``dispersion`` is given, a gap
  and H1/H2/... bars (bluish green) on a shared 0–1 axis (ylabel
  ``ITH``). Pass ``dispersion=habitat_ith_dispersion(labels)``. The old
  ``per_habitat`` region-count mapping is rejected; the name remains a
  deprecated alias for ``dispersion`` through v1.x.
- Graph topology defaults now use corner connectivity: components
  ``connectivity='full'`` and edges ``adjacency_connectivity='corner'``
  (8-connected in 2D / 26-connected in 3D). Previously both defaulted to
  ``face`` (4-connected / 6-connected). Pass ``connectivity='face'`` and
  ``adjacency_connectivity='face'`` to keep the old neighborhood.
- Graph topology default edge rule is now voxel adjacency with a minimum
  contact of 10 voxels (``edge_method='adjacency'``,
  ``adjacency_min_voxels=10``). Previously the default was centroid
  proximity (``edge_method='centroid_distance'``, ``distance_threshold=5.0``)
  and adjacency edges used ``adjacency_min_voxels=1``. An edge now exists
  when two regions are adjacent and the shared-boundary voxel count is
  >= 10. Pass ``edge_method='centroid_distance'`` to keep the old rule.
- Graph topology default ``erosion_radius`` is now ``0`` (off). Adjacency
  and the contact >= 10 rule are measured on the habitat labels as drawn.
  Previously the default was one binary-erosion iteration before labeling,
  which could drop thin contacts. Pass ``erosion_radius=1`` (or higher) to
  shrink habitats before edges.
- Docs: habitat analysis is the product spine (concept, atomic embed,
  ``SubjectPipeline``, ``Study``). Image preprocessing and tabular ML
  are supporting. CLI and Python quickstarts no longer show each
  other's twins. New tutorial: parallel backends and fault tolerance.
  Precise screening lives under How-to, not the Tutorial start list.
- Docs: habitat Spec chooser splits voxel / supervoxel stages into
  single-modality leaves and multi-modality combiners; nested trees stay
  on the feature-composition example.
- Default habitat feature extraction now includes the built-in ``graph``
  family with the other light types (``volume``, ``msi``, ``ith_score``,
  ``non_radiomics``). Shipped extract YAMLs and the documented default
  ``feature_types`` / ``HabitatSpec.habitat_features`` light set are
  additive; heavy radiomics stay opt-in. Tests that pin an explicit
  family list are unchanged.
- Docs: real-data gallery for habitat-feature contrast
  (``to_habitat_feature_panel`` / ``compare_habitat_features``) on
  ``demo_data/`` maps with ``each_habitat`` + ``graph``.
- ITH auxiliary columns are opt-in; connected-component cleanup crops
  maps before labeling.

## [1.1.3] - 2026-08-09

**vs 1.1.2:** same demo-config packaging (``copy-demo-config``); this release
fixes habitat **display orientation** in viewers / ``habit view``. Upgrade with
``pip install -U "habitat-analysis[tables,viz]"`` (install docs stay unpinned).

### Fixed

- Habitat display orientation aligned across backends: omit-direction default is
  LPS identity (was RAS); napari no longer flips the axial (z) axis so slice
  indices match file / ITK-SNAP order; default display convention is
  radiological, with ``habit view --convention`` /
  ``view_habitat_napari(..., convention=...)`` for radiological / neurological /
  native.

### Changed

- Docs: beginner Miniconda-to-PyPI install flow; recommend
  ``habitat-analysis[tables,viz]`` for quickstart; note extras for demo
  ``get-habitat``.

## [1.1.2] - 2026-08-08

### Added

- Ship demo YAML templates with the wheel (baked at build time from
  repository ``config/``). Users materialize them without a git clone via
  ``habit copy-demo-config --dest <work_dir>`` or
  ``habit.copy_demo_config(work_dir)``. ``demo_data/`` remains a separate
  download and is never packaged.
- Docs (installation / before_you_start / quickstart) walk through a user
  ``work_dir``: conda terminal → copy demo config → download ``demo_data``
  beside it → run CLI demos.

### Changed

- Single source of truth for demo YAML is repo-root ``config/``. Editable
  installs read that tree live; ``setup.py`` ``build_py`` / ``sdist`` sync
  into ``habit/resources/demo_config/`` for wheels (generated files are
  gitignored).

## [1.1.1] - 2026-08-08

### Added

- ``habit view`` CLI and ``habit.viz.view_habitat_napari`` / overlay helpers:
  open habitat label maps on anatomy in napari (optional ``[view]`` extra),
  with a PNG fallback when napari is unavailable. Replaces the old GUI entry.
- Connected-component habitat/supervoxel cleanup on the v1 path:
  ``HabitatSpec.postprocess_habitat`` / ``postprocess_supervoxel``,
  ``ConnectedComponentPostprocess``, L0 kernel
  ``remove_small_connected_components``, wired through ``SubjectPipeline``
  (in-memory cleanup before features; writers do not re-run it). YAML
  ``enabled: true`` maps into Spec slots (default remains off).
- Image/mask geometry alignment policy on ``HabitatSpec`` /
  ``SubjectPipeline`` (``on_geometry_mismatch``).
- Quickstart napari screenshots refreshed for the demo ``subj001`` view
  path.

### Changed

- Docs and example scripts updated for the view / napari eye-check flow and
  habitat configuration notes for v1 postprocess ownership (B5 closed).

## [1.1.0] - 2026-08-07

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
- **DEPRECATED: ``MLSpec``'s three fixed table chains.**
  ``pre_preprocessing_feature_selectors``, ``table_preprocessors`` and
  ``feature_selectors`` expressed step order through three slots, which
  offered a selector exactly two positions — before all preprocessing or
  after all of it — so an order such as ``zscore → variance → minmax →
  lasso`` had no representation. They are superseded by the single ordered
  ``MLSpec.steps`` list (below) and remain accepted, with a
  ``DeprecationWarning``, for all of v1.x. A spec that declares both layouts
  with different content is rejected rather than resolved by precedence.
  ``MLSpec.to_dict()`` keeps emitting the three deprecated keys (and no
  ``steps`` key) for a spec declared with them, so **every fingerprint
  written before this release is unchanged**; a spec declared with ``steps``
  serialises as ``steps``.
- ``variance`` / ``variance_filter`` and ``correlation`` /
  ``correlation_filter`` are now one implementation each, in
  ``habit.kernels.feature_transforms``, reached through all four registry
  names. Defaults, parameter spellings and numbers are unchanged; the
  behavioural difference the two variance names always had — the filter keeps
  the highest-variance column when nothing clears the threshold, the selector
  keeps nothing — is now the explicit ``keep_at_least_one`` parameter,
  defaulting per name to what that name always did. It is recorded in
  ``spec.params`` only when it deviates from that default, so no existing
  fingerprint moves.

### Added

- ``habit.recipes.search_hyperparameters`` and ``habit.recipes.SearchResult``:
  hyperparameter tuning as a recipe. It drives scikit-learn's
  ``GridSearchCV`` / ``RandomizedSearchCV`` (``strategy="grid"`` /
  ``"random"``) over a ``TablePipeline``, on folds produced by
  ``habit.domain.split.kfold_indices`` so a search partitions the rows exactly
  as ``cross_validate`` does for the same ``n_splits`` and seed, and **writes
  the winning parameters back into the ``MLSpec``** — the tuned definition
  fingerprints, serialises to YAML and re-runs, so tuning does not end the
  provenance chain. Grid keys read ``"<step>__component__<parameter>"``
  (``"model__component__C"``, ``"variance__component__threshold"``); a key
  that cannot be written back into the spec is rejected before the search
  starts. The objective is a registered HABIT metric name whose own
  ``greater_is_better`` sets the direction, defaulting to the spec's first
  declared metric and then to ``auc``. No new dependency: there is
  deliberately no Bayesian/Optuna backend.
- ``cross_validate(..., inner_cv=..., param_grid=...)``: nested
  cross-validation. The hyperparameters are re-tuned inside every outer fold's
  training rows and scored on the untouched validation rows, so the reported
  panel estimates the whole tuning procedure. Each fold's winner is returned
  in the new ``CVResult.fold_best_params``. Passing only one of the two
  arguments is an error, because a grid without ``inner_cv`` would tune on the
  rows it scores. Plain ``cross_validate`` calls are unaffected.
- ``MLSpec.steps``: ONE ordered list of table steps (preprocessors and
  feature selectors, freely interleaved) whose list order is the execution
  order. Step names resolve across both registries; the two vocabularies are
  disjoint, and an ambiguous or unknown name is an explicit error rather than
  a skipped step. See
  ``config/machine_learning/config_machine_learning_steps_v1.yaml`` for a
  runnable native-v1 document.
- ``habit.domain.assembly.build_table_step``: builds one ``MLSpec.steps``
  entry by resolving its name across the table-preprocessor and
  feature-selector registries.
- ``keep_at_least_one`` on the ``variance`` selector and the
  ``variance_filter`` preprocessor, making the historical
  "never empty the feature block" fallback selectable from either name.

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

### Changed — BREAKING (packaging only; the public Python API is unchanged)

- **Seven packages moved out of the required dependency set.** A bare
  ``pip install habitat-analysis`` now installs 11 packages instead of 20:
  ``numpy``, ``scipy``, ``pandas``, ``scikit-learn``, ``SimpleITK``,
  ``pydantic``, ``PyYAML``, ``click``, ``tqdm``, ``joblib``, ``kneed``. Measured
  on CPython 3.10 / Linux, that is **212 MB → 129 MB of wheel downloads**
  (39 wheels → 20) and **931 MB → 635 MB of installed ``site-packages``**,
  i.e. 43 → 23 distributions on disk.

  No public symbol was added, removed or resigned. What changed is what
  ``pip install`` gives you:

  | Package | New extra | Why it could move |
  | --- | --- | --- |
  | ``pyarrow`` | ``tables`` | 83 MB wheel, **zero** direct imports in ``habit/**`` — only the pandas parquet engine |
  | ``scikit-image`` | ``slic`` | 44 MB for one function, ``skimage.segmentation.slic``. The kernel contract is the ``Supervoxelizer`` *protocol*; the default ``kmeans`` / ``gmm`` backends need nothing extra |
  | ``matplotlib`` | ``viz`` | 58 MB incl. its transitive tail. ``habit.viz`` already imported it lazily per function |
  | ``pydicom`` | ``dicom`` | 21 MB used by exactly one module; NIfTI / NRRD input needs only SimpleITK |
  | ``seaborn`` | ``viz`` | Plotting only |
  | ``openpyxl`` | ``tables`` | **Zero** direct imports — the pandas ``.xlsx`` engine |
  | ``chardet`` | *removed* | One encoding-probe call site whose caller already retries a fixed candidate list |

- **Migration is one command**: ``pip install -U "habitat-analysis[full]"``
  reproduces everything a pre-1.1.0 bare install plus ``[all]`` provided.
  ``[all]`` now also aggregates the four new groups, so existing ``[all]``
  users lose nothing. Both meta-extras are written as self-referencing extras
  so they cannot drift from the groups they aggregate.
- ``ml`` and ``analysis`` now pull ``viz`` and ``tables``: their selectors draw
  diagnostic figures and read feature tables from ``.xlsx`` / ``.parquet``.
- ``chardet`` is no longer declared anywhere.
  ``habit.compat.test_retest_mapper.detect_file_encoding`` still consults
  chardet when it happens to be installed, and otherwise returns no guess so
  that the caller's existing candidate list (utf-8 → gbk → gb2312 → gb18030 →
  big5) decides, with a new UTF-8 BOM shortcut in front. A locale-derived guess
  was measured and rejected: on a Chinese Windows install
  ``locale.getpreferredencoding()`` reports ``cp936``, which decodes most UTF-8
  byte sequences without raising and would therefore turn the commonest case
  into silent mojibake instead of a ``UnicodeDecodeError`` the retry loop can
  act on.

### Fixed

- Test-retest config files written in GBK were decoded into mojibake. chardet
  guessed ``Windows-1250`` from the 1000-byte sample, and single-byte codecs
  decode any byte sequence without raising, so the reader's fallback loop never
  got a chance to reach the correct codec. Dropping the statistical guess makes
  the deterministic candidate list authoritative and the result correct;
  ``tests/commands/test_cmd_test_retest_recipes.py`` now covers ascii / utf-8 /
  utf-8-sig / gbk under both a UTF-8 and a ``cp936`` locale.
- ``habit/compat/engines/habitat_analysis/clustering/base_clustering.py``
  imported ``matplotlib.pyplot`` twice and never used it. Removing the dead
  imports frees the whole compat clustering factory from the ``viz`` extra.
- ``vif_selector.py`` imported ``seaborn`` without using it.

### Unchanged on purpose

- ``habitats_results_format`` still defaults to ``parquet``. Missing pyarrow
  raises ``OptionalDependencyError`` listing **both** exits — install
  ``[tables]``, or set ``habitats_results_format: csv`` — and never silently
  writes ``habitats.csv`` where ``habitats.parquet`` was expected. No optional
  dependency anywhere in HABIT degrades silently.

### Added

- ``habit.utils.optional_deps.require(module, *, extra, purpose)``: the generic
  import gate every optional backend now goes through. It raises
  ``OptionalDependencyError`` carrying a copy-pasteable
  ``pip install "habitat-analysis[<extra>]"`` and a one-line statement of what
  the package was needed for, instead of a bare ``ModuleNotFoundError``.
  Companions: ``install_command``, ``optional_dependency_hint``,
  ``require_excel_backend``, ``require_parquet_backend``, and
  ``OPTIONAL_EXTRA_MODULES`` (the machine-readable extras matrix).
  ``require_pyradiomics()`` is kept as the documented specialization — its
  platform-dependent Windows Release-wheel hint cannot be templated from an
  extra name.
- ``habit.utils.dicom_utils.is_pydicom_available()``. The legacy
  ``PYDICOM_AVAILABLE`` module attribute still resolves, but lazily (PEP 562),
  so importing the module no longer imports pydicom or warns when it is absent.
- Two packaging gates in ``tests/test_packaging_contracts.py``: an equality
  assertion on an explicit required-dependency whitelist (so adding a required
  dependency turns the suite red and forces a decision), and a bare-install
  smoke contract that trains a habitat model with all seven optional packages
  hidden behind a ``sys.meta_path`` blocker.
- A ``bare-install`` job in ``.github/workflows/tests.yml`` that installs the
  built distribution with **no** extras (a real resolve, not ``-e .`` and not
  ``--no-deps``) and runs that smoke contract.

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
