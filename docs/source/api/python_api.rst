Python API guide (v2.0)
=======================

This guide is the **canonical** way to use HABIT from Python in v2.0.
``import habit`` exposes only ``__version__``. Import L3 components from
``habit.<capability>`` and workflow helpers from ``habit.recipes`` or
``habit.api.<area>``.

.. note::

   ``import habit; print(habit.__version__)`` → ``2.0.0``

   Imaging cohorts live at :class:`~habit.contracts.Cohort`
   (``from habit.contracts import Cohort``). Do not import components from
   the package root.

Architecture in one diagram
---------------------------

.. code-block:: text

   L4  recipes          Study.fit / fit_predict / predict (+ habitat factories) → StudyResult
   L4  report           Report(persist, retain, figures, writer) — run-scoped, not a Spec
   L3  capability pkgs  voxel_features / supervoxel / habitat_model / pipeline / ...
   L2  contracts        Subject, Cohort, HabitatModel, FeatureTable, RunManifest
   L1  adapters         DirectoryDataSource, FileImageRef
   L0  kernels          pure NumPy MSI / ITH / ICC / DeLong / ...
        + spec          HabitatSpec.stages + Stage + RunPolicy  (YAML ↔ Python)
        + execution     SerialBackend / ProcessPoolBackend / CheckpointStore

Mental model
------------

1. **Subject-level operators** are one-argument callables
   (``voxel(subject)``, ``pipe(subject)``). No YAML, no directory layout.
   This is the embedding surface — :doc:`../examples/habitat_atomic_ops`.
2. **Cohort** = ordered subjects (lazy images). Optional until you fit
   a shared :class:`~habit.contracts.HabitatModel` or ``map`` a pipeline.
3. **``HabitatSpec.stages``** = ordered named stages (source of truth);
   strategy is inferred (partition+pool → two_step; pool only →
   direct_pooling; neither → one_step).
4. **``pool``** is the only subject↔cohort watershed; post-pool feature
   preprocess is first-class.
5. Publish **``HabitatModel`` + ``SubjectPipeline``** = definition +
   procedure.
6. Writing to disk is explicit (``HabitatModel.save`` /
   ``StudyResult.save``). For a one-step cohort that must not hold every
   subject's volumes, pass ``report=Report(...)`` so each subject is
   persisted (and optionally drawn) as it completes.

Beginners: copy :doc:`../tutorial/quickstart_python` (a ``Study``
recipe). Integrators: atoms first, then this page. Concept:
:doc:`../tutorial/habitat_analysis`. Parallel / fault tolerance:
:doc:`../tutorial/execution`.

Primary entry: ``Study`` (cohort recipe)
----------------------------------------

Declare stages, then call :meth:`~habit.recipes.Study.fit_predict`. Nothing is
written until you ask for it:

.. code-block:: python

   from habit.spec import HabitatSpec, Spec, Stage
   import habit.recipes as recipes

   spec = HabitatSpec(
       name="demo",
       stages=(
           Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})),
           Stage("partition", Spec("slic", {"n_supervoxels": 30})),
           Stage("pool", Spec("pool")),
           Stage("fit", Spec("kmeans", {"n_habitats": 3, "n_init": 5})),
           Stage("assign", Spec("nearest_centroid")),
           Stage("quantify", Spec("volume")),
       ),
       random_seed=42,
   )
   result = recipes.Study(spec=spec).fit_predict(cohort)
   result.save("out/study")                         # optional, explicit

   # Apply a published definition to a new cohort
   predicted = recipes.Study.from_model(result.habitat_model, spec).predict(other_cohort)

Recommended stage labels (documentation only, not keywords):
``extract_voxel_features``, ``preprocess1`` / ``preprocess2`` / …,
``partition``, ``extract_supervoxel_features``, ``pool``, ``fit``,
``assign``, ``quantify``. Do not teach ``role=`` as the primary API.

Named-field ``HabitatSpec`` plus the factories
:func:`~habit.recipes.two_step_habitat` /
:func:`~habit.recipes.one_step_habitat` /
:func:`~habit.recipes.direct_pooling_habitat` remain convenience builders that
return a :class:`~habit.recipes.Study` with a declared ``design`` (see
:doc:`spec`).

One-step (no ``pool``) has no cohort-level definition:
``result.habitat_model`` is ``None``; per-subject definitions live in
``result.subject_models``. Stream maps / figures with
:class:`~habit.report.Report` — see :doc:`../examples/one_step_habitat`.

.. code-block:: python

   from habit.kernels import HabitatGraphFeatureOptions
   from habit.recipes import Study
   from habit.report import (
       ClusterValidation,
       GraphNetwork2D,
       GraphSlice,
       ITH,
       MSI,
       Overlay,
       Report,
       VolumeFractions,
   )
   from habit.adapters import DirectoryResultWriter

   graph = HabitatGraphFeatureOptions(edge_method="min_distance", block_size=8)
   result = Study(spec=spec, design="one_step").fit_predict(
       cohort,
       report=Report(
           persist=("habitat_map", "subject_model"),
           retain="tables",
           figures=(
               Overlay(modality="T1"),
               VolumeFractions(),
               MSI(),
               ITH(),
               ClusterValidation(),
               GraphSlice(options=graph),
               GraphNetwork2D(options=graph),
           ),
           writer=DirectoryResultWriter("out/study"),
           figure_layout="by_subject",
       ),
   )
   result.save("out/study")   # tables + manifest; maps already on disk

To open habitat labels on anatomy right after the recipe (napari screenshots
included), see **View the habitat maps** in :doc:`../tutorial/quickstart_python`.

Common workflows
----------------

The snippets below use :func:`~habit.datasets.make_synthetic_cohort` so they
run without a download. For the official imaging pack (and to see the
folder tree your own data must match) call :func:`~habit.datasets.fetch_demo`, then
:func:`~habit.contracts.cohort_from_directory` (see :doc:`../examples/data_from_arrays`).

Environment fingerprint
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from habit.api.utils import show_versions

   print(show_versions())  # HABIT, Python, NumPy, …

Synthetic cohort and three strategy shapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from habit.datasets import make_synthetic_cohort
   from habit.spec import HabitatSpec, Spec, Stage
   import habit.recipes as recipes

   cohort = make_synthetic_cohort(
       n_subjects=4,
       modalities=("T1", "T2"),
       shape=(32, 32, 32),
       rng=42,
   )

   fit = Stage("fit", Spec("kmeans", {"n_habitats": 3, "n_init": 5}))
   assign = Stage("assign", Spec("nearest_centroid"))
   quantify = (
       Stage("quantify", Spec("volume")),
       Stage("quantify2", Spec("msi")),
       Stage("quantify3", Spec("ith_score")),
       Stage("quantify4", Spec("non_radiomics")),
       Stage("quantify5", Spec("graph")),
       # Heavy PyRadiomics families (opt-in; require pyradiomics):
       # Stage("quantify6", Spec("traditional")),
       # Stage("quantify7", Spec("whole_habitat")),
       # Stage("quantify8", Spec("each_habitat")),
   )
   extract = Stage(
       "extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})
   )

   two_step = recipes.Study(
       spec=HabitatSpec(
           name="two_step",
           stages=(
               extract,
               Stage("partition", Spec("slic", {"n_supervoxels": 30})),
               Stage("pool", Spec("pool")),
               fit,
               assign,
               *quantify,
           ),
           random_seed=42,
       )
   ).fit_predict(cohort)
   direct = recipes.Study(
       spec=HabitatSpec(
           name="direct_pooling",
           stages=(extract, Stage("pool", Spec("pool")), fit, assign, *quantify),
           random_seed=42,
       )
   ).fit_predict(cohort)
   per_subject = recipes.Study(
       spec=HabitatSpec(
           name="one_step",
           stages=(extract, fit, assign, *quantify),
           random_seed=42,
       )
   ).fit_predict(cohort)

   print(two_step.habitat_model.summary())
   print(per_subject.subject_models.keys())  # one HabitatModel per subject

Persist results (requires SimpleITK for NRRD output)::

   two_step.save("out/two_step_study")
   # <subject>_habitats.nrrd, habitat_model.habitatmodel,
   # habitat_features.csv, run_manifest.json

Apply a saved definition to a new cohort
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train (or load) a cohort-level model, then project it onto held-out subjects.
The upstream ``HabitatSpec`` must match the stages used during fitting.
Prediction inherits the model's persisted ``postprocess_habitat``; an
explicit conflicting declaration raises ``HABITAPIError``.

.. code-block:: python

   from pathlib import Path

   from habit.contracts import HabitatModel
   from habit.datasets import make_synthetic_cohort
   from habit.spec import HabitatSpec, Spec, Stage
   import habit.recipes as recipes

   train_cohort = make_synthetic_cohort(n_subjects=4, rng=42)
   spec = HabitatSpec(
       name="two_step",
       stages=(
           Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})),
           Stage("partition", Spec("slic", {"n_supervoxels": 30})),
           Stage("pool", Spec("pool")),
           Stage("fit", Spec("kmeans", {"n_habitats": 3})),
           Stage("assign", Spec("nearest_centroid")),
           Stage("quantify", Spec("volume")),
           Stage("quantify2", Spec("msi")),
           Stage("quantify3", Spec("ith_score")),
           Stage("quantify4", Spec("non_radiomics")),
           Stage("quantify5", Spec("graph")),
       ),
       random_seed=42,
   )

   train = recipes.Study(spec=spec).fit_predict(train_cohort)
   out = train.save("out/train_study")

   held_out = make_synthetic_cohort(n_subjects=2, rng=99)
   predicted = recipes.Study.from_model(train.habitat_model, spec).predict(held_out)

   # Equivalent after reload from disk
   reloaded = HabitatModel.load(out / "habitat_model.habitatmodel")
   predicted = recipes.Study.from_model(reloaded, spec).predict(held_out)

Run a YAML config from Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _common-workflows-yaml:

:func:`habit.recipes.run_from_yaml` is the programmatic twin of the CLI: it
reads a YAML file, detects its version, and dispatches to the same recipes
the command line uses. Documents with ``version: '1.0'`` are read directly
(habitat train/predict and ML train/cv workflows); older YAML is translated
through :class:`~habit.spec.legacy.LegacyConfigAdapter`:

.. code-block:: python

   import habit.recipes as recipes

   result = recipes.run_from_yaml(
       "config/habitat/config_habitat_two_step.yaml",
       workflow="habitat",   # optional; guessed from the path when omitted
       save=True,            # write outputs like the CLI would (default False)
   )

To translate an older YAML document by hand — for example to swap the data
source or run on an in-memory cohort — use ``LegacyConfigAdapter`` and call
:meth:`~habit.recipes.Study.fit_predict` (full details in :doc:`spec`):

.. code-block:: python

   from pathlib import Path

   import yaml
   from habit.datasets import make_synthetic_cohort
   from habit.spec import HabitatSpec, LegacyConfigAdapter
   import habit.recipes as recipes

   payload = yaml.safe_load(
       Path("config/habitat/config_habitat_two_step.yaml").read_text(
           encoding="utf-8"
       )
   )
   translation = LegacyConfigAdapter().translate(payload, "habitat")
   spec = HabitatSpec.from_dict(translation.document["spec"])
   cohort = make_synthetic_cohort(n_subjects=4, rng=42)  # or DirectoryDataSource(...).load()
   result = recipes.Study(spec=spec).fit_predict(cohort)

The section below shows the same two-step analysis assembled by hand, which is
what the stage executor runs under the hood and what you extend for custom
designs.

Canonical end-to-end example
----------------------------

.. code-block:: python

   from pathlib import Path

   from habit.datasets import make_synthetic_cohort
   from habit.spec import HabitatSpec, Spec
   from habit.habitat_features import HabitatVolumeFeatures, IthHabitatFeatures, MsiHabitatFeatures, NonRadiomicsHabitatFeatures
   from habit.habitat_model import KMeansHabitatModelFitter
   from habit.voxel_features import RawVoxelFeatures
   from habit.supervoxel import SlicSupervoxelizer
   from habit.pipeline import SubjectPipeline
   from habit.execution import SerialBackend

   # 1) In-memory cohort (no files on disk)
   cohort = make_synthetic_cohort(
       n_subjects=4,
       modalities=["T1", "T2"],
       shape=(32, 32, 32),
       rng=42,
   )

   # 2) Subject-level operators (Seedable; default seed 0)
   modalities = ["T1", "T2"]
   voxel = RawVoxelFeatures(modalities=modalities)
   svx = SlicSupervoxelizer(n_supervoxels=30)
   svx.set_random_state(42)

   # 3) The only cohort-level step: fit a population HabitatModel
   units = [svx(voxel(s)) for s in cohort]
   fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=5)
   fitter.set_random_state(42)
   model = fitter.fit(units, cohort=cohort)
   print(model.summary())
   model.save(Path("out/demo.habitatmodel"))

   # 4) Definition + procedure
   pipe = SubjectPipeline(voxel, svx, model.assigner())
   maps = cohort.map(pipe, backend=SerialBackend())
   table = pipe.extract_features(
       cohort[0],
       [
           HabitatVolumeFeatures(),
           MsiHabitatFeatures(),
           IthHabitatFeatures(),
           NonRadiomicsHabitatFeatures(),
           # Heavy PyRadiomics families (opt-in; require pyradiomics):
           # TraditionalRadiomicsHabitatFeatures(),
           # WholeHabitatRadiomicsFeatures(),
           # EachHabitatRadiomicsFeatures(),
       ],
   )

   # 5) Optional: declare the same design as stages (YAML-isomorphic)
   from habit.spec import Stage

   spec = HabitatSpec(
       name="two_step_demo",
       stages=(
           Stage("extract_voxel_features", Spec("raw", {"modalities": modalities})),
           Stage("partition", Spec("slic", {"n_supervoxels": 30})),
           Stage("pool", Spec("pool")),
           Stage("fit", Spec("kmeans", {"n_habitats": 3})),
           Stage("assign", Spec("nearest_centroid")),
           Stage("quantify", Spec("volume")),
           Stage("quantify2", Spec("msi")),
           Stage("quantify3", Spec("ith_score")),
           Stage("quantify4", Spec("non_radiomics")),
           Stage("quantify5", Spec("graph")),
       ),
       random_seed=42,
   )
   print(spec.describe_methods(style="radiology"))

Where to go next
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Topic
     - Page
   * - ``Subject`` / ``Cohort`` / ``HabitatModel`` / ``FeatureTable`` / ``RunManifest``
     - :doc:`data_model`
   * - Directory / file image adapters
     - :doc:`adapters`
   * - Embed ``op(subject)`` / ``SubjectPipeline``
     - :doc:`../examples/habitat_atomic_ops`
   * - Habitat protocols, registries, ``SubjectPipeline``,
       ``HabitatComponents``
     - :doc:`domain_habitat`
   * - Parallel backends, timeout, resume
     - :doc:`../tutorial/execution` · :doc:`execution`
   * - Table ML: preprocessors, selectors, classifiers, ``TablePipeline``
     - :doc:`domain_table`
   * - ``HabitatSpec`` / ``RunPolicy`` / migrate YAML
     - :doc:`spec`
   * - Streaming persist + per-subject figures (``Report``)
     - :doc:`../examples/one_step_habitat`
   * - Pure numeric kernels
     - :doc:`kernels`
   * - sklearn / MONAI / nnU-Net
     - :doc:`compat`
   * - Habitat ``Spec`` chooser (names + parameters)
     - :doc:`../how_to/habitat_components`
   * - ``list_plugins`` / schemas
     - :doc:`plugins`
   * - Custom ``ComponentRegistry``
     - :doc:`registry`

Tabular machine learning
------------------------

The ML recipes are v1-native: :func:`~habit.recipes.train_model`,
:func:`~habit.recipes.cross_validate`, and
:func:`~habit.recipes.predict_model` take a :class:`~habit.contracts.FeatureTable`
plus an :class:`~habit.spec.specs.MLSpec` and run a
:class:`~habit.pipeline.TablePipeline` — fitted preprocessing and selection
state travels inside the saved pipeline, so prediction never refits on new
data. Tabular building blocks are documented in :doc:`domain_table`.

.. code-block:: python

   from habit.datasets import make_synthetic_feature_table
   from habit.spec import MLSpec, Spec
   import habit.recipes as recipes

   table = make_synthetic_feature_table(n_rows=60, n_features=8, rng=42)
   # ``steps`` is ONE ordered list and its order is the execution order.
   # Variance goes first, on the raw table: after z-score every feature has
   # variance ~1, so a variance step placed later is uninformative.
   # Supervised selectors (ANOVA, LASSO, ...) normally follow the scaling.
   spec = MLSpec(
       name="demo",
       steps=(
           Spec("variance", {"threshold": 0.01}),
           Spec("zscore"),
       ),
       classifier=Spec("LogisticRegression", {"max_iter": 500}),
       metrics=(Spec("accuracy"), Spec("auc")),
   )
   result = recipes.train_model(table, spec, test_size=0.3, seed=42)
   print(result.train_metrics)   # in-sample readout
   print(result.test_metrics)    # held-out rows

``make_synthetic_feature_table`` is built for golden tests (one strong
``signal`` column) and will often print AUC 1.0. For publication-style
figures see :doc:`../examples/visualization`.

Steps interleave freely, which the three predecessor fields could not
express. ``zscore`` → ``variance`` → ``minmax`` → ``lasso`` is a plain list:

.. code-block:: python

   spec = MLSpec(
       name="interleaved",
       steps=(
           Spec("zscore"),
           Spec("variance", {"threshold": 0.01}),
           Spec("minmax"),
           Spec("lasso", {"cv": 5}),
       ),
       classifier=Spec("LogisticRegression", {"max_iter": 500}),
   )

The fields ``pre_preprocessing_feature_selectors``, ``table_preprocessors``
and ``feature_selectors`` are deprecated aliases kept for all of v1.x. They
are folded into ``steps`` in that documented order and raise a
``DeprecationWarning``; a spec declaring both layouts with different content
is rejected rather than resolved by precedence.

Hyperparameter search and nested cross-validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~habit.recipes.search_hyperparameters` tunes a spec with scikit-learn's
``GridSearchCV`` / ``RandomizedSearchCV`` and writes the winners **back into
the** :class:`~habit.spec.specs.MLSpec`. That is the point of the recipe: a
tuned model is a *definition*, so it keeps a fingerprint, serialises to YAML
and can be re-run by a reader — a search that returned only a fitted object
would end the provenance chain where the parameters were chosen.

Grid keys address one HABIT component parameter through the pipeline's step
names, ``"<step>__component__<parameter>"``. The terminal model's step is
``"model"``; every other step is named after its registered spec name:

.. code-block:: python

   tuned = recipes.search_hyperparameters(
       table,
       spec,
       {
           "model__component__C": [0.01, 0.1, 1.0, 10.0],
           "variance__component__threshold": [0.0, 0.01],
       },
       n_splits=5,
       seed=42,
       objective="auc",          # a registered HABIT metric, not a scorer string
   )
   print(tuned.best_params, tuned.best_score)
   print(tuned.spec.classifier.params["C"])   # written back into the spec
   fitted = tuned.model.pipeline                # refitted with the tuned spec

The folds come from :func:`habit.evaluation.split.kfold_indices`, so a search
partitions the rows exactly as :func:`~habit.recipes.cross_validate` does for
the same ``n_splits`` and seed, and every candidate is fitted on training rows
only — preprocessing statistics and feature selection included, because they
are pipeline steps that get cloned per fold. ``strategy="random"`` switches to
a sampled search with an ``n_iter`` budget. The objective's own
``greater_is_better`` sets the direction, so a "lower is better" metric is
minimised without the caller negating anything; omitted, the objective is the
first metric of ``spec.metrics``, and ``auc`` for a spec with no panel.

Tuning once on all the rows and cross-validating afterwards reports an
optimistically biased number, because the validation rows took part in the
selection. Passing ``inner_cv`` together with ``param_grid`` makes
:func:`~habit.recipes.cross_validate` nested instead: each outer fold re-tunes
on its own training rows and is scored on rows neither the tuning nor the fit
has seen.

.. code-block:: python

   nested = recipes.cross_validate(
       table,
       spec,
       n_splits=5,
       inner_cv=3,
       param_grid={"model__component__C": [0.01, 1.0, 100.0]},
       seed=42,
   )
   print(nested.mean_metrics)        # estimate of the whole tuning procedure
   print(nested.fold_best_params)    # one winner per outer fold

Report ``fold_best_params`` alongside the panel: parameters that change from
fold to fold say the search is fitting noise. The two arguments are required
together — a grid without ``inner_cv`` has nowhere to tune except the outer
validation rows, and ``inner_cv`` without a grid searches over nothing. There
is deliberately no Bayesian/Optuna backend, which would be a new hard
dependency.

CLI users keep ``habit get-habitat -c ...``; that path translates YAML through
``LegacyConfigAdapter`` / ``HabitatSpec`` into the same domain core, and
:func:`~habit.recipes.run_from_yaml` exposes the identical path to Python
callers (see :ref:`the YAML section <common-workflows-yaml>` above).

Feature extraction (config-driven)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``habit.recipes.extract_habitat_features`` and
``habit.recipes.traditional_radiomics`` exist so ``habit extract`` and
``habit radiomics`` can route through L4 recipes. They are **not** the
recommended Python path: no domain-native cohort assembly, no
``SubjectPipeline`` / ``TablePipeline``, and no ``StudyResult`` contract.
For habitat features in library code, prefer
:class:`~habit.pipeline.SubjectPipeline` ``.extract_features(...)`` with the
``habitat_feature_extractor`` registry (:doc:`domain_habitat`).
