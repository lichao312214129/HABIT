Python API guide (v1.0)
=======================

This guide is the **canonical** way to use HABIT from Python in v1.0.
Everything below is implemented on the ``v1.0.0`` branch.

.. note::

   ``import habit; print(habit.__version__)`` → ``1.0.0``

   Since v1.0.0 the top-level ``habit.Cohort`` **is** the imaging cohort
   (``habit.contracts.subject.Cohort``). The v0.1 clinical directory wrapper
   was renamed :class:`habit.ClinicalCohort`. You can equally import the data
   model from its canonical home: ``from habit.contracts import Cohort``.

Architecture in one diagram
---------------------------

.. code-block:: text

   L4  recipes          fit_habitat (+ thin two_step / one_step / direct_pooling) → StudyResult
   L3  domain           stages executor + protocols + SubjectPipeline / TablePipeline
   L2  contracts        Subject, Cohort, HabitatModel, FeatureTable, RunManifest
   L1  adapters         DirectoryDataSource, FileImageRef, NnUNetDataSource
   L0  kernels          pure NumPy MSI / ITH / ICC / DeLong / ...
        + spec          HabitatSpec.stages + Stage + RunPolicy  (YAML ↔ Python)
        + execution     SerialBackend / ProcessPoolBackend / CheckpointStore
        + compat        sklearn / MONAI / nnU-Net

Mental model
------------

1. **Cohort** = ordered subjects (lazy images).
2. **``HabitatSpec.stages``** = ordered named stages (source of truth); strategy
   is inferred (partition+pool → two_step; pool only → direct_pooling;
   neither → one_step).
3. **Subject-level operators** are one-argument callables (no YAML required).
4. **``pool``** is the only subject↔cohort watershed; post-pool feature
   preprocess is first-class.
5. Publish **``HabitatModel`` + ``SubjectPipeline``** = definition + procedure.
6. Writing to disk is explicit (``HabitatModel.save`` / ``StudyResult.save``).

Primary recipe: ``fit_habitat``
-------------------------------

Declare stages, then call :func:`~habit.recipes.fit_habitat`. Nothing is
written until you ask for it:

.. code-block:: python

   from habit import HabitatSpec, Spec, Stage
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
   result = recipes.fit_habitat(cohort, spec)
   result.save("out/study")                         # optional, explicit

   # Apply a published definition to a new cohort
   predicted = recipes.apply_habitat_model(other_cohort, spec, result.habitat_model)

Recommended stage labels (documentation only, not keywords):
``extract_voxel_features``, ``preprocess1`` / ``preprocess2`` / …,
``partition``, ``extract_supervoxel_features``, ``pool``, ``fit``,
``assign``, ``quantify``. Do not teach ``role=`` as the primary API.

Named-field ``HabitatSpec`` plus :func:`~habit.recipes.two_step` /
:func:`~habit.recipes.one_step` / :func:`~habit.recipes.direct_pooling`
remain thin sugar/aliases that validate and call ``fit_habitat`` (see
:doc:`spec`).

One-step (no ``pool``) has no cohort-level definition:
``result.habitat_model`` is ``None``; per-subject definitions live in
``result.subject_models``.

To open habitat labels on anatomy right after the recipe (napari screenshots
included), see **View the habitat maps** in :doc:`../tutorial/quickstart_python`.

Common workflows
----------------

The snippets below use :func:`~habit.datasets.make_synthetic_cohort` so they
run without ``demo_data`` or any on-disk layout. For real studies, swap in
:class:`~habit.adapters.DirectoryDataSource` or
:func:`~habit.contracts.cohort_from_directory` (see :doc:`adapters` and
:doc:`data_model`).

Environment fingerprint
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from habit import show_versions

   print(show_versions())  # HABIT, Python, NumPy, …

Synthetic cohort and three strategy shapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from habit import HabitatSpec, Spec, Stage, make_synthetic_cohort
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
       # Heavy PyRadiomics families (opt-in; require pyradiomics):
       # Stage("quantify5", Spec("traditional")),
       # Stage("quantify6", Spec("whole_habitat")),
       # Stage("quantify7", Spec("each_habitat")),
   )
   extract = Stage(
       "extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})
   )

   two_step = recipes.fit_habitat(
       cohort,
       HabitatSpec(
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
       ),
   )
   direct = recipes.fit_habitat(
       cohort,
       HabitatSpec(
           name="direct_pooling",
           stages=(extract, Stage("pool", Spec("pool")), fit, assign, *quantify),
           random_seed=42,
       ),
   )
   per_subject = recipes.fit_habitat(
       cohort,
       HabitatSpec(
           name="one_step",
           stages=(extract, fit, assign, *quantify),
           random_seed=42,
       ),
   )

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

.. code-block:: python

   from pathlib import Path

   from habit import HabitatSpec, Spec, Stage, make_synthetic_cohort
   from habit.contracts import HabitatModel
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
       ),
       random_seed=42,
   )

   train = recipes.fit_habitat(train_cohort, spec)
   out = train.save("out/train_study")

   held_out = make_synthetic_cohort(n_subjects=2, rng=99)
   predicted = recipes.apply_habitat_model(
       held_out, spec, train.habitat_model
   )

   # Equivalent after reload from disk
   reloaded = HabitatModel.load(out / "habitat_model.habitatmodel")
   predicted = recipes.apply_habitat_model(held_out, spec, reloaded)

Run a YAML config from Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _common-workflows-yaml:

:func:`habit.recipes.run_from_yaml` is the programmatic twin of the CLI: it
reads a YAML file, detects its version, and dispatches to the same recipes
the command line uses. **v0.1** documents are translated through
:class:`~habit.spec.legacy.LegacyConfigAdapter`; **v1** documents are read
directly (habitat train/predict and ML train/cv workflows):

.. code-block:: python

   import habit.recipes as recipes

   result = recipes.run_from_yaml(
       "config/habitat/config_habitat_two_step.yaml",
       workflow="habitat",   # optional; guessed from the path when omitted
       save=True,            # write outputs like the CLI would (default False)
   )

To translate a v0.1 document by hand — for example to swap the data source or
run on an in-memory cohort — use ``LegacyConfigAdapter`` and call
:func:`~habit.recipes.fit_habitat` (full details in :doc:`spec`):

.. code-block:: python

   from pathlib import Path

   import yaml
   from habit import LegacyConfigAdapter, make_synthetic_cohort
   from habit.spec import HabitatSpec
   import habit.recipes as recipes

   payload = yaml.safe_load(
       Path("config/habitat/config_habitat_two_step.yaml").read_text(
           encoding="utf-8"
       )
   )
   translation = LegacyConfigAdapter().translate(payload, "habitat")
   spec = HabitatSpec.from_dict(translation.document["spec"])
   cohort = make_synthetic_cohort(n_subjects=4, rng=42)  # or DirectoryDataSource(...).load()
   result = recipes.fit_habitat(cohort, spec)

The section below shows the same two-step analysis assembled by hand, which is
what the stage executor runs under the hood and what you extend for custom
designs.

Canonical end-to-end example
----------------------------

.. code-block:: python

   from pathlib import Path

   from habit import HabitatSpec, Spec, make_synthetic_cohort
   from habit.domain import (
       HabitatVolumeFeatures,
       IthHabitatFeatures,
       KMeansHabitatModelFitter,
       MsiHabitatFeatures,
       NonRadiomicsHabitatFeatures,
       RawVoxelFeatures,
       SlicSupervoxelizer,
       SubjectPipeline,
   )
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
   from habit import Stage

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
   * - Habitat protocols, registries, ``SubjectPipeline``,
       ``HabitatComponents``
     - :doc:`domain_habitat`
   * - Table ML: preprocessors, selectors, classifiers, ``TablePipeline``
     - :doc:`domain_table`
   * - ``HabitatSpec`` / ``RunPolicy`` / migrate YAML
     - :doc:`spec`
   * - Parallel execution and checkpoints
     - :doc:`execution`
   * - Pure numeric kernels
     - :doc:`kernels`
   * - sklearn / MONAI / nnU-Net
     - :doc:`compat`
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
:class:`~habit.domain.TablePipeline` — fitted preprocessing and selection
state travels inside the saved pipeline, so prediction never refits on new
data. Tabular building blocks are documented in :doc:`domain_table`.

.. code-block:: python

   from habit import MLSpec, Spec, make_synthetic_feature_table
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

The folds come from :func:`habit.domain.split.kfold_indices`, so a search
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

The older configuration-object entry points (``run_ml``, ``run_kfold``,
``run_model_comparison`` from ``habit.api.machine_learning``) remain
available for YAML-parity workflows; the CLI commands ``habit model`` /
``habit cv`` / ``habit compare`` use them.

CLI users keep ``habit get-habitat -c ...``; that path translates YAML through
``LegacyConfigAdapter`` / ``HabitatSpec`` into the same domain core, and
:func:`~habit.recipes.run_from_yaml` exposes the identical path to Python
callers (see :ref:`the YAML section <common-workflows-yaml>` above).

Feature extraction (config-driven)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``habit.recipes.extract_habitat_features`` and
``habit.recipes.traditional_radiomics`` are thin wrappers around the v0.1
engine (``habit.compat.engines`` via ``habit.api.habitat``). They exist so
``habit extract`` and ``habit radiomics`` can route through L4 recipes without
new direct ``habit.compat.engines`` imports in the command layer. They are **not** the
recommended v1 Python path: no domain-native cohort assembly, no
``SubjectPipeline`` / ``TablePipeline``, and no ``StudyResult`` contract.
For habitat features in library code, prefer
:class:`~habit.domain.SubjectPipeline` ``.extract_features(...)`` with the
``habitat_feature_extractor`` registry (:doc:`domain_habitat`).
