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

   L4  recipes          two_step / one_step / direct_pooling → StudyResult
   L3  domain           protocols + registries + SubjectPipeline / TablePipeline
   L2  contracts        Subject, Cohort, HabitatModel, FeatureTable, RunManifest
   L1  adapters         DirectoryDataSource, FileImageRef, NnUNetDataSource
   L0  kernels          pure NumPy MSI / ITH / ICC / DeLong / ...
        + spec          HabitatSpec + RunPolicy  (YAML ↔ Python)
        + execution     SerialBackend / ProcessPoolBackend / CheckpointStore
        + compat        sklearn / MONAI / nnU-Net

Mental model
------------

1. **Cohort** = ordered subjects (lazy images).
2. **Subject-level operators** are one-argument callables (no YAML, no pool).
3. **Only** ``HabitatModelFitter.fit`` is cohort-level (shared habitat definition).
4. Publish **``HabitatModel`` + ``SubjectPipeline``** = definition + procedure.
5. Writing to disk is explicit (``HabitatModel.save`` / ``StudyResult.save``).

One-line recipes
----------------

Three habitat designs are available as assembly functions. Each takes a cohort
and a :class:`~habit.spec.specs.HabitatSpec`, runs entirely in memory, and
returns a ``StudyResult``; nothing is written until you ask for it:

.. code-block:: python

   import habit.recipes as recipes

   result = recipes.two_step(cohort, spec)          # supervoxels, then habitats
   result = recipes.direct_pooling(cohort, spec)    # cohort-level voxel habitats
   result = recipes.one_step(cohort, spec)          # per-subject voxel habitats

   result.save("out/study")                         # optional, explicit

   # Apply a published definition to a new cohort
   predicted = recipes.apply_habitat_model(other_cohort, spec, result.habitat_model)

``one_step`` defines habitats inside each subject independently, so it has no
cohort-level definition: ``result.habitat_model`` is ``None`` and the
per-subject definitions are in ``result.subject_models``.

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

Synthetic cohort and three habitat designs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from habit import HabitatSpec, Spec, make_synthetic_cohort
   import habit.recipes as recipes

   cohort = make_synthetic_cohort(
       n_subjects=4,
       modalities=("T1", "T2"),
       shape=(32, 32, 32),
       rng=42,
   )

   # Shared spec fields; only supervoxelizer differs by design
   base = dict(
       voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
       habitat_model_fitter=Spec("kmeans", {"n_habitats": 3, "n_init": 5}),
       habitat_assigner=Spec("nearest_centroid"),
       habitat_features=(Spec("volume"), Spec("msi")),
       random_seed=42,
   )

   two_step = recipes.two_step(
       cohort,
       HabitatSpec(name="two_step", supervoxelizer=Spec("slic", {"n_supervoxels": 30}), **base),
   )
   direct = recipes.direct_pooling(
       cohort,
       HabitatSpec(name="direct_pooling", supervoxelizer=None, **base),
   )
   per_subject = recipes.one_step(
       cohort,
       HabitatSpec(name="one_step", supervoxelizer=None, **base),
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

   from habit import HabitatSpec, Spec, make_synthetic_cohort
   from habit.contracts import HabitatModel
   import habit.recipes as recipes

   train_cohort = make_synthetic_cohort(n_subjects=4, rng=42)
   spec = HabitatSpec(
       name="two_step",
       voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
       supervoxelizer=Spec("slic", {"n_supervoxels": 30}),
       habitat_model_fitter=Spec("kmeans", {"n_habitats": 3}),
       habitat_assigner=Spec("nearest_centroid"),
       habitat_features=(Spec("volume"),),
       random_seed=42,
   )

   train = recipes.two_step(train_cohort, spec)
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
run on an in-memory cohort — use ``LegacyConfigAdapter`` and call a recipe
directly (full details in :doc:`spec`):

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
   result = recipes.two_step(cohort, spec)

The section below shows the same two-step analysis assembled by hand, which is
what a recipe does internally and what you extend when a design is not one of
the three.

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

   # 2) Subject-level operators
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
       [MsiHabitatFeatures(), IthHabitatFeatures(), HabitatVolumeFeatures()],
   )

   # 5) Optional: declare the same design as a HabitatSpec (YAML-isomorphic)
   spec = HabitatSpec(
       name="two_step_demo",
       voxel_feature_extractor=Spec("raw", {"modalities": modalities}),
       supervoxelizer=Spec("slic", {"n_supervoxels": 30}),
       habitat_model_fitter=Spec("kmeans", {"n_habitats": 3}),
       habitat_assigner=Spec("nearest_centroid"),
       habitat_features=(
           Spec("msi"),
           Spec("ith_score"),
           Spec("volume"),
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
   * - Habitat protocols, registries, ``SubjectPipeline``
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
   spec = MLSpec(
       name="demo",
       table_preprocessors=(Spec("zscore"),),
       feature_selectors=(Spec("variance", {"threshold": 0.01}),),
       classifier=Spec("LogisticRegression", {"max_iter": 500}),
       metrics=(Spec("accuracy"), Spec("auc")),
   )
   result = recipes.train_model(table, spec, test_size=0.3, seed=42)
   print(result.train_metrics)   # in-sample readout
   print(result.test_metrics)    # held-out rows

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
