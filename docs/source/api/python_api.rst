Python API guide (v1.0)
=======================

This guide is the **canonical** way to use HABIT from Python in v1.0.
Everything below is implemented on the ``v1.0.0`` branch.

.. note::

   ``import habit; print(habit.__version__)`` → ``1.0.0``

   Prefer ``from habit.contracts import ...`` and ``from habit.domain import ...``
   for the data model and operators. Top-level ``habit.Cohort`` is a *legacy*
   clinical directory wrapper and is **not** the contracts cohort used here.

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

Run a legacy YAML from Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _common-workflows-yaml:

There is no ``recipes.run_from_yaml``. Translate v0.1 YAML with
:class:`~habit.spec.legacy.LegacyConfigAdapter`, then call a recipe (full
details in :doc:`spec`):

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

Not yet on the v1 stack
-----------------------

Machine-learning train / cross-validation (``run_ml``, ``run_kfold``,
``run_model_comparison``) still route through ``habit.core.*``. Use the CLI
(``habit model``, ``habit cv``, ``habit compare``) or the compat layer
(:doc:`compat`) until those workflows move to ``habit.recipes`` /
:class:`~habit.domain.TablePipeline`. Tabular building blocks are documented
in :doc:`domain_table`.

``habit.recipes`` has no ``run_from_yaml``: reading configuration files is the
CLI's job, not the library's. To run a v0.1 YAML from Python, translate it
first with ``LegacyConfigAdapter`` and then call a recipe (see
:ref:`common-workflows-yaml` above).

CLI users keep ``habit get-habitat -c ...``; that path translates YAML through
``LegacyConfigAdapter`` / ``HabitatSpec`` into the same domain core.

Feature extraction (compat-only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``habit.recipes.extract_habitat_features`` and
``habit.recipes.traditional_radiomics`` are thin wrappers around the v0.1
engine (``habit.core`` via ``habit.api.habitat``). They exist so
``habit extract`` and ``habit radiomics`` can route through L4 recipes without
new direct ``habit.core`` imports in the command layer. They are **not** the
recommended v1 Python path: no domain-native cohort assembly, no
``SubjectPipeline`` / ``TablePipeline``, and no ``StudyResult`` contract.
Use the CLI for those workflows today, or wait for domain migration. Full API
docs are deferred until that migration lands.
