Data model (``habit.contracts``)
================================

Every volumetric and tabular object in the v1 core lives here. Import explicitly::

   from habit.contracts import Geometry, Subject, Cohort, HabitatModel, FeatureTable

Geometry
--------

.. code-block:: python

   from habit.contracts import Geometry

   geom = Geometry.from_array(
       (64, 128, 128),                 # NumPy shape (z, y, x)
       spacing=(1.0, 1.0, 1.0),        # SimpleITK (x, y, z) mm
       origin=(0.0, 0.0, 0.0),
   )
   assert geom.is_compatible_with(geom)

Or construct with all fields::

   geom = Geometry(
       shape=(64, 128, 128),
       spacing=(1.0, 1.0, 1.0),
       origin=(0.0, 0.0, 0.0),
       direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
   )

Image references and volumes
----------------------------

.. code-block:: python

   import numpy as np
   from habit.contracts import ArrayImageRef, Geometry, ImageVolume, MaskVolume

   geom = Geometry.from_array((32, 64, 64), spacing=(1.0, 1.0, 1.0))
   array = np.zeros(geom.shape, dtype=np.float32)
   mask = np.ones(geom.shape, dtype=np.uint8)

   # Materialised volumes
   image = ImageVolume(array, geom)
   mask_vol = MaskVolume(mask, geom)

   # Lazy in-memory reference (load() materialises)
   ref = ArrayImageRef(array, geom)
   loaded = ref.load()

``ImageRef`` is the protocol; ``ArrayImageRef`` and ``FileImageRef``
(:doc:`adapters`) are concrete implementations.

Subject and Cohort
------------------

.. code-block:: python

   import numpy as np
   from habit.contracts import Cohort, Geometry, ImageVolume, MaskVolume, Subject

   geom = Geometry.from_array((32, 64, 64))
   subject = Subject(
       subject_id="P001",
       images={
           "T1": ImageVolume(np.zeros(geom.shape, dtype=np.float32), geom),
           "T2": ImageVolume(np.zeros(geom.shape, dtype=np.float32), geom),
       },
       masks={"tumor": MaskVolume(np.ones(geom.shape, dtype=np.uint8), geom)},
       metadata={"center": "A"},
   )
   cohort = Cohort([subject], name="synthetic")
   print(len(cohort), cohort.subject_ids)
   fingerprint = cohort.summarize()  # -> CohortFingerprint

Synthetic cohort (no files on disk)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For tutorials, tests, and API exploration, build an in-memory cohort:

.. code-block:: python

   from habit import make_synthetic_cohort

   cohort = make_synthetic_cohort(
       n_subjects=4,
       modalities=("T1", "T2"),
       shape=(32, 32, 32),
       rng=42,
   )

See :doc:`python_api` for recipe workflows on synthetic data.

Load from a HABIT directory layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When images and masks follow the conventional layout on disk, use either
the convenience helper or :class:`~habit.adapters.DirectoryDataSource`
(:doc:`adapters`):

.. code-block:: python

   from habit.contracts import cohort_from_directory

   cohort = cohort_from_directory(
       "/path/to/processed_images",
       modalities=["T1", "T2"],
       roi="tumor",
       name="training",
   )

Equivalent::

   from habit.contracts import Cohort
   cohort = Cohort.from_directory(
       "/path/to/processed_images",
       modalities=["T1", "T2"],
       roi="tumor",
       name="training",
   )

Map a subject operator over the cohort (default serial backend)::

   maps = cohort.map(pipeline)

Habitat artefacts
-----------------

.. code-block:: python

   from pathlib import Path
   from habit.contracts import HabitatModel

   # After HabitatModelFitter.fit(...)
   print(model.n_habitats, model.feature_names, model.model_id)
   print(model.summary())
   assigner = model.assigner()                 # HabitatAssigner
   model.save(Path("out/model.habitatmodel"))
   restored = HabitatModel.load(Path("out/model.habitatmodel"))

Related types produced along the pipeline:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Type
     - Role
   * - ``VoxelFeatureField``
     - Per-voxel features inside the ROI (``feature_names``, ``values``, ``voxel_index``)
   * - ``Supervoxelization``
     - Per-subject units (``label_array``, ``features``, ``geometry``)
   * - ``HabitatMap``
     - Habitat label map (``model_id``, ``habitat_ids``, ``provenance``)
   * - ``HabitatModel``
     - Population definition (centroids, preprocessing_state, spec_payload, …)

FeatureTable
------------

.. code-block:: python

   import pandas as pd
   from habit.contracts import BinaryOutcome, FeatureTable

   table = FeatureTable(
       frame=pd.DataFrame(
           {
               "subject": ["a", "b"],
               "msi": [0.1, 0.2],
               "ith": [0.3, 0.4],
               "label": [0, 1],
           }
       ),
       id_columns=("subject",),
       feature_columns=("msi", "ith"),
       outcome=BinaryOutcome("label"),
   )
   X = table.feature_matrix()          # features only
   merged = table.join(other_table)    # column-role-aware join

Outcome
-------

The endpoint is declared as an **object**, not a column name: a survival
endpoint occupies two columns, and a bare name cannot tell a downstream metric
whether to compute AUC or R-squared.

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Declaration
     - ``task``
     - Use
   * - ``BinaryOutcome("label", positive_label=1)``
     - ``binary``
     - Two-class endpoints; the positive class is explicit because
       sensitivity, PPV and decision-curve analysis are defined relative to it
   * - ``MulticlassOutcome("grade", classes=("I", "II", "III"))``
     - ``multiclass``
     - Three or more classes; declaring ``classes`` pins probability-column
       and confusion-matrix order
   * - ``ContinuousOutcome("volume_change")``
     - ``continuous``
     - Regression endpoints
   * - ``SurvivalOutcome(time_column="os_time", event_column="os_event")``
     - ``survival``
     - Right-censored time-to-event; ``event_value`` carries the coding, so
       ``1``/``0`` and ``"Dead"``/``"Alive"`` tables both work unchanged

Components read the endpoint through :mod:`habit.domain.outcome_access` rather
than indexing the frame themselves:

.. code-block:: python

   from habit.domain.outcome_access import (
       outcome_series,          # one-column endpoints -> Series
       require_outcome,         # declare which families a component supports
       structured_survival_array,  # scikit-survival (event, time) layout
       survival_target,         # -> (time, event mask), validated
   )

   y = outcome_series(table, owner="classifier.logistic")

``outcome_series`` deliberately **raises** on a survival endpoint instead of
returning the time column, and ``survival_target`` validates the follow-up
times and rejects a fully censored table. ``FeatureTable.outcome_column``
remains available as a shortcut for one-column endpoints.

Dispatch on the ``task`` string, never on ``isinstance`` against a closed set:
an endpoint family added later (competing risks, for instance) is then
rejected with a precise message rather than silently mistaken for a built-in
one.

Provenance
----------

.. code-block:: python

   from habit.contracts import Provenance

   root = Provenance.source("raw_images")
   derived = root.derive(
       produced_by="habitat_model_fitter.kmeans",
       spec_fingerprint="abc123",
       random_seed=42,
   )

RunManifest and StudyResult
---------------------------

Contracts ``RunManifest`` is the **study-level** manifest (not the legacy
workflow JSON helper).

``StudyResult`` is what a recipe returns and therefore lives at L4
(``habit.recipes``), not in ``habit.contracts``: no layer below the recipes
produces or consumes one, and only L4 is allowed to know about output
directories.

.. code-block:: python

   from habit.contracts import RunManifest
   from habit.recipes import StudyResult

   # Built during a study; then:
   text = manifest.describe_methods(style="radiology")  # or "nature"
   checklist = manifest.checklist("CLEAR")  # IBSI | CLEAR | METRICS | TRIPOD+AI
   versions = manifest.software_versions()
   seeds = manifest.random_seeds()
   manifest.to_json("out/run_manifest.json")

   result = StudyResult(
       habitat_model=model,
       pipeline=pipe,
       features=table,
       habitat_maps=tuple(maps),
       manifest=manifest,
   )
   out_dir = result.save("out/study")
   # writes <subject>_habitats.nrrd, habitat_model.habitatmodel,
   # habitat_features.csv, run_manifest.json

``save`` is convenience sugar over ``result.write(writer)``: the layout above
belongs to ``habit.adapters.DirectoryResultWriter``, and any object satisfying
the ``ResultWriter`` protocol (an object store, a DICOM-SEG exporter, a no-op
sink) can take its place without the study knowing.

Operator protocols
------------------

These are structural contracts (not registry components):

.. code-block:: python

   from habit.contracts import (
       CohortOperator,
       DataSource,
       ExecutionBackend,
       ResultWriter,
       SubjectOperator,
       SubjectResult,
   )

* ``SubjectOperator`` — callable on one ``Subject``
* ``CohortOperator`` — callable on a ``Cohort``
* ``DataSource.load()`` → ``Cohort``
* ``ResultWriter`` — persist study outputs
* ``ExecutionBackend.map(op, cohort)`` — see :doc:`execution`
* ``SubjectResult`` — per-subject outcome wrapper
