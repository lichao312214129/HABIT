Habitat analysis
================

HABIT's core is **habitat analysis**: turn images + an ROI into a
:class:`~habit.contracts.HabitatMap`, then quantify those subregions.
Image preprocessing and tabular ML are supporting tools — they are not
required to call a habitat operator.

Beginners: run the demo first (:doc:`quickstart` or
:doc:`quickstart_python`), then come back here to choose a strategy.
Integrators: skip YAML; start at **Three layers** and
:doc:`../examples/habitat_atomic_ops`.

Three layers (outer shells last)
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Layer
     - When to use it
     - Start here
   * - **Atomic operators**
     - Embed one step in a notebook, MONAI loop, or another product.
       ``op(subject)`` — no directory layout, no YAML, no ``Cohort``.
     - :doc:`../examples/habitat_atomic_ops`
   * - **SubjectPipeline**
     - Bind **definition + procedure** and label one
       :class:`~habit.contracts.Subject`.
     - :doc:`../api/domain_habitat`
   * - **Study recipe**
     - Whole-cohort fit / predict, optional persist + figures.
     - :doc:`quickstart_python`, :doc:`../examples/two_step_habitat`

CLI / YAML assemble the same operators. They are optional shells
(:doc:`quickstart`).

Bring your own arrays (nibabel, SimpleITK, MONAI, …) with
:doc:`../examples/data_from_arrays`. Parallel runs and partial-cohort
failure: :doc:`execution`.

Dataflow (classical two-step)
-----------------------------

::

   Subject
     │  voxel_feature_extractor     ← op(subject)
     ▼
   VoxelFeatureField
     │  supervoxelizer              ← omit for one_step / direct_pooling
     ▼
   Supervoxelization (units)
     │  pool across subjects        ← only cohort watershed
     ▼
   HabitatModel.fit(...)            ← only cohort-level step
     │  assigner bound to model
     ▼
   SubjectPipeline(subject) → HabitatMap
     │  quantify families           ← op(subject, map)
     ▼
   FeatureTable (+ RunManifest)

You may **stop after any arrow** and hand the object to your own code.
That is the embedding contract.

Which strategy
--------------

Strategy is **inferred from stages**, not from the function name:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Strategy
     - Stages
     - Use when
   * - **two_step**
     - ``partition`` + ``pool``
     - Shared cohort definition; supervoxels first (typical paper pipeline).
   * - **direct_pooling**
     - ``pool`` only
     - Shared cohort definition; cluster voxels (no supervoxels).
   * - **one_step**
     - neither
     - Habitats defined **per subject**. Integer ids are permuted —
       align them before comparing patients (:doc:`../examples/habitat_label_match`).

Primary recipe API: :class:`~habit.recipes.Study`
(:meth:`~habit.recipes.Study.fit_predict` /
:meth:`~habit.recipes.Study.predict`). Factories
``two_step_habitat`` / ``one_step_habitat`` /
``direct_pooling_habitat`` declare the intended design.

A :class:`~habit.report.Report` is a **run** object (persist + figures),
not a scientific stage. Use it when a one-step cohort must stream
artefacts per subject (:doc:`../examples/one_step_habitat`).

What the maps look like
-----------------------

Figures from the two-step gallery (:doc:`../examples/two_step_habitat`).

.. figure:: ../_static/images/examples/two_step_overlay.png
   :alt: Habitat overlay on anatomy
   :width: 420

   Habitat overlay — the primary visual product.

.. figure:: ../_static/images/examples/two_step_triptych.png
   :alt: Anatomy, supervoxels, habitats
   :width: 720

   Two-step partitions.

.. figure:: ../_static/images/examples/two_step_msi_matrix.png
   :alt: MSI heatmap
   :width: 360

   MSI after habitats exist.

What must stay paired
---------------------

A published habitat definition is **two objects**:

1. :class:`~habit.contracts.HabitatModel` (centroids + cohort preprocessing
   state)
2. the :class:`~habit.domain.SubjectPipeline` (or the same
   :class:`~habit.HabitatSpec` stages) that produced the fit-time units

Shipping the model without the matching procedure — or changing upstream
extractors silently — is how labels look plausible but are wrong. See
:doc:`../examples/apply_saved_model`.

Matching ids after independent clustering
-----------------------------------------

One-step (and any per-subject ``fit_predict``) emits **permuted**
integers. Two matchers, one order:

* Same tumour, two observers: overlap
  (:func:`~habit.kernels.habitat_label_match.match_labels_by_overlap`).
* Different patients: unscaled texture means, one cohort z-score,
  Hungarian
  (:func:`~habit.kernels.habitat_label_match.match_labels_by_features`).

Runnable numbers: :doc:`../examples/habitat_label_match`.

Voxel features that define habitats
-----------------------------------

Clustering is only as reproducible as the voxel maps you feed it.
Morphology-aware screening (which features survive a simulated
re-acquisition) is :doc:`precise_screening`. Local entropy / GLCM as
**inputs** (not post-habitat CSV families): :doc:`../how_to/voxel_texture`.

Which ``Spec("...")`` name and parameters to put in each stage:
:doc:`../how_to/habitat_components`.

Next
----

* Beginner demo: :doc:`quickstart` (CLI) or :doc:`quickstart_python` (Study)
* Embed operators: :doc:`../examples/habitat_atomic_ops`
* Your arrays: :doc:`../examples/data_from_arrays`
* Swap components: :doc:`../examples/habitat_custom_pipeline`
* Parallel / fault tolerance: :doc:`execution`
* Features on existing maps: :doc:`../how_to/extract_features` ·
  :doc:`../how_to/graph_features`
