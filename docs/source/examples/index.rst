Examples
========

Runnable, end-to-end examples of the v1.0 Python API. Every script is
self-contained, deterministic (fixed seeds), and runs without downloaded
data — synthetic cohorts and tables stand in for real data so the examples
work anywhere. When ``demo_data/`` is present locally, several scripts
automatically exercise the same paths as ``demo_data/results/api/``.

Embedding HABIT
---------------

HABIT is designed to drop into **your** pipeline — not only batch YAML jobs:

* **Batch** — pass a :class:`~habit.contracts.Cohort` or feature table to
  ``habit.recipes.*`` (same objects the CLI builds internally).
* **Non-batch / atomic** — call :class:`~habit.domain.pipeline.SubjectPipeline`
  on one :class:`~habit.contracts.Subject`, slice ``cohort[0:1]``, or run
  tabular recipes on a single-row table. Image I/O also exposes
  :func:`~habit.preprocess_subject` / :func:`~habit.preprocess_image`.

Coverage map (aligned with ``demo_data/results/api/run_api_coverage.py``)
-------------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Capability
     - Example page(s)
   * - Image preprocessing (batch + atomic)
     - :doc:`image_preprocessing`, :doc:`image_preprocessing_api`
   * - Habitat three modes
     - :doc:`two_step_habitat`, :doc:`one_step_habitat`, :doc:`direct_pooling_habitat`, :doc:`habitat_recipes_api`
   * - Clustering preprocessing (subject + cohort chains)
     - :doc:`habitat_preprocessing`, :doc:`habitat_preprocessing_api`
   * - Feature routes (raw / concat / radiomics / slic)
     - :doc:`habitat_feature_routes`
   * - Feature trees (combiners / statistics / aliases)
     - :doc:`feature_composition`
   * - Precise-feature screen (precision screen + whitelist)
     - :doc:`precise_features`
   * - Train + apply ``.habitatmodel``
     - :doc:`two_step_habitat`, :doc:`apply_saved_model`
   * - Feature extraction + radiomics
     - :doc:`feature_extraction`, :doc:`features_radiomics_api`
   * - Tabular ML (train / cv / predict / compare / pipeline I/O)
     - :doc:`tabular_ml`, :doc:`ml_advanced`, :doc:`tabular_ml_api`
   * - Visualization (``habit.viz``)
     - :doc:`visualization`, :doc:`viz_parallel_extras_api`
   * - Persist ``StudyResult`` / ``RunManifest`` / models
     - :doc:`persistence`, :doc:`apply_saved_model`
   * - Parallel ``RunPolicy`` + ``ProcessPoolBackend``
     - :doc:`parallel_execution`, :doc:`viz_parallel_extras_api`
   * - Fault tolerance (geometry / fail_fast / plugins / Cohort.map)
     - :doc:`fault_tolerance`
   * - YAML / CLI twin
     - :doc:`run_from_yaml`, :doc:`cli_yaml_workflows`
   * - Cohort, plugins, icc/retest/dice/config tools
     - :doc:`cohort_plugins_auxiliary`

.. note::

   Synthetic demos use :func:`~habit.datasets.make_synthetic_cohort` and
   :func:`~habit.datasets.make_synthetic_feature_table`. For your own images,
   swap cohort construction for :func:`~habit.cohort_from_directory` —
   everything downstream is identical.

.. toctree::
   :maxdepth: 1

   two_step_habitat
   one_step_habitat
   direct_pooling_habitat
   habitat_preprocessing
   habitat_feature_routes
   feature_composition
   precise_features
   custom_voxel_features
   apply_saved_model
   image_preprocessing
   feature_extraction
   tabular_ml
   ml_advanced
   visualization
   persistence
   parallel_execution
   fault_tolerance
   run_from_yaml
   cli_yaml_workflows
   cohort_plugins_auxiliary
   image_preprocessing_api
   habitat_recipes_api
   habitat_preprocessing_api
   tabular_ml_api
   features_radiomics_api
   viz_parallel_extras_api

The scripts live in ``docs/source/examples/scripts/`` in the repository and
can be run directly::

   python docs/source/examples/scripts/two_step_habitat_quickstart.py

Why not sphinx-gallery
----------------------

sphinx-gallery is not part of this repository's toolchain; adding it would
execute every example on each documentation build and require a CI imaging
stack. These pages follow the same narrative-plus-runnable-code format with
plain ``literalinclude`` scripts and captured output instead — the scripts
remain directly runnable and testable without a gallery plugin.
