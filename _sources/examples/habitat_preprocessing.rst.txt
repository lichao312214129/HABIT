Habitat feature preprocessing chains
======================================

Clustering operates on **preprocessed feature matrices**, not raw intensities.
v1 ``HabitatSpec`` exposes three ordered chains (v0.1 names in parentheses):

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - v1 field
     - v0.1 YAML key
     - When it runs
   * - ``voxel_feature_preprocessors``
     - ``preprocessing_for_subject_level``
     - Per subject, on voxel features **before** supervoxels/units form
   * - ``supervoxel_feature_preprocessors``
     - (subject-level, two-step only)
     - Per subject, on supervoxel features **after** supervoxelization
   * - ``cohort_feature_preprocessors``
     - ``preprocessing_for_group_level``
     - Fitted once on pooled training rows; replayed at apply

Design rules
------------

* **One-step** — ``voxel_feature_preprocessors`` only. No cohort chain at
  train: each subject clusters independently; frozen state lives in
  ``subject_models[*].preprocessing_state``.
* **Two-step / direct-pooling** — all applicable chains run during training.
  The cohort chain is fitted on pooled units, stored in
  ``HabitatModel.preprocessing_state['cohort_feature_preprocessor']``, and
  **replayed** by :func:`~habit.recipes.apply_habitat_model` (centroids only
  mean something in the training feature space).
* **Batch** — ``recipes.two_step(cohort, spec)`` (or ``one_step`` /
  ``direct_pooling``).
* **Non-batch (atomic)** — :class:`~habit.domain.pipeline.SubjectPipeline`:

  * ``pipeline.units(subject)`` — fit-time units with subject-level chains
  * ``pipeline(subject)`` — label one subject when an assigner is attached

Script
------

.. literalinclude:: scripts/habitat_preprocessing_demo.py
   :language: python

Output (abbreviated)
--------------------

::

   === v0.1 -> v1 preprocessing chain names ===
     preprocessing_for_subject_level          -> voxel_feature_preprocessors
     (two-step only, subject)                 -> supervoxel_feature_preprocessors
     preprocessing_for_group_level            -> cohort_feature_preprocessors

   === Non-batch: SubjectPipeline.units (fit-time, no assigner) ===
     subj001: 42 units, 3 features, range [0.000, 1.000]

   === Batch: two_step (voxel + supervoxel + cohort chains) ===
   HabitatModel kmeans-...
     produced by        : habitat_model_fitter.kmeans+cohort_preprocessing
   Preprocessing state keys: ['cohort_feature_preprocessor', 'inertia', ...]

   === Batch: one_step (voxel chain only; no cohort chain) ===
     subject_models: 2
     cohort habitat_model: None

   === Train freeze + apply replay (cohort preprocessing) ===
     apply batch: 2 maps
     apply atomic (1 subject): subj001

What to read next
-----------------

* :doc:`habitat_feature_routes` — raw / radiomics / concat / slic feature paths
* :doc:`apply_saved_model` — replaying frozen preprocessing at apply time
* :doc:`../configuration/habitat` — YAML field reference
