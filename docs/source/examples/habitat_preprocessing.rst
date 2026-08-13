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

Assembled live objects use the same vocabulary on
:class:`~habit.domain.assembly.HabitatComponents`
(``voxel_feature_preprocessor``, ``cohort_feature_preprocessor``, …);
see :doc:`habitat_preprocessing_api` (API deep dive) and :doc:`../api/domain`.

Extractors and preprocessors keep **different** call shapes on purpose:
``extractor(subject).feature_frame()`` is subject-level; ``preprocessor(X)``
is sklearn-like on a feature table. Prefer the assembled components from
``build_habitat_components(spec)`` for that handoff (see the API page).

* **One-step** — ``voxel_feature_preprocessors`` only. No cohort chain at
  train: each subject clusters independently; frozen state lives in
  ``subject_models[*].preprocessing_state``.
* **Two-step / direct-pooling** — all applicable chains run during training.
  The cohort chain is fitted on pooled units, stored in
  ``HabitatModel.preprocessing_state['cohort_feature_preprocessor']``, and
  **replayed** by :meth:`~habit.recipes.Study.predict` (centroids only
  mean something in the training feature space).
* **Batch** — ``recipes.Study(spec=spec).fit_predict(cohort)`` (strategy from stages /
  sugar: partition+pool, pool only, or neither). Mode-named aliases remain.
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

   === Extract → preprocess handoff (assembled components) ===
     subj001: raw (N, F) -> after voxel chain (N, F)

   === Non-batch: SubjectPipeline.units (fit-time, no assigner) ===
     subj001: 42 units, 3 features, range [0.000, 1.000]

   === Batch: Study fit_predict two_step stages (voxel + supervoxel + cohort) ===
   HabitatModel kmeans-...
     produced by        : habitat_model_fitter.kmeans+cohort_preprocessing
   Preprocessing state keys: ['cohort_feature_preprocessor', 'inertia', ...]

   === Batch: Study fit_predict one_step stages (voxel chain only) ===
     subject_models: 2
     cohort habitat_model: None

   === Train freeze + apply replay (cohort preprocessing) ===
     apply batch: 2 maps
     apply atomic (1 subject): subj001

Running the script may open a **napari eye-check**. ``HABIT_NO_VIEW=1``
skips the viewer. It also writes ``out/habitat_preprocessing_overlay.png``.

Figures
-------

Preprocessing changes the feature space; the visible product is still habitat
maps from this demo.

.. figure:: ../_static/images/examples/habitat_preprocessing_overlay.png
   :alt: Two-step habitats after preprocessing chains
   :width: 420

   Two-step with voxel + supervoxel + cohort chains
   (:func:`~habit.viz.plot_habitat_overlay`).

Skipping the feature chain (raw intensities)
--------------------------------------------

On ``demo_data/preprocessed`` (5 subjects, ``LAP``, auto-K 2–10, two-step
with 20 supervoxels, seed 0) an **empty** ``voxel_feature_preprocessors``
chain does not fail — it under-expresses habitats. Cohort ``model_k`` was
still 4, but per-subject maps used a mean of **2.8** labels (one subject
only 2). The YAML-style voxel chain ``winsorize`` then ``minmax`` restored
**4 / 4** habitats on every subject. Image-level
``zscore_normalization`` via :func:`~habit.api.preprocess_subject` (ROI
masked) before the same empty chain recovered a mean of **3.8**, but
k-means then warned that distinct intensities were fewer than
``n_supervoxels`` — z-scoring a single modality can collapse the
supervoxel feature space.

**Recipe (do not duplicate a second scaler):** keep the feature chain for
clustering; use image z-score only when you need intensity harmonization
*before* extraction::

   from dataclasses import replace
   from habit import Cohort, Spec, preprocess_subject, two_step_habitat

   processed = Cohort([
       preprocess_subject(
           s, {"zscore_normalization": {"only_inmask": True}}, mask_roi="LAP"
       )
       for s in cohort
   ])
   spec = replace(
       two_step_habitat(modalities=("LAP",), roi="LAP").spec,
       voxel_feature_preprocessors=(
           Spec("winsorize", params={"winsor_limits": (0.05, 0.05)}),
           Spec("minmax"),
       ),
   )

One-step (habitats defined inside each subject) was insensitive to the
feature chain on this demo (all 4 habitats either way); image z-score
slightly *lowered* the mean (3.6). Inter-subject scale is a two-step /
direct-pooling problem.

What to read next
-----------------

* :doc:`habitat_feature_routes` — raw / radiomics / concat / slic feature paths
* :doc:`apply_saved_model` — replaying frozen preprocessing at apply time
* :doc:`../configuration/habitat` — YAML field reference
