Habitat feature-preprocessing API (subject + cohort chains)
===========================================================

Distinguish two domains:

* **Image preprocessing** — intensities / geometry
  (:doc:`image_preprocessing_api`).
* **Clustering feature preprocessing** — voxel / supervoxel matrices on the
  way to a habitat definition (this page).

Declare chains on :class:`~habit.spec.HabitatSpec`:

* ``voxel_feature_preprocessors`` / ``supervoxel_feature_preprocessors`` —
  **subject-level**, stateless (each subject's own statistics).
* ``cohort_feature_preprocessors`` — **cohort-level**, fitted once; state
  travels inside :class:`~habit.contracts.HabitatModel`.

After :func:`~habit.domain.assembly.build_habitat_components`, the same
vocabulary appears on :class:`~habit.domain.assembly.HabitatComponents`
(singular ``*_feature_preprocessor`` for the assembled chains). There is no
abbreviated second naming (``voxel_extractor`` / ``cohort_chain`` / …).

Atomic call without a recipe: build a
:class:`~habit.domain.feature_preprocessing.SubjectPreprocessingChain` via
:func:`~habit.domain.assembly.build_subject_chain` and invoke it on one
feature matrix.

To inspect **raw** clustering units (no feature preprocessing), build a bare
spec without the three preprocessor fields and call
``components.pipeline(assigner=None).units(subject)``. For raw voxel features
only: ``components.voxel_feature_extractor(subject).feature_frame()``.

Inspect every step
------------------

For debugging and QA, pass ``inspect=`` a :class:`~habit.inspection.StepRecorder`
into ``recipes.Study(...).fit_predict`` (or a factory-built Study) and
``Study.from_model(...).predict``. Default ``inspect=None`` is zero-cost and
bit-identical.

Captured boundaries are primarily **stage-bound** names such as
``extract_voxel_features.output``, ``preprocess1.output``,
``partition.output``, ``extract_supervoxel_features.output``,
``pool.output``, ``fit.output``, ``assign.output``, and
``quantify.output``. Legacy aliases in ``habit.STEP_NAMES``
(``voxel_features.raw``, ``supervoxels.described``, …) may still appear
on some paths; prefer the stage-bound names when reading
``result.inspection``. After ``pool`` / ``fit``, cohort-level records use
subject id ``__cohort__``. Use ``steps=``, ``subjects=``, and
``max_subjects=`` to limit memory.

In-memory inspection is **not** supported with the process backend; use
serial / ``workers=1`` while debugging.

Script
------

.. literalinclude:: scripts/habitat_preprocessing_api_demo.py
   :language: python

Coverage
--------

Feature chains are enabled in
``demo_data/results/api/run_api_coverage.py`` steps ``02`` / ``03`` / ``04``.

The script ends with a **napari eye-check**. ``HABIT_NO_VIEW=1`` skips it.

What to read next
-----------------

* :doc:`habitat_fit_modes` — three habitat modes that consume these chains
* :doc:`habitat_preprocessing` — narrative twin of the subject / cohort chains
* :doc:`../api/domain` — ``HabitatComponents`` attribute table
* :doc:`../api/domain_habitat` — ``StepRecorder`` / supervoxel feature extractors
