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

Atomic call without a recipe: build a
:class:`~habit.domain.feature_preprocessing.SubjectPreprocessingChain` via
:func:`~habit.domain.assembly.build_subject_chain` and invoke it on one
feature matrix.

Script
------

.. literalinclude:: scripts/habitat_preprocessing_api_demo.py
   :language: python

Coverage
--------

Feature chains are enabled in
``demo_data/results/api/run_api_coverage.py`` steps ``02`` / ``03`` / ``04``.

What to read next
-----------------

* :doc:`habitat_recipes_api` — three habitat modes that consume these chains
* :doc:`habitat_preprocessing` — narrative twin of the subject / cohort chains
