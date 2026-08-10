Image preprocessing API (batch + atomic)
========================================

Public symbols (registered in ``habit.api.registry``):

* :func:`~habit.run_preprocess` / recipe :func:`~habit.recipes.preprocess_images`
  — **batch** directory pipeline (``data_dir`` → ``processed_images/``).
* :func:`~habit.preprocess_subject` — **atomic** subject-level operator
  (``Subject`` in → ``Subject`` out; no YAML, no filesystem).
* :func:`~habit.preprocess_image` — **atomic** single-volume operator
  (:class:`~habit.api.image.ImageVolume` in → ``ImageVolume`` out).

The atomic surfaces satisfy the embedding red line: a third-party notebook
can call ``preprocess_subject(cohort[0], steps)`` on one failing case without
accepting HABIT's directory conventions.

Steps use the same ordered mapping shape as the YAML ``preprocessing:``
block (``resample``, ``zscore_normalization``, ``n4_correction``, …).

Script
------

.. literalinclude:: scripts/image_preprocessing_api_demo.py
   :language: python

Coverage
--------

Exercised by ``demo_data/results/api/run_api_coverage.py`` step
``01_preprocess`` (batch under ``01_preprocess/batch/`` plus in-memory
atomic checks).

What to read next
-----------------

* :doc:`habitat_fit_modes` — habitat modes on a processed cohort
* :doc:`habitat_preprocessing_api` — clustering **feature** chains
  (different domain: voxel/supervoxel matrices, not images)
* :doc:`../configuration/preprocessing` — every preprocessor module
