Habitat recipes API (three modes + apply + persistence)
=======================================================

Named study designs in ``habit.recipes``:

* :func:`~habit.recipes.fit_habitat` — unified entry; dispatches on the
  spec's declared dataflow (``HabitatSpec.pooling`` + ``supervoxelizer``)
* :func:`~habit.recipes.two_step` — supervoxels then cohort habitats
* :func:`~habit.recipes.one_step` — per-subject voxel habitats
* :func:`~habit.recipes.direct_pooling` — pool voxels across the cohort
* :func:`~habit.recipes.apply_habitat_model` — reuse a fitted
  :class:`~habit.contracts.HabitatModel`

The three mode-named functions are thin aliases: they validate the spec's
dataflow declaration and call ``fit_habitat``, so numerics are identical
whichever entry you use.

**Batch** — pass a :class:`~habit.contracts.Cohort`.
**Atomic** — ``result.pipeline(subject)`` labels one
:class:`~habit.contracts.Subject` with no cohort / backend / YAML.

Persistence:

* :meth:`~habit.recipes.StudyResult.save` — NRRD maps, feature table,
  units parquet, cluster plots, ``run_manifest.json``
* :meth:`~habit.contracts.HabitatModel.save` /
  :meth:`~habit.contracts.HabitatModel.load` — ``.habitatmodel`` archive

Both subject-level and cohort-level **feature** preprocess chains
(``voxel_feature_preprocessors`` / ``cohort_feature_preprocessors``) are
shown in the script; see also :doc:`habitat_preprocessing_api`.

Script
------

.. literalinclude:: scripts/habitat_recipes_api_demo.py
   :language: python

Coverage
--------

``demo_data/results/api/`` steps ``02_habitat_two_step``,
``03_habitat_one_step``, ``04_habitat_direct_pooling``.

The script ends with a **napari eye-check**. ``HABIT_NO_VIEW=1`` skips it.

What to read next
-----------------

* :doc:`habitat_preprocessing_api` — feature-chain details
* :doc:`parallel_execution` / :doc:`viz_parallel_extras_api` — ``RunPolicy`` + process pool
* :doc:`features_radiomics_api` — habitat / traditional feature extraction
