Habitat fit modes (fit_habitat + apply + I/O)
=============================================

**Level:** recipe API · **Data:** synthetic · **Extras:** optional ``[view]`` · **Time:** ~30–90 s

Named study designs in ``habit.recipes``:

* :func:`~habit.recipes.fit_habitat` — unified entry; stage dataflow executor
* :func:`~habit.recipes.two_step` / :func:`~habit.recipes.one_step` /
  :func:`~habit.recipes.direct_pooling` — thin aliases that validate shape
* :func:`~habit.recipes.apply_habitat_model` — reuse a fitted
  :class:`~habit.contracts.HabitatModel`

Strategy is inferred: partition+pool → two_step; pool only → direct_pooling;
neither → one_step. The ``pool`` marker is the subject↔cohort watershed.

**Batch** — pass a :class:`~habit.contracts.Cohort`.
**Atomic** — ``result.pipeline(subject)`` labels one subject (see also
:doc:`habitat_atomic_ops`).

Persistence:

* :meth:`~habit.recipes.StudyResult.save` — NRRD maps, feature table, units
  parquet, cluster plots, ``run_manifest.json``
* :meth:`~habit.contracts.HabitatModel.save` /
  :meth:`~habit.contracts.HabitatModel.load` — ``.habitatmodel`` archive

Script
------

.. literalinclude:: scripts/habitat_recipes_api_demo.py
   :language: python

The script ends with a **napari eye-check**. ``HABIT_NO_VIEW=1`` skips it.

What to read next
-----------------

* :doc:`habitat_analysis_overview` — layer map
* :doc:`habitat_preprocessing` — feature-chain details
* :doc:`parallel_execution` — ``RunPolicy`` + process pool
* :doc:`feature_extraction` — habitat / traditional feature extraction
