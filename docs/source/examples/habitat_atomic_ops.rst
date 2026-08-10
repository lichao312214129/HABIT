Atomic habitat operators (no YAML, no recipe)
=============================================

**Level:** atomic · **Data:** synthetic · **Extras:** none · **Time:** ~10–30 s

Every subject-level step is a single-argument callable. This page walks the
classical two-step design **without** :meth:`~habit.recipes.Study.fit_predict`, so
you can embed HABIT inside another notebook or debug one failing case.

Pipeline shown
--------------

1. ``voxel(subject)`` → :class:`~habit.contracts.VoxelFeatureField`
2. ``svx(field)`` → :class:`~habit.contracts.Supervoxelization`
3. ``fitter.fit(units, cohort=...)`` → :class:`~habit.contracts.HabitatModel`
4. ``SubjectPipeline(..., model.assigner())(subject)`` → :class:`~habit.contracts.HabitatMap`
5. ``pipeline.extract_features(subject, families)`` → :class:`~habit.contracts.FeatureTable`

Fit-time vs apply-time
----------------------

* **Fit-time** pipeline: ``habitat_assigner=None`` — :meth:`~habit.domain.SubjectPipeline.units`
  works; ``__call__`` does not.
* **Apply-time** pipeline: bind ``model.assigner()`` — one callable labels any
  new :class:`~habit.contracts.Subject`.

Script
------

.. literalinclude:: scripts/habitat_atomic_ops_demo.py
   :language: python

Run::

   python docs/source/examples/scripts/habitat_atomic_ops_demo.py

What to read next
-----------------

* :doc:`habitat_custom_pipeline` — Registry.create and Spec-stage customisation
* :doc:`habitat_analysis_overview` — where this sits vs recipes
* :doc:`two_step_habitat` — same design via ``Study.fit_predict``
* :doc:`data_from_arrays` — build ``Subject`` from NumPy
