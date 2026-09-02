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
   :start-after: # BEGIN example
   :end-before: # END example

Draw the figures
----------------

Paste this after the Script block (it uses ``subject0`` and
``habitat_map``). Writes ``out/habitat_atomic_overlay.png``.

.. literalinclude:: scripts/habitat_atomic_ops_demo.py
   :language: python
   :start-after: # BEGIN figures
   :end-before: # END figures

Run::

   python docs/source/examples/scripts/habitat_atomic_ops_demo.py

Output
------

Illustrative (synthetic cohort)::

   Cohort: 4 subjects -> ['subj001', 'subj002', 'subj003', 'subj004']
   HabitatMap[subj001]: habitats_present=[1, 2, 3]
   Wrote out/habitat_atomic_overlay.png

Figures
-------

Same scientific product as the two-step recipe, built operator-by-operator.

.. figure:: ../_static/images/examples/habitat_atomic_overlay.png
   :alt: Habitat overlay from atomic operators
   :width: 420

   ``SubjectPipeline(...)(subject)`` → habitat labels
   (:func:`~habit.viz.plot_habitat_overlay`).

What to read next
-----------------

* :doc:`habitat_label_match` — remap ids across observers or patients
* :doc:`habitat_custom_pipeline` — Registry.create and Spec-stage customisation
* :doc:`habitat_analysis_overview` — where this sits vs recipes
* :doc:`two_step_habitat` — same design via ``Study.fit_predict``
* :doc:`data_from_arrays` — build ``Subject`` from NumPy
