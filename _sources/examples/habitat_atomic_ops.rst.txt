Atomic operators
================

Each subject-level step is a single-argument callable. Call
``op(subject)`` (or ``op(field)``) with no :class:`~habit.recipes.Study`.

.. list-table::
   :header-rows: 1
   :widths: 24 28 24 24

   * - Operator
     - Call
     - Input
     - Output
   * - Voxel features
     - ``voxel(subject)``
     - :class:`~habit.contracts.Subject`
     - :class:`~habit.contracts.VoxelFeatureField`
   * - Supervoxels
     - ``svx(field)``
     - ``VoxelFeatureField``
     - :class:`~habit.contracts.Supervoxelization`
   * - Fit (cohort)
     - ``fitter.fit(units, cohort=...)``
     - list of units
     - :class:`~habit.contracts.HabitatModel`
   * - Assign
     - ``model.assigner()(units)``
     - units
     - :class:`~habit.contracts.HabitatMap`
   * - Pipeline
     - ``pipe(subject)``
     - ``Subject``
     - ``HabitatMap``
   * - Quantify
     - ``msi(subject, habitat_map)``
     - subject + map
     - :class:`~habit.contracts.FeatureTable`

Skip ``svx`` to cluster voxels. Skip ``fit`` when you already hold a
``.habitatmodel``. Bind extract + partition + ``model.assigner()`` as a
:class:`~habit.pipeline.SubjectPipeline` so one callable labels any new
subject.

.. literalinclude:: scripts/habitat_atomic_ops_demo.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

.. literalinclude:: scripts/habitat_atomic_ops_demo.py
   :language: python
   :start-after: # BEGIN figures
   :end-before: # END figures

.. figure:: ../_static/images/examples/habitat_atomic_overlay.png
   :alt: Habitat overlay from atomic operators
   :width: 420

   ``SubjectPipeline(...)(subject)`` → habitat labels
   (:func:`~habit.viz.plot_habitat_overlay`).

Next: :doc:`habitat_recipes`.
