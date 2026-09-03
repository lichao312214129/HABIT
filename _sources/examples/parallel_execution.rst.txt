Parallel and checkpoints
========================

:class:`~habit.spec.HabitatSpec` declares what to compute;
:class:`~habit.spec.RunPolicy` declares how to schedule it. Pass a
:class:`~habit.execution.process_pool.ProcessPoolBackend` into
:meth:`~habit.recipes.Study.fit_predict`. With a fixed ``random_seed``
the maps match serial execution.

.. include:: ../_includes/windows_multiprocessing.rst

The demo already wraps work in ``if __name__ == "__main__":``.

.. literalinclude:: scripts/parallel_execution_demo.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

.. literalinclude:: scripts/parallel_execution_demo.py
   :language: python
   :start-after: # BEGIN figures
   :end-before: # END figures

.. figure:: ../_static/images/examples/parallel_execution_overlay.png
   :alt: Habitat overlay from the parallel execution demo
   :width: 420

   Habitats after ``Study.fit_predict`` (serial or process-pool)
   (:func:`~habit.viz.plot_habitat_overlay`).

Checkpoints
-----------

Attach a :class:`~habit.execution.CheckpointStore` so a second run skips
subjects already recorded as success (``checkpoint=store`` on
:meth:`~habit.contracts.Cohort.map` or the recipe backend).
Recorded failures stay skipped unless ``retry_failed_subjects=True``;
force a few IDs with ``force_rerun_subjects``.
On the recipe path, cache keys embed the spec fingerprint — changing
extractors or preprocess stages without a new store looks like a cache
hit on the wrong definition.
``RunPolicy.on_subject_failure="continue"`` isolates errors inside the
backend; recipes pass ``raise_on_failure=False`` so a partial cohort
can finish.
Full knob list: :doc:`../tutorial/execution`.

Next: :doc:`habitat_recipes` · :doc:`rigor`.
