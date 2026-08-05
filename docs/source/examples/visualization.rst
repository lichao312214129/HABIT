Publication figures with habit.viz
==================================

Every function in :mod:`habit.viz` is **pure**: arrays or contract objects in,
a matplotlib ``Figure`` out — no ``savefig`` inside the library. Callers choose
where figures go. All text is English/ASCII via
:func:`~habit.viz.labels.sanitize_label`.

This example covers:

* population-level habitat-clustering PCA from a two-step ``StudyResult``,
* Kaplan-Meier curves (synthetic survival table),
* regression diagnostics (predicted vs observed).

Binary ML ROC/calibration plots are produced by
:func:`~habit.recipes.compare_models` (see :doc:`ml_advanced`).

Script
------

.. literalinclude:: scripts/visualization_demo.py
   :language: python

Output
------

::

   Wrote habitat_pca_2d.png (34879 bytes)
   Wrote kaplan_meier.png (25066 bytes)
   Wrote predicted_vs_observed.png (39005 bytes)

   All figures under .../habit_viz_demo_...
   Binary ML ROC/calibration plots: see ml_advanced_demo.py (compare_models).

What to read next
-----------------

* :doc:`persistence` — ``StudyResult.save(write_cluster_plots=True)``
* :doc:`../api/python_api` — when to use ``habit.viz`` vs CLI plot outputs
