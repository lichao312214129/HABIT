Publication figures with habit.viz
==================================

Interactive napari
(:func:`~habit.viz.view_habitat_napari` / default ``habit view``)
is a convenient way to inspect habitat maps after ``get-habitat`` — see
region vs contour screenshots in :doc:`../tutorial/quickstart`. It needs the
optional ``[view]`` extra (recommended at :doc:`../tutorial/installation`);
without it the CLI falls back to a matplotlib PNG. Force the static path with
``habit view --backend matplotlib`` (:func:`~habit.viz.plot_habitat_overlay`,
needs ``[viz]``).

For fuller **3D** review, load the source image and ``*_habitats.nrrd``
together in **ITK-SNAP**, **3D Slicer**, or a **SimpleITK**-based viewer
(label overlay / segmentation).

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

Habitat graph topology figures
------------------------------

When extracting the built-in ``graph`` family with ``graph.visualize: true``,
the recipe writes optional figures under ``visualizations/graph/``. The same
construction is available as pure plotters (``[viz]``; 3D also needs
``[view]``)::

   from habit.viz import (
       plot_habitat_graph_network_2d,
       plot_habitat_graph_slice,
       render_habitat_graph_network_3d,
       render_habitat_graph_surface_3d,
   )

See :doc:`graph_features` and :doc:`../reference/features/graph`.

Voxel texture / feature-map slices
----------------------------------

Local entropy (and densified ``VoxelFeatureField`` columns such as
``voxel_radiomics``) use the same pure-figure contract::

   from habit.viz import dense_voxel_feature_map, plot_voxel_texture_slice

See :doc:`voxel_texture` and :doc:`../how_to/voxel_texture`.

What to read next
-----------------

* :doc:`persistence` — ``StudyResult.save(write_cluster_plots=True)``
* :doc:`../api/python_api` — when to use ``habit.viz`` vs CLI plot outputs
