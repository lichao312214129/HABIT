Direct-pooling habitat analysis
===============================

Direct pooling skips supervoxels and clusters **all ROI voxels pooled across
the cohort**. Both ``voxel_feature_preprocessors`` (per subject) and
``cohort_feature_preprocessors`` (on the pooled table) run during training,
producing one cohort-level :class:`~habit.contracts.HabitatModel`.

Script
------

.. literalinclude:: scripts/direct_pooling_habitat_demo.py
   :language: python

Output
------

::

   Cohort: 5 subjects

   --- Cohort-level habitat model ---
   HabitatModel kmeans-b2267e39480550ff
     habitats           : 3
     features (2)    : T1, T2
     produced by        : habitat_model_fitter.kmeans+cohort_preprocessing
     preprocessing state: cohort_feature_preprocessor, inertia, ...

   Habitat maps: 5
   Clustering units (voxel rows): 4148
   Feature table: (5, 65)

The script ends with a **napari eye-check**. Close the window to finish;
``HABIT_NO_VIEW=1`` skips it. Prefer ITK-SNAP / 3D Slicer / SimpleITK for 3D.

What to read next
-----------------

* :doc:`habitat_preprocessing` — subject vs cohort preprocessing chains
* :doc:`two_step_habitat` — supervoxel intermediate stage
* :doc:`apply_saved_model` — reuse a cohort-level model on new subjects
