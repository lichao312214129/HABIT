Direct-pooling habitat analysis
===============================

**Level:** recipe · **Data:** synthetic · **Extras:** optional ``[view]`` · **Time:** ~20–60 s

Direct pooling skips supervoxels and clusters **all ROI voxels pooled across
the cohort**. Declare stages with a ``pool`` marker and **no** ``partition``,
then call :func:`~habit.recipes.fit_habitat`. Preprocess stages may run
before and after ``pool`` (post-pool feature preprocess is first-class),
producing one cohort-level :class:`~habit.contracts.HabitatModel`.

The thin alias :func:`~habit.recipes.direct_pooling` remains for compat.

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

* :doc:`habitat_analysis_overview` — recipe / atomic / custom map
* :doc:`habitat_custom_pipeline` — customise stages for pooling designs
* :doc:`habitat_preprocessing` — subject vs cohort preprocessing chains
* :doc:`two_step_habitat` — supervoxel intermediate stage
* :doc:`apply_saved_model` — reuse a cohort-level model on new subjects
