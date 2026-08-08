One-step habitat analysis
=========================

The one-step design clusters **voxels inside each subject independently**.
There is no supervoxel stage and **no cohort-level preprocessing chain** at
train time — per-subject state is frozen into
``StudyResult.subject_models`` rather than a single
:class:`~habit.contracts.HabitatModel`.

Script
------

.. literalinclude:: scripts/one_step_habitat_demo.py
   :language: python

Output
------

::

   Cohort: 4 subjects

   Cohort-level habitat_model: None
   Per-subject models: 4 subjects
     subj001: 3 habitats, id=kmeans-...
     subj002: 3 habitats, id=kmeans-...
     subj003: 3 habitats, id=kmeans-...
     subj004: 3 habitats, id=kmeans-...

   Habitat maps: 4
   Feature table: 4 rows x 47 columns

The script ends with a **napari eye-check**. Close the window to finish;
``HABIT_NO_VIEW=1`` skips it. Prefer ITK-SNAP / 3D Slicer / SimpleITK for 3D.

What to read next
-----------------

* :doc:`habitat_preprocessing` — how preprocessing chains differ by design
* :doc:`two_step_habitat` — the cohort-level alternative
* :doc:`direct_pooling_habitat` — pool all voxels before clustering
