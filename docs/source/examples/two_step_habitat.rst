Two-step habitat analysis, end to end
=====================================

The classical habitat design: each subject's ROI is partitioned into
supervoxels, every supervoxel is described by its features, and the habitat
definition is learned from all subjects' supervoxels pooled together. This
example runs the full train path on a synthetic cohort:

1. build a cohort (:func:`~habit.datasets.make_synthetic_cohort`),
2. declare the analysis as a :class:`~habit.spec.HabitatSpec`,
3. fit with :func:`~habit.recipes.two_step`,
4. inspect the model, maps, feature table, and the auto-generated methods
   paragraph.

Script
------

.. literalinclude:: scripts/two_step_habitat_quickstart.py
   :language: python

Output
------

Real output of the script above (Sphinx build machine, HABIT 1.0.0)::

   Cohort: 6 subjects -> ['subj001', 'subj002', 'subj003', 'subj004', 'subj005', 'subj006']
   Spec fingerprint: 805517450b2a0e8b26b2b7fe8c10adde5aae349992127e20053ff0a93eee41b6

   --- Fitted habitat model ---
   HabitatModel kmeans-1f45d79eaaa3d7b5
     habitats           : 3
     features (2)    : T1, T2
     defining cohort    : n=6, name=synthetic
     modalities         : T1, T2
     cohort digest      : 9e5093ef0a362899...
     produced by        : habitat_model_fitter.kmeans
     habit version      : 1.0.0
     random seed        : 42
     preprocessing state: inertia, selection_report, validation

   Habitat maps: 6 (one per subject, label ids 1..3)
   Feature table: 6 subjects x 40 features
   First feature columns: ['habitat_1_voxel_count', 'habitat_1_volume_fraction', 'habitat_2_voxel_count', 'habitat_2_volume_fraction', 'habitat_3_voxel_count', 'habitat_3_volume_fraction']

   --- Methods paragraph (from the run manifest) ---
   Habitat imaging analysis was performed with HABIT (version 1.0.0). The analysis
   specification 'habitat_two_step' comprised voxel feature extraction with raw
   (modalities=['T1', 'T2']); supervoxelization with kmeans (n_init=5,
   n_supervoxels=8); habitat model fitting with kmeans (max_habitats=3,
   min_habitats=2, n_init=5, validation='silhouette'); habitat assignment with
   nearest_centroid (default parameters); habitat feature families: volume (default
   parameters), msi (default parameters), ith_score (default parameters). ...

   To persist everything: result.save('out/two_step_demo')

What to read next
-----------------

* :doc:`apply_saved_model` — persist the model and project it onto new subjects
* :doc:`../api/python_api` — the narrative Python API guide
* :class:`~habit.recipes.StudyResult` — what a recipe returns
