Two-step habitat analysis, end to end
=====================================

**Level:** recipe · **Data:** synthetic · **Extras:** optional ``[view]`` · **Time:** ~30–90 s

The classical habitat design: each subject's ROI is partitioned into
supervoxels, every supervoxel is described by its features, and the habitat
definition is learned from all subjects' supervoxels pooled together. This
example runs the full train path on a synthetic cohort:

1. build a cohort (:func:`~habit.datasets.make_synthetic_cohort`),
2. declare ordered :class:`~habit.spec.Stage` entries on
   :class:`~habit.spec.HabitatSpec` (partition + pool ⇒ two_step),
3. fit with :func:`~habit.recipes.fit_habitat`,
4. inspect the model, maps, feature table, and the auto-generated methods
   paragraph.

The thin alias :func:`~habit.recipes.two_step` remains for compat; it
validates the design then calls ``fit_habitat``.

Script
------

.. literalinclude:: scripts/two_step_habitat_quickstart.py
   :language: python

Output
------

Real output of the script above (Sphinx build machine, HABIT 1.1.0)::

   Cohort: 6 subjects -> ['subj001', 'subj002', 'subj003', 'subj004', 'subj005', 'subj006']
   Spec fingerprint: d5021a7c1b5700a806deaff4fe51f259f91d58f19818cc6b8017b3acd590a631

   --- Fitted habitat model ---
   HabitatModel kmeans-5730b58c20f11648
     habitats           : 3
     features (2)    : T1, T2
     defining cohort    : n=6, name=synthetic
     modalities         : T1, T2
     cohort digest      : 9e5093ef0a362899...
     produced by        : habitat_model_fitter.kmeans
     habit version      : 1.1.0
     random seed        : 42
     preprocessing state: inertia, selection_report, validation

   Habitat maps: 6 (one per subject, label ids 1..3)
   Feature table: 6 subjects x 47 features
   First feature columns: ['habitat_1_voxel_count', 'habitat_1_volume_fraction', 'habitat_2_voxel_count', 'habitat_2_volume_fraction', 'habitat_3_voxel_count', 'habitat_3_volume_fraction']

   --- Methods paragraph (from the run manifest) ---
   Habitat imaging analysis was performed with HABIT (version 1.1.0). The analysis
   specification 'habitat_two_step' comprised voxel feature extraction with raw
   (modalities=['T1', 'T2']); supervoxelization with kmeans (n_init=5,
   n_supervoxels=8); habitat model fitting with kmeans (max_habitats=3,
   min_habitats=2, n_init=5, validation='silhouette'); habitat assignment with
   nearest_centroid (default parameters); habitat feature families: volume (default
   parameters), msi (default parameters), ith_score (default parameters),
   non_radiomics (default parameters). ...

   To persist everything: result.save('out/two_step_demo')

The script ends with a **napari eye-check** (first subject’s habitats on
anatomy). Close the window to finish; set ``HABIT_NO_VIEW=1`` to skip. For
3D review, also open the image + habitat map in ITK-SNAP / 3D Slicer /
SimpleITK.

Export YAML for CLI / YAML-API replay
-------------------------------------

After constructing the same :class:`~habit.spec.HabitatSpec` in Python, call
:func:`~habit.spec.save_habitat_config` to write a complete effective v1
document (defaults expanded). Then
:func:`~habit.recipes.run_from_yaml` or ``habit get-habitat --config`` on
that file reproduces the habitat maps voxel-wise. See
:doc:`../tutorial/quickstart_python` and :doc:`run_from_yaml`.

What to read next
-----------------

* :doc:`habitat_analysis_overview` — recipe / atomic / custom map
* :doc:`habitat_atomic_ops` — same science as single-argument callables
* :doc:`habitat_custom_pipeline` — swap components safely
* :doc:`../tutorial/quickstart_python` — demo_data path + napari screenshots
* :doc:`apply_saved_model` — persist the model and project it onto new subjects
* :class:`~habit.recipes.StudyResult` — what a recipe returns
