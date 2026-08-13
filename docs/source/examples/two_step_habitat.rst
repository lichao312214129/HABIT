Two-step habitat analysis, end to end
=====================================

**Level:** recipe · **Data:** ``demo_data/preprocessed`` · **Extras:** optional
``[view]`` · **Time:** ~30–90 s

The classical habitat design: each subject's ROI is partitioned into
supervoxels, every supervoxel is described by its features, and the habitat
definition is learned from all subjects' supervoxels pooled together.

1. load a cohort (:func:`~habit.cohort_from_directory`),
2. declare ordered :class:`~habit.spec.Stage` entries on
   :class:`~habit.spec.HabitatSpec` (partition + pool ⇒ two_step),
3. fit with :meth:`~habit.recipes.Study.fit_predict`,
4. save maps / features under ``out/``.

The factory :func:`~habit.recipes.two_step_habitat` remains for convenience.

Script
------

Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed tree (same
layout as :func:`~habit.cohort_from_directory`).

.. literalinclude:: scripts/two_step_habitat_quickstart.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

Output
------

Illustrative (counts / fingerprint depend on your ``demo_data``)::

   Cohort: 2 subjects -> ['subj001', 'subj002']
   Habitat maps: 2
   Saved study to out/two_step_demo

Running the script also regenerates gallery PNGs and may open a **napari
eye-check**. ``HABIT_NO_VIEW=1`` skips the viewer.

Figures
-------

.. figure:: ../_static/images/examples/two_step_overlay.png
   :alt: Two-step habitat overlay on anatomy
   :width: 420

   Habitat overlay (:func:`~habit.viz.plot_habitat_overlay`).

.. figure:: ../_static/images/examples/two_step_triptych.png
   :alt: Anatomy, supervoxels, and habitats
   :width: 720

   Anatomy | supervoxels | habitats (:func:`~habit.viz.plot_partition_triptych`).

.. figure:: ../_static/images/examples/two_step_volume_fractions.png
   :alt: Habitat volume fractions bar chart
   :width: 420

   Volume fractions (:func:`~habit.viz.plot_habitat_volume_fractions`).

.. figure:: ../_static/images/examples/two_step_msi_matrix.png
   :alt: MSI spatial interaction heatmap
   :width: 420

   MSI matrix (:func:`~habit.viz.plot_msi_matrix`).

.. figure:: ../_static/images/examples/two_step_ith_summary.png
   :alt: ITH score summary
   :width: 520

   ITH summary (:func:`~habit.viz.plot_ith_summary`).

.. figure:: ../_static/images/examples/two_step_cluster_validation.png
   :alt: Auto-K cluster validation curves
   :width: 520

   Auto-K curves (:func:`~habit.viz.plot_cluster_validation_from_report`).

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
