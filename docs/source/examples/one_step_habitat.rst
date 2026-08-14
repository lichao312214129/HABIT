One-step habitat analysis
=========================

**Level:** recipe · **Data:** ``demo_data/preprocessed`` · **Extras:** optional
``[view]`` · **Time:** ~20–60 s

The one-step design clusters **voxels inside each subject independently**.
Declare stages with **neither** ``partition`` **nor** ``pool``, then call
:meth:`~habit.recipes.Study.fit_predict`. There is **no cohort-level preprocessing
chain** at train time — per-subject state is frozen into
``StudyResult.subject_models`` rather than a single
:class:`~habit.contracts.HabitatModel`.

The factory :func:`~habit.recipes.one_step_habitat` remains for convenience.

Script
------

Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed tree.

.. literalinclude:: scripts/one_step_habitat_demo.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

Output
------

Illustrative::

   Cohort: 2 subjects
   Cohort-level habitat_model: None
   Per-subject models: 2
   Habitat maps: 2

Running the script regenerates gallery PNGs; ``HABIT_NO_VIEW=1`` skips napari.

Figures
-------

.. figure:: ../_static/images/examples/one_step_overlay.png
   :alt: One-step habitat overlay
   :width: 420

   Per-subject habitats (:func:`~habit.viz.plot_habitat_overlay`).

.. figure:: ../_static/images/examples/one_step_volume_fractions.png
   :alt: One-step volume fractions
   :width: 420

   Volume fractions.

.. figure:: ../_static/images/examples/one_step_msi_matrix.png
   :alt: One-step MSI heatmap
   :width: 420

   MSI matrix.

.. figure:: ../_static/images/examples/one_step_ith_summary.png
   :alt: One-step ITH summary
   :width: 520

   ITH summary.

.. figure:: ../_static/images/examples/one_step_cluster_validation.png
   :alt: One-step cluster validation curves
   :width: 520

   Auto-K curves from a per-subject ``selection_report`` when present.

What to read next
-----------------

* :doc:`../how_to/habitat_components` — which ``Spec`` names exist, and what each parameter means
* :doc:`habitat_analysis_overview` — recipe / atomic / custom map
* :doc:`habitat_atomic_ops` — operator-level walkthrough
* :doc:`habitat_preprocessing` — how preprocessing chains differ by design
* :doc:`two_step_habitat` — the cohort-level alternative
* :doc:`direct_pooling_habitat` — pool all voxels before clustering
