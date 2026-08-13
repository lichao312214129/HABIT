Model comparison
================

Goal: compare prediction CSVs (ROC, etc.) after training.

Run the demo
------------

Train clinical + radiomics models first (:doc:`train_model`), then::

   habit check-config --config config/model_comparison/config_model_comparison_demo.yaml
   habit compare --config config/model_comparison/config_model_comparison_demo.yaml

Your data
---------

★ Edit ``output_dir`` and each ``files_config`` entry (path + ID / label /
probability column names).

Success: plots and ``metrics/`` under ``output_dir``.

Python overlay from two fitted hold-out models (staged LR vs shallow RF on
the gallery subset; AUC 0.85 vs 0.88 — not a clinical claim). See
:doc:`../examples/ml_advanced`.

::

   from habit.viz import plot_roc

   fig = plot_roc(curves=curves, title="Hold-out ROC")

.. figure:: ../_static/images/examples/ml_advanced_roc.png
   :alt: Two-model hold-out ROC
   :width: 420

   Hold-out ROC overlay (:func:`~habit.viz.plot_roc`).

.. figure:: ../_static/images/examples/ml_advanced_pr.png
   :alt: Two-model hold-out precision-recall
   :width: 420

   Hold-out precision-recall (:func:`~habit.viz.plot_precision_recall`).

.. figure:: ../_static/images/examples/ml_advanced_calibration.png
   :alt: Two-model hold-out calibration
   :width: 420

   Hold-out calibration (:func:`~habit.viz.plot_calibration`).

.. figure:: ../_static/images/examples/ml_advanced_dca.png
   :alt: Two-model hold-out decision curve
   :width: 420

   Hold-out DCA (:func:`~habit.viz.plot_decision_curve`).
