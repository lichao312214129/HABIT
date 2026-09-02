Machine learning
================

.. note::

   Supporting tool, not the habitat core. Habitat maps and features:
   :doc:`../tutorial/habitat_analysis` · :doc:`extract_features`.

Goal: train / CV models on feature CSVs.

Demo tabular data
-----------------

ML demos read CSVs under ``demo_data/ml_data/`` (separate from imaging).
Download |download_ml_data| (extract code: |ml_data_code|) and extract so
you have e.g. ``demo_data/ml_data/breast_cancer_dataset.csv`` next to
``config/``. If the zip top level is ``ml_data/``, extract into
``demo_data/``. Habitat imaging (``preprocessed.zip``) is **not** required
for these tabular ML demos.

See also :doc:`before_you_start`.

Run (fast demo)
---------------

::

   habit check-config --config config/machine_learning/config_machine_learning_radiomics_minimal.yaml
   habit model --config config/machine_learning/config_machine_learning_radiomics_minimal.yaml --mode train

Other useful commands::

   habit model --config config/machine_learning/config_machine_learning_clinical.yaml --mode train
   habit cv --config config/machine_learning/config_machine_learning_kfold_demo.yaml
   habit model --config config/machine_learning/config_machine_learning_predict.yaml --mode predict

Your data
---------

★ Edit ``input[*].path``, subject-ID / label columns, and ``output``. Prefer
``*_minimal.yaml`` until a first train succeeds. Default ``plot_types``
includes the SHAP beeswarm (``shap``); add ``shap_bar`` /
``shap_violin`` / ``shap_heatmap`` / ``shap_dependence`` /
``shap_waterfall`` / ``shap_decision`` / ``shap_force`` when you want the
full explanation set (needs ``habitat-analysis[explain]``).

Success: metrics / predictions under the YAML output folder.

Hold-out figures from the Python gallery (demo subset of
``demo_data/ml_data/breast_cancer_dataset.csv``; test AUC 0.86 — not a
clinical claim). Full YAML configs that keep tumour-size features can look
nearly perfect; the gallery uses a harder column subset. See
:doc:`../examples/tabular_ml`.

Python twin of the ROC panel::

   from habit.viz import plot_roc

   fig = plot_roc(y_true, y_prob, model_name="LogisticRegression", title="Hold-out ROC")

.. figure:: ../_static/images/examples/tabular_ml_roc.png
   :alt: Hold-out ROC
   :width: 420

   Hold-out ROC (:func:`~habit.viz.plot_roc`).

.. figure:: ../_static/images/examples/tabular_ml_pr.png
   :alt: Hold-out precision-recall
   :width: 420

   Hold-out precision-recall (:func:`~habit.viz.plot_precision_recall`).

.. figure:: ../_static/images/examples/tabular_ml_calibration.png
   :alt: Hold-out calibration
   :width: 420

   Hold-out calibration (:func:`~habit.viz.plot_calibration`).

.. figure:: ../_static/images/examples/tabular_ml_dca.png
   :alt: Hold-out decision curve
   :width: 420

   Hold-out DCA (:func:`~habit.viz.plot_decision_curve`).

.. figure:: ../_static/images/examples/tabular_ml_confusion.png
   :alt: Hold-out confusion matrix
   :width: 420

   Hold-out confusion matrix (:func:`~habit.viz.plot_confusion_matrix`).

.. figure:: ../_static/images/examples/tabular_ml_cv_auc.png
   :alt: Five-fold CV AUC
   :width: 360

   Five-fold CV AUC from the same gallery script.

SHAP family from the same hold-out
(``docs/source/examples/scripts/tabular_ml_quickstart.py``, ``# BEGIN figures``;
needs ``shap``). Same ``plot_shap_*`` calls and titles as that script.
Not a clinical claim.

.. figure:: ../_static/images/examples/tabular_ml_shap_summary.png
   :alt: Hold-out SHAP beeswarm
   :width: 420

   :func:`~habit.viz.plot_shap_summary`

.. figure:: ../_static/images/examples/tabular_ml_shap_bar.png
   :alt: Hold-out SHAP bar
   :width: 360

   :func:`~habit.viz.plot_shap_bar`

.. figure:: ../_static/images/examples/tabular_ml_shap_violin.png
   :alt: Hold-out SHAP violin
   :width: 420

   :func:`~habit.viz.plot_shap_violin`

.. figure:: ../_static/images/examples/tabular_ml_shap_heatmap.png
   :alt: Hold-out SHAP heatmap
   :width: 420

   :func:`~habit.viz.plot_shap_heatmap`

.. figure:: ../_static/images/examples/tabular_ml_shap_dependence.png
   :alt: Hold-out SHAP dependence
   :width: 360

   :func:`~habit.viz.plot_shap_dependence`

.. figure:: ../_static/images/examples/tabular_ml_shap_waterfall.png
   :alt: Hold-out SHAP waterfall
   :width: 420

   :func:`~habit.viz.plot_shap_waterfall`

.. figure:: ../_static/images/examples/tabular_ml_shap_decision.png
   :alt: Hold-out SHAP decision
   :width: 420

   :func:`~habit.viz.plot_shap_decision`

.. figure:: ../_static/images/examples/tabular_ml_shap_force.png
   :alt: Hold-out SHAP force
   :width: 480

   :func:`~habit.viz.plot_shap_force`

Next: :doc:`compare_models`.
