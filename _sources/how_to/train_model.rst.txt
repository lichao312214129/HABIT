Machine learning
================

.. code-block:: bash

   habit model --config config/machine_learning/config_machine_learning_radiomics.yaml --mode train

K-fold: ``config/machine_learning/config_machine_learning_kfold_demo.yaml`` + ``habit cv`` .

**Input**: CSV / Excel feature tables (paths and column names in YAML).

**Output** (under the YAML ``output`` directory):

* Fitted pipelines (when ``is_save_model`` is true) and prediction tables
  (e.g. ``all_prediction_results.csv``).
* ``metrics.json`` — train / test (hold-out) or fold-aggregated (CV) metrics.
* Evaluation figures under ``output/visualizations/`` when visualization is
  enabled. Filenames use a split prefix:

  * hold-out train: ``train_*`` (e.g. ``train_roc_curve.pdf``)
  * hold-out test: ``test_*``
  * cross-validation (pooled OOF): ``cv_*``

  Curve types follow ``visualization.plot_types`` (``roc``, ``dca``,
  ``calibration``, ``pr``, ``confusion``, plus optional ``shap`` /
  ``shap_dependence`` / ``shap_waterfall`` / ``permutation``). Figures are
  drawn by ``habit.viz`` and written through
  :mod:`habit.recipes.ml_reporting`.

**Configuration**: :doc:`../configuration/machine_learning`
