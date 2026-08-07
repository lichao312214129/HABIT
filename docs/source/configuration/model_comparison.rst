Model Comparison Configuration
==============================

This page documents **model comparison** YAML used by ``habit compare``.
Schema: ``ModelComparisonConfig``. Demo:
``config/model_comparison/config_model_comparison_demo.yaml``.

Runtime path: :func:`~habit.recipes.compare_models`
(:mod:`habit.recipes.comparison`) →
:mod:`habit.domain.evaluation.comparison` →
:mod:`habit.recipes.comparison_reporting` + ``habit.viz.classification``.
The v0.1 ML comparison engine is not on this path.

Command usage: :doc:`../how_to/compare_models`. Python API:
:doc:`../api/python_api` and :doc:`../api/domain_table`.

**Example configuration:**

.. code-block:: yaml

   output_dir: ../../demo_data/results/model_comparison

   files_config:
     - path: ../../demo_data/results/ml/radiomics/all_prediction_results.csv
       model_name: radiomics
       subject_id_col: subject_id
       label_col: label
       prob_col: LogisticRegression_prob
       pred_col: LogisticRegression_pred
       split_col: dataset
     - path: ../../demo_data/results/ml/clinical/all_prediction_results.csv
       model_name: clinical
       subject_id_col: subject_id
       label_col: label
       prob_col: LogisticRegression_prob
       pred_col: LogisticRegression_pred
       split_col: dataset

   merged_data:
     enabled: true
     save_name: combined_predictions.csv

   split:
     enabled: true

   visualization:
     roc:
       enabled: true
       save_name: roc_curves.pdf
       title: ROC Curves
     dca:
       enabled: true
       save_name: decision_curves.pdf
       title: Decision Curves
     calibration:
       enabled: true
       save_name: calibration_curves.pdf
       n_bins: 5
       title: Calibration Curves
     pr_curve:
       enabled: true
       save_name: precision_recall_curves.pdf
       title: Precision-Recall Curves

   delong_test:
     enabled: true
     save_name: delong_results.json

   metrics:
     basic_metrics:
       enabled: true
     youden_metrics:
       enabled: true
     target_metrics:
       enabled: true
       targets:
         sensitivity: 0.91
         specificity: 0.91

Top-level fields
----------------

**output_dir** (required)

- **Type**: string (directory path)
- **Description**: root directory for plots, merged tables, and JSON metrics.
  Relative paths resolve from the YAML file directory.

**files_config** (required)

- **Type**: non-empty list
- **Description**: one entry per model prediction table to compare.

Each ``files_config`` item
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Field
     - Required
     - Description
   * - ``path``
     - yes
     - Prediction CSV/Excel from ``habit model`` / ``habit cv``
   * - ``subject_id_col``
     - yes
     - Subject identifier column
   * - ``label_col``
     - yes
     - Ground-truth binary label column (``0`` / ``1``)
   * - ``prob_col``
     - yes
     - Predicted probability column (continuous in ``[0, 1]``)
   * - ``pred_col``
     - no
     - Predicted class column when available
   * - ``split_col``
     - no
     - Split name column (``train`` / ``test`` / …); used when ``split.enabled``
   * - ``model_name``
     - recommended
     - Display name in plots and reports; must be unique across entries
   * - ``name``
     - no
     - Alias for ``model_name`` when ``model_name`` is omitted

If both ``model_name`` and ``name`` are omitted, HABIT uses the file stem of
``path``.

**merged_data**

- ``enabled`` (bool, default ``true``): write a combined prediction table
- ``save_name`` (default ``combined_predictions.csv``)

**split**

- ``enabled`` (bool, default ``false``): when ``true``, generate per-split
  analyses using ``split_col`` in each prediction file. Figures and DeLong
  JSON are written under ``<output_dir>/<split_name>/``.

**visualization**

Sub-blocks ``roc``, ``dca``, ``calibration``, and ``pr_curve`` each accept:

- ``enabled`` (bool, default ``true``)
- ``save_name`` (output filename under ``output_dir`` or the split subdirectory)
- ``title`` (plot title, English)
- ``n_bins`` (calibration only; number of probability bins)

**delong_test**

- ``enabled`` (bool, default ``true``)
- ``save_name`` (default ``delong_results.json``)

**metrics**

- ``basic_metrics.enabled``: accuracy / sensitivity / specificity style metrics
- ``youden_metrics.enabled``: Youden-index optimal threshold metrics
- ``target_metrics.enabled`` plus ``targets``: evaluate at fixed operating
  points (each target value must be in ``(0, 1)``)

Typical outputs under ``output_dir``
------------------------------------

- ``combined_predictions.csv`` (when merge is enabled)
- ``metrics/metrics.json``
- ``habit_run_manifest.json``
- With ``split.enabled: true`` (demo layout)::

     train/roc_curves.pdf
     train/decision_curves.pdf
     train/calibration_curves.pdf
     train/precision_recall_curves.pdf
     train/delong_results.json
     test/...   (same filenames)

- With ``split.enabled: false``, the curve PDFs and ``delong_results.json``
  sit directly under ``output_dir``.

Related machine-learning training fields remain on
:doc:`machine_learning`.
