Model comparison
================

Train models first, then:

.. code-block:: bash

   habit compare --config config/model_comparison/config_model_comparison_demo.yaml

``habit compare`` calls :func:`~habit.recipes.compare_models`
(:mod:`habit.recipes.comparison`): domain merge / metrics / DeLong in
:mod:`habit.domain.evaluation.comparison`, then multi-model figures via
:mod:`habit.recipes.comparison_reporting` and ``habit.viz.classification``.

**Output** (under ``output_dir``):

* ``combined_predictions.csv`` — merged prediction table (when merge is enabled)
* ``metrics/metrics.json`` — per-model metric panels
* When ``split.enabled`` is true (and each file has ``split_col``), curves and
  DeLong land in per-split subdirectories (e.g. ``train/``, ``test/``):

  * ``roc_curves.pdf``, ``decision_curves.pdf``, ``calibration_curves.pdf``,
    ``precision_recall_curves.pdf``
  * ``delong_results.json``

* When split is disabled, the same figure / DeLong filenames sit directly under
  ``output_dir``.
* ``habit_run_manifest.json`` — run provenance

**Configuration**: :doc:`../configuration/model_comparison`
