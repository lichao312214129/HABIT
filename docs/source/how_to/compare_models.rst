Model comparison
================

Train models first, then:

.. code-block:: bash

   habit compare --config config/model_comparison/config_model_comparison_demo.yaml

**Output** (under ``output_dir``): ``roc_curves.pdf`` , ``decision_curves.pdf`` ,
``calibration_curves.pdf`` , ``precision_recall_curves.pdf`` , ``delong_results.json`` ,
``combined_predictions.csv`` .

**Configuration**: :doc:`../configuration/model_comparison`
