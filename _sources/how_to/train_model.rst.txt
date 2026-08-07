Machine learning
================

.. code-block:: bash

   habit model --config config/machine_learning/config_machine_learning_radiomics.yaml --mode train

K-fold: ``config/machine_learning/config_machine_learning_kfold_demo.yaml`` + ``habit cv`` .

**Input**: CSV / Excel feature tables (paths and column names in YAML).

**Output**: model files, ``all_prediction_results.csv`` , ROC / calibration plots.

**Configuration**: :doc:`../configuration/machine_learning`
