Machine learning
================

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
``*_minimal.yaml`` until a first train succeeds (full configs may enable slow
SHAP plots).

Success: metrics / predictions under the YAML output folder.

Next: :doc:`compare_models`.
