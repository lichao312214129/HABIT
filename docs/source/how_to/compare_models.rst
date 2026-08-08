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
