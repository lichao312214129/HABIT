Python API
============

Use HABIT programmatically from notebooks or scripts. The pattern is the same
for every pipeline step:

1. Load a typed config with ``XxxConfig.from_file(path)``.
2. Optionally set up a logger with ``habit.utils.log_utils.setup_logger``.
3. Call the domain ``run_*_from_config`` helper (or construct a Configurator).

CLI commands are thin wrappers around these helpers.

Preprocessing
-------------

.. code-block:: python

   import logging
   from pathlib import Path

   from habit.core.preprocessing.config_schemas import PreprocessingConfig
   from habit.core.preprocessing.run import run_preprocess_from_config
   from habit.utils.log_utils import setup_logger

   config_path = "config/preprocessing/config_preprocessing_demo.yaml"
   config = PreprocessingConfig.from_file(config_path)
   out = Path(config.out_dir)
   out.mkdir(parents=True, exist_ok=True)

   logger = setup_logger(
       name="my_script.preprocess",
       output_dir=out,
       log_filename="processing.log",
       level=logging.INFO,
   )
   run_preprocess_from_config(config, logger=logger)

DICOM sort
----------

.. code-block:: python

   from habit.core.dicom_sort import DicomSortConfig, run_dicom_sort

   config = DicomSortConfig.from_file("config/dicom_sort/config_sort_dicom.yaml")
   run_dicom_sort(config)

Habitat segmentation
--------------------

Train and predict are separate entry points. Predict mode requires
``pipeline_path`` on the config (or via overrides).

.. code-block:: python

   import logging
   from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
   from habit.core.habitat_analysis.run import (
       apply_habitat_cli_overrides,
       run_habitat_analysis_from_config,
   )
   from habit.utils.log_utils import setup_logger

   config = HabitatAnalysisConfig.from_file(
       "config/habitat/config_habitat_two_step.yaml"
   )
   apply_habitat_cli_overrides(config, debug=False, resume=False)

   logger = setup_logger(
       name="my_script.habitat",
       output_dir=config.out_dir,
       log_filename="habitat_analysis.log",
       level=logging.DEBUG if config.debug else logging.INFO,
   )
   results_df = run_habitat_analysis_from_config(config, logger=logger)

Feature extraction
------------------

.. code-block:: python

   from habit.core.habitat_analysis.config_schemas import FeatureExtractionConfig
   from habit.core.habitat_analysis.run import run_feature_extraction_from_config

   config = FeatureExtractionConfig.from_file(
       "config/feature_extraction/config_extract_features_demo.yaml"
   )
   run_feature_extraction_from_config(config)

Machine learning
----------------

Holdout train/predict and K-fold share ``MLConfig``. Set ``run_mode`` in YAML
or override before calling the runner.

.. code-block:: python

   from habit.core.machine_learning.config_schemas import MLConfig
   from habit.core.machine_learning.run import (
       apply_ml_mode_override,
       run_kfold_from_config,
       run_ml_from_config,
   )

   config = MLConfig.from_file(
       "config/machine_learning/config_machine_learning_radiomics.yaml"
   )
   config = apply_ml_mode_override(config, mode="train")
   run_ml_from_config(config)

   kfold_config = MLConfig.from_file(
       "config/machine_learning/config_machine_learning_kfold_demo.yaml"
   )
   run_kfold_from_config(kfold_config)

Model comparison
----------------

.. code-block:: python

   from habit.core.machine_learning.config_schemas import ModelComparisonConfig
   from habit.core.machine_learning.run import run_model_comparison_from_config

   config = ModelComparisonConfig.from_file(
       "config/model_comparison/config_model_comparison_demo.yaml"
   )
   run_model_comparison_from_config(config)

Top-level package exports
-------------------------

``import habit`` lazily exposes ``HabitatAnalysis``, ``HabitatFeatureExtractor``,
and ``Modeling``. For new code, prefer explicit imports from
``habit.core.*`` as shown above.

See also: :doc:`../api/index` (autodoc reference), :doc:`../configuration/index` (YAML fields).
