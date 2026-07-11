Python API
============

Use HABIT programmatically from notebooks or scripts. The **stable public
contract** is the top-level ``habit`` package (see :doc:`../development/index`
and ``CHANGELOG.md``). CLI commands are thin wrappers around the same runners.

Typical pattern:

1. Load a typed config with ``XxxConfig.from_file(path)`` from ``habit``.
2. Optionally set up a logger with ``habit.setup_logger``.
3. Call the matching ``run_*`` helper.

Migration from deep imports
---------------------------

Older examples imported from ``habit.core.*``. Those paths still work, but new
code should prefer the top-level names below.

Preprocessing
-------------

.. code-block:: python

   import logging
   from pathlib import Path

   from habit import PreprocessingConfig, run_preprocess, setup_logger

   config = PreprocessingConfig.from_file(
       "config/preprocessing/config_preprocessing_demo.yaml"
   )
   out = Path(config.out_dir)
   out.mkdir(parents=True, exist_ok=True)

   logger = setup_logger(
       name="my_script.preprocess",
       output_dir=out,
       log_filename="processing.log",
       level=logging.INFO,
   )
   run_preprocess(config, logger=logger)

DICOM sort
----------

.. code-block:: python

   from habit import DicomSortConfig, run_dicom_sort

   config = DicomSortConfig.from_file("config/dicom_sort/config_sort_dicom.yaml")
   run_dicom_sort(config)

Habitat segmentation
--------------------

Train and predict are separate entry points. Predict mode requires
``pipeline_path`` on the config (or via overrides).

.. code-block:: python

   import logging

   from habit import (
       HabitatAnalysisConfig,
       apply_habitat_cli_overrides,
       run_habitat_analysis,
       setup_logger,
   )

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
   results_df = run_habitat_analysis(config, logger=logger)

Feature extraction
------------------

.. code-block:: python

   from habit import load_feature_extraction_config, run_feature_extraction

   config, plugin_configs = load_feature_extraction_config(
       "config/feature_extraction/config_extract_features_demo.yaml"
   )
   run_feature_extraction(config, plugin_configs=plugin_configs)

``FeatureExtractionConfig.from_file(...)`` remains suitable when the YAML uses
only built-in feature types. Use ``load_feature_extraction_config`` when the
configuration includes plugin-specific sections, so programmatic execution
matches the ``habit extract-features`` CLI.

Traditional radiomics
---------------------

.. code-block:: python

   from habit import RadiomicsConfig, run_radiomics

   config = RadiomicsConfig.from_file(
       "config/radiomics/config_traditional_radiomics.yaml"
   )
   run_radiomics(config)

Machine learning
----------------

Holdout train/predict and K-fold share ``MLConfig``. Set ``run_mode`` in YAML
or override before calling the runner.

.. code-block:: python

   from habit import (
       MLConfig,
       apply_ml_mode_override,
       run_kfold,
       run_ml,
   )

   config = MLConfig.from_file(
       "config/machine_learning/config_machine_learning_radiomics.yaml"
   )
   config = apply_ml_mode_override(config, mode="train")
   run_ml(config)

   kfold_config = MLConfig.from_file(
       "config/machine_learning/config_machine_learning_kfold_demo.yaml"
   )
   run_kfold(kfold_config)

Model comparison
----------------

.. code-block:: python

   from habit import ModelComparisonConfig, run_model_comparison

   config = ModelComparisonConfig.from_file(
       "config/model_comparison/config_model_comparison_demo.yaml"
   )
   run_model_comparison(config)

ICC analysis
------------

.. code-block:: python

   from habit import ICCConfig, run_icc_analysis

   config = ICCConfig.from_file("config/auxiliary/config_icc_demo.yaml")
   run_icc_analysis(config)

Test-retest analysis
--------------------

``run_test_retest_analysis`` maps retest habitat labels to the corresponding
test labels, then writes the remapped images. It returns the mapping for
callers that need to inspect or persist it::

   from habit import TestRetestConfig, run_test_retest_analysis

   config = TestRetestConfig.from_file(
       "config/auxiliary/config_test_retest.yaml"
   )
   habitat_mapping = run_test_retest_analysis(config)

Top-level package exports
-------------------------

``import habit`` lazily exposes the runners, config classes, and helpers listed
above. The canonical symbol list is defined in ``habit.api.registry.PUBLIC_API_SYMBOLS``
and verified by ``tests/api/test_public_api.py``.

Optional dependencies can be probed without importing heavy backends::

   import habit

   if habit.is_available("radiomics"):
       ...

See also: :doc:`../api/index` (autodoc reference), :doc:`../configuration/index` (YAML fields).
