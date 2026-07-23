Python API
============

Use HABIT programmatically from notebooks or scripts. The **stable public
contract** is the top-level ``habit`` package (see :doc:`../development/index`
and ``CHANGELOG.md``). CLI commands are thin wrappers around the same runners.

Typical pattern:

1. Use a high-level object for an interactive clinical research workflow, load
   a typed config with ``XxxConfig.from_file(path)``, or construct a Python
   dictionary and pass it directly to ``run_*``.
2. Optionally set up a logger with ``habit.setup_logger``.
3. Call the matching ``run_*`` helper and inspect its ``WorkflowResult``.

Migration from deep imports
---------------------------

Older examples imported from ``habit.core.*``. Those paths still work, but new
code should prefer the top-level names below.

Recommended object API
----------------------

For notebook-based clinical research, use the high-level object API. It keeps
the familiar sklearn lifecycle while preserving the existing CLI and YAML
contracts. ``Cohort`` represents a prepared study directory;
``ClinicalPreprocessor`` applies an established preprocessing configuration;
and ``HabitatSegmenter`` trains or applies a habitat model.

.. code-block:: python

   from habit import (
       ClinicalPreprocessor,
       Cohort,
       HabitatSegmenter,
       OutcomeClassifier,
   )

   training_cohort = Cohort.from_directory(
       "data/training",
       name="development",
   )

   preprocessor = ClinicalPreprocessor(preprocessing_config)
   prepared_cohort = preprocessor.fit_transform(training_cohort)

   segmenter = HabitatSegmenter(
       habitat_config,
       prediction_output_dir="results/external_habitats",
   )
   training_habitats = segmenter.fit_transform(prepared_cohort)
   external_habitats = segmenter.predict(
       Cohort.from_directory("data/external", name="external_validation")
   )

   classifier = OutcomeClassifier(ml_config)
   classifier.fit(patient_feature_table, outcome_labels)
   outcome_probabilities = classifier.predict_proba(external_feature_table)

Use the configuration-driven functions below for a one-command workflow,
batch execution, CLI parity, and fully explicit experiment provenance.

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
   result = run_preprocess(config, logger=logger)
   print(result.artifact("output_dir"))

In-memory configuration
-----------------------

Every public ``run_*`` workflow accepts its matching validated config class or
a plain Python dictionary. Dictionaries use the same Pydantic validation as
``XxxConfig.from_file`` and invalid values fail before the workflow starts.

.. code-block:: python

   from habit import run_preprocess

   result = run_preprocess(
       {
           "data_dir": "data/raw",
           "out_dir": "results/preprocessed",
           "processes": 4,
           "preprocessing": {
               "resample": {
                   "images": ["T1", "T2"],
                   "target_spacing": [1.0, 1.0, 1.0],
               }
           },
       }
   )
   processed_root = result.artifact("output_dir")

DICOM sort
----------

.. code-block:: python

   from habit import DicomSortConfig, run_dicom_sort

   config = DicomSortConfig.from_file("config/dicom_sort/config_sort_dicom.yaml")
   run_dicom_sort(config)

The same operation accepts an in-memory dictionary:

.. code-block:: python

   from pathlib import Path
   import yaml

   from habit import run_dicom_sort

   config = yaml.safe_load(
       Path("config/dicom_sort/config_sort_dicom.yaml").read_text(encoding="utf-8")
   )
   config["out_dir"] = "results/dicom_sorted"
   result = run_dicom_sort(config)

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
   result = run_habitat_analysis(config, logger=logger)
   results_df = result.data
   pipeline_path = result.artifacts.get("pipeline")

For an in-memory configuration, construct the same schema as the YAML template:

.. code-block:: python

   from pathlib import Path
   import yaml

   from habit import run_habitat_analysis

   config = yaml.safe_load(
       Path("config/habitat/config_habitat_two_step.yaml").read_text(encoding="utf-8")
   )
   config["data_dir"] = "data/training"
   config["out_dir"] = "results/habitats"
   config["run_mode"] = "train"
   result = run_habitat_analysis(config)

Feature extraction
------------------

.. code-block:: python

   from habit import load_feature_extraction_config, run_feature_extraction

   config, plugin_configs = load_feature_extraction_config(
       "config/feature_extraction/config_extract_features_demo.yaml"
   )
   result = run_feature_extraction(config, plugin_configs=plugin_configs)

``FeatureExtractionConfig.from_file(...)`` remains suitable when the YAML uses
only built-in feature types. Use ``load_feature_extraction_config`` when the
configuration includes plugin-specific sections, so programmatic execution
matches the ``habit extract`` CLI.

For an in-memory configuration, use ``build_feature_extraction_config``. It
preserves plugin-specific settings rather than discarding them during shared
schema validation:

.. code-block:: python

   from pathlib import Path
   import yaml

   from habit import build_feature_extraction_config, run_feature_extraction

   config_mapping = yaml.safe_load(
       Path(
           "config/feature_extraction/config_extract_features_demo.yaml"
       ).read_text(encoding="utf-8")
   )
   config_mapping["out_dir"] = "results/habitat_features"
   config, plugin_configs = build_feature_extraction_config(config_mapping)
   result = run_feature_extraction(config, plugin_configs=plugin_configs)

Traditional radiomics
---------------------

.. code-block:: python

   from habit import RadiomicsConfig, run_radiomics

   config = RadiomicsConfig.from_file(
       "config/radiomics/config_traditional_radiomics.yaml"
   )
   result = run_radiomics(config)

The workflow also accepts an in-memory dictionary:

.. code-block:: python

   from pathlib import Path
   import yaml

   from habit import run_radiomics

   config = yaml.safe_load(
       Path(
           "config/radiomics/config_traditional_radiomics.yaml"
       ).read_text(encoding="utf-8")
   )
   config["out_dir"] = "results/radiomics"
   result = run_radiomics(config)

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
   result = run_ml(config)
   ml_run = result.data

   kfold_config = MLConfig.from_file(
       "config/machine_learning/config_machine_learning_kfold_demo.yaml"
   )
   kfold_result = run_kfold(kfold_config)

Both holdout and K-fold workflows accept an in-memory dictionary. The example
below constructs a mapping from a template, then changes only study-specific
values:

.. code-block:: python

   from pathlib import Path
   import yaml

   from habit import run_kfold, run_ml

   holdout_config = yaml.safe_load(
       Path(
           "config/machine_learning/config_machine_learning_radiomics.yaml"
       ).read_text(encoding="utf-8")
   )
   holdout_config["output"] = "results/holdout"
   holdout_result = run_ml(holdout_config)

   kfold_config = yaml.safe_load(
       Path(
           "config/machine_learning/config_machine_learning_kfold_demo.yaml"
       ).read_text(encoding="utf-8")
   )
   kfold_config["output"] = "results/kfold"
   kfold_result = run_kfold(kfold_config)

sklearn-compatible estimators
----------------------------

Use the estimators when a HABIT component needs to participate in a standard
scikit-learn workflow. ``SubjectFeatureAggregator`` converts a habitat long
table to one feature row per subject. Subject identifiers stay in the index by
default, preventing them from being treated as predictive features.

.. code-block:: python

   from habit import SubjectFeatureAggregator

   aggregator = SubjectFeatureAggregator(aggregation="mean")
   subject_features = aggregator.fit_transform(habitat_feature_table)
   assert subject_features.index.name == "subject"

``HabitClassifier`` accepts either a validated ``MLConfig`` or an in-memory
dictionary accepted by ``MLConfig``. It follows the standard
``fit`` / ``predict`` / ``predict_proba`` / ``score`` classifier protocol.

.. code-block:: python

   from habit import HabitClassifier

   classifier = HabitClassifier(
       config={
           "run_mode": "train",
           "input": [{"path": "train.csv", "subject_id_col": "subject", "label_col": "label"}],
           "output": "results/ml",
           "models": {"LogisticRegression": {"params": {"solver": "liblinear"}}},
           "feature_selection_methods": [],
           "is_visualize": False,
       }
   )
   classifier.fit(x_train, y_train)
   probabilities = classifier.predict_proba(x_test)
   classifier.save("models/classifier.joblib")

The saved estimator records HABIT and scikit-learn version metadata. Load only
artifacts from trusted sources because joblib uses Python pickle semantics.

.. code-block:: python

   loaded_classifier = HabitClassifier.load("models/classifier.joblib")
   predictions = loaded_classifier.predict(x_test)

Model comparison
----------------

.. code-block:: python

   from habit import ModelComparisonConfig, run_model_comparison

   config = ModelComparisonConfig.from_file(
       "config/model_comparison/config_model_comparison_demo.yaml"
   )
   result = run_model_comparison(config)
   metrics = result.data

An in-memory model-comparison configuration follows the same schema:

.. code-block:: python

   from pathlib import Path
   import yaml

   from habit import run_model_comparison

   config = yaml.safe_load(
       Path(
           "config/model_comparison/config_model_comparison_demo.yaml"
       ).read_text(encoding="utf-8")
   )
   config["output_dir"] = "results/model_comparison"
   result = run_model_comparison(config)

ICC analysis
------------

.. code-block:: python

   from habit import ICCConfig, run_icc_analysis

   config = ICCConfig.from_file("config/auxiliary/config_icc_demo.yaml")
   result = run_icc_analysis(config)

The matching in-memory form is:

.. code-block:: python

   from pathlib import Path
   import yaml

   from habit import run_icc_analysis

   config = yaml.safe_load(
       Path("config/auxiliary/config_icc_demo.yaml").read_text(encoding="utf-8")
   )
   config["output"]["path"] = "results/icc.csv"
   result = run_icc_analysis(config)

Test-retest analysis
--------------------

``run_test_retest_analysis`` maps retest habitat labels to the corresponding
test labels, then writes the remapped images. The mapping is returned through
``WorkflowResult.data`` for callers that need to inspect or persist it::

   from habit import TestRetestConfig, run_test_retest_analysis

   config = TestRetestConfig.from_file(
       "config/auxiliary/config_test_retest.yaml"
   )
   habitat_mapping = run_test_retest_analysis(config).data

The same analysis can be started from an in-memory mapping:

.. code-block:: python

   from pathlib import Path
   import yaml

   from habit import run_test_retest_analysis

   config = yaml.safe_load(
       Path(
           "config/auxiliary/config_test_retest.yaml"
       ).read_text(encoding="utf-8")
   )
   config["out_dir"] = "results/test_retest"
   habitat_mapping = run_test_retest_analysis(config).data

Image I/O and geometry
----------------------

See :doc:`image` for ``read_image`` / ``read_mask`` / ``align_image_mask``.

Low-level radiomics extraction
------------------------------

See :doc:`radiomics_api` for ``extract_features`` / ``extract_batch`` (component
API). The YAML workflow remains ``run_radiomics``.

Plugins, provenance, and errors
-------------------------------

See :doc:`plugins` and :doc:`contracts`.

Top-level package exports
-------------------------

``import habit`` lazily exposes the runners, config classes, and helpers listed
above. The canonical symbol list is defined in ``habit.api.registry.PUBLIC_API_SYMBOLS``
and verified by ``tests/api/test_public_api.py``.

Optional dependencies can be probed without importing heavy backends::

   import habit

   if habit.is_available("radiomics"):
       ...

See also: :doc:`index` (autodoc reference), :doc:`../configuration/index` (YAML fields).
