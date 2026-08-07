Complete Request Lifecycle
==========================

This page follows one command from invocation to generated artifacts:

.. code-block:: bash

   habit cv -c config/machine_learning/config_machine_learning_kfold_demo.yaml

The command loads a v0.1 ML configuration, translates it to the v1 document
model, runs K-fold cross-validation through the v1 recipe, and writes models,
metrics, and plots. It is a concrete tour of the :doc:`architecture` layers:
L5 CLI → v0.1 schema → translation → L4 recipe → L3 pipeline → typed result.

Seven stages
------------

.. mermaid::

   flowchart TD
     S1["1. CLI entry<br/>habit/cli.py"] --> S2["2. Command layer<br/>commands/cmd_ml.py"]
     S2 --> S3["3. Load and validate<br/>MLConfig.from_file()"]
     S3 --> S4["4. Translate<br/>LegacyConfigAdapter"]
     S4 --> S5["5. Assemble<br/>MLSpec + FeatureTable"]
     S5 --> S6["6. Execute<br/>recipes.cross_validate -> TablePipeline"]
     S6 --> S7["7. Report<br/>models, metrics, plots"]

Stage 1: CLI entry
------------------

The ``cv`` command declares its configuration option and uses a lazy import:

.. code-block:: python

   @cli.command("cv")
   @config_option()
   def cv(config):
       """Run K-fold cross-validation for model evaluation."""
       from habit.commands.cmd_ml import run_kfold
       run_kfold(config)

The function-level import keeps ``habit --help`` fast and avoids loading
optional dependencies for unrelated commands.

Stage 2: command layer
----------------------

The command layer loads configuration, creates output directories, configures
logging, delegates to the recipe, and converts failures into clean CLI
errors. It contains no domain algorithm — this is the L5 wiring rule.

Stage 3: loading and validation
-------------------------------

``load_config_or_exit(MLConfig, path)`` reads the file, resolves paths relative
to the configuration file, and validates it with Pydantic. Invalid fields and
types fail before computation starts. Step parameters are checked through
``ParamSchemaRegistry``. The result is the v0.1 configuration object — the
YAML parsing contract the CLI honours.

Stage 4: translation
--------------------

``LegacyConfigAdapter().translate(config.model_dump(), "cv")`` converts the
validated v0.1 configuration into the v1 document model (``spec`` / ``data``
/ ``legacy`` sections) and reports any lossy or renamed fields as warnings.
This is the single bridge between the two configuration worlds; the same
adapter powers :func:`habit.recipes.run_from_yaml` and
``habit migrate-config``.

Stage 5: assembly
-----------------

The command builds the two recipe inputs from the translated document:

* an immutable :class:`~habit.spec.MLSpec` — classifier, preprocessors,
  feature selectors, and metrics as ``Spec("name", params)`` references;
* a :class:`~habit.contracts.FeatureTable` — the CSV data loaded into the
  typed table contract with its outcome column.

Stage 6: execution
------------------

:func:`habit.recipes.cross_validate` runs the study. For each fold it builds
a fresh :class:`~habit.domain.TablePipeline` from the spec — table
preprocessors → feature selectors → classifier — fits it on the training
folds only, and evaluates on the held-out fold:

.. mermaid::

   flowchart TD
     CV["recipes.cross_validate"] --> SPEC["MLSpec<br/>(immutable, fingerprinted)"]
     CV --> TBL["FeatureTable"]
     SPEC --> REG["habit.domain registries<br/>resolve Spec('name', params)"]
     TBL --> FOLD["K-fold split<br/>seeded, stratified"]
     REG --> PIPE["TablePipeline per fold<br/>preprocess -> select -> classify"]
     FOLD --> PIPE
     PIPE --> FIT["fit on train folds<br/>evaluate held-out fold"]
     FIT --> RES["CVResult<br/>per-fold + aggregated metrics"]

The complete pipeline is fitted only on training folds, preventing data
leakage — the same guarantee the v0.1 ``KFoldWorkflow``/``KFoldRunner`` pair
enforced, now carried by the fitted ``TablePipeline`` itself.

Stage 7: artifacts
------------------

The structured ``CVResult`` is passed to reporting and visualization. Models,
metrics, and plots are written to the configured output directory. Plot text
is English by project convention.

The same lifecycle shape applies to the other pipeline commands — validate,
translate if v0.1, call a recipe, write artifacts. Variations worth knowing:

* ``habit model`` / ``habit cv`` **predict** mode routes by artifact format
  (a v1 ``.habitpipeline`` goes to :func:`~habit.recipes.predict_model`;
  an opaque v0.1 ``*_final_pipeline.pkl`` stays on the v0.1 engine, the only
  loader that understands it).
* Train / CV figure reporting is L4 + ``habit.viz``:
  :mod:`habit.recipes.ml_reporting` writes under ``output/visualizations/``
  with filename prefixes ``train_`` / ``test_`` / ``cv_``.
* ``habit compare`` calls :func:`~habit.recipes.compare_models`
  (:mod:`habit.recipes.comparison` + :mod:`habit.domain.evaluation.comparison`
  + :mod:`habit.recipes.comparison_reporting` + ``habit.viz`` multi-model
  curves). It does **not** use the v0.1 comparison engine.
* ``habit extract`` calls :func:`~habit.recipes.extract_habitat_features`
  (domain extractors for built-in feature types; optional unregistered plugins
  may still fall back to the compat ``HabitatMapAnalyzer``).
* ``habit radiomics`` calls :func:`~habit.recipes.traditional_radiomics`
  (recipe facade over the radiomics workflow helper).
