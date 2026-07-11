Complete Request Lifecycle
=========================

This page follows one command from invocation to generated artifacts:

.. code-block:: bash

   habit cv -c config/machine_learning/config_machine_learning_kfold_demo.yaml

The command loads an ML configuration, runs K-fold cross-validation, and
generates models, metrics, and plots.

Seven stages
------------

.. mermaid::

   flowchart TD
     S1["1. CLI entry<br/>habit/cli.py"] --> S2["2. Command layer<br/>commands/cmd_ml.py"]
     S2 --> S3["3. Load and validate<br/>MLConfig.from_file()"]
     S3 --> S4["4. Core API<br/>run_kfold_from_config()"]
     S4 --> S5["5. Assemble<br/>MLConfigurator"]
     S5 --> S6["6. Execute<br/>KFoldWorkflow -> KFoldRunner"]
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
logging, delegates to the core API, and converts failures into clean CLI
errors. It contains no domain algorithm.

Stage 3: loading and validation
------------------------------

``load_config_or_exit(MLConfig, path)`` reads the file, resolves paths relative
to the configuration file, and validates it with Pydantic. Invalid fields and
types fail before computation starts. Step parameters are checked through
``ParamSchemaRegistry``.

Stage 4: core API
-----------------

``run_kfold_from_config()`` is the boundary shared by CLI and Python callers:

.. code-block:: python

   def run_kfold_from_config(config, *, logger=None, output_dir=None):
       if config.run_mode != "train":
           raise ValueError("K-fold cross-validation requires run_mode='train'.")
       configurator = MLConfigurator(config=config, logger=logger,
                                     output_dir=output_dir)
       workflow = configurator.create_kfold_workflow()
       return workflow.run()

The important sequence is **Configurator assembly followed by Orchestrator
execution**.

Stage 5: assembly
-----------------

``MLConfigurator`` translates the validated configuration into a
``KFoldWorkflow`` and resolves models, selectors, evaluators, and reporting
services through registries. The configurator assembles objects but does not
run the workflow.

Stage 6: execution
------------------

``KFoldWorkflow`` orchestrates the run, while ``KFoldRunner`` performs each
fold:

.. mermaid::

   flowchart TD
     WF["KFoldWorkflow"] --> PLAN["WorkflowPlan<br/>frozen config"]
     PLAN --> RUN["KFoldRunner"]
     RUN --> DM["DataManager<br/>load and split"]
     DM --> PB["PipelineBuilder<br/>selector -> scaler -> model"]
     PB --> FIT["fit on train fold<br/>evaluate validation fold"]
     FIT --> RES["KFoldRunResult"]

The complete sklearn Pipeline is fitted only on training folds, preventing
data leakage.

Stage 7: artifacts
------------------

The structured result is passed to reporting and visualization. Models,
metrics, and plots are written to the configured output directory. Plot text
is English by project convention.

The same lifecycle applies to preprocessing, habitat segmentation, and model
training; only the schema, configurator, workflow, and runner change.
