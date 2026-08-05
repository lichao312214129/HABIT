Configuration System
====================

Configuration is the center of HABIT. It explains how a YAML file or Python
dictionary becomes an executable workflow. This page covers loading, path
resolution, schema layers, parameter registries, and domain configurators.

Overview
--------

.. mermaid::

   flowchart TD
     Y["YAML or Python dict"] --> L["load_config / validation"]
     L --> P["PathResolver"]
     P --> W["Workflow schema"]
     W --> V["validate_step_params"]
     V --> R["ParamSchemaRegistry"]
     R --> S["Step parameter schema"]
     W --> C["Domain Configurator"]
     C --> O["Runtime object"]

The system has three responsibilities:

* **Loading** in ``habit/core/common/configs`` reads files, resolves paths, and
  creates Pydantic models.
* **Schemas** in ``habit/core/schemas`` define structure, constraints,
  parameter registration, and reflection metadata.
* **Configurators** assemble validated objects into executable workflows.

Loading and path resolution
---------------------------

Important symbols under ``habit/core/common/configs/`` include:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Symbol
     - Responsibility
   * - ``loader.load_config(path, resolve_paths=True)``
     - Reads YAML or JSON and resolves relative paths by default.
   * - ``loader.PathResolver``
     - Resolves paths relative to the configuration file, including
       Windows/WSL conversion.
   * - ``base.BaseConfig``
     - Pydantic base model for root configurations. Strict models reject
       unknown fields with ``extra='forbid'``.
   * - ``BaseConfig.from_file(path)``
     - Convenience factory combining loading, path resolution, and validation.
   * - ``base.ConfigValidationError``
     - Validation exception containing the configuration path and Pydantic
       error details.

The public API also accepts an in-memory dictionary. It uses the same schema
validation path as ``from_file``:

.. code-block:: python

   from habit import run_preprocess

   result = run_preprocess({
       "data_dir": "data/raw",
       "out_dir": "results/preprocessed",
       "preprocessing": {},
   })

Schema layers
-------------

``habit/core/schemas/`` separates configuration into three layers:

.. list-table::
   :header-rows: 1
   :widths: 24 32 44

   * - Layer
     - Location
     - Responsibility
   * - **Workflows**
     - ``schemas/workflows/*.py``
     - Root models such as ``PreprocessingConfig``,
       ``HabitatAnalysisConfig``, and ``MLConfig``.
   * - **Steps**
     - ``schemas/steps/*.py``
     - Type definitions for individual ``params`` blocks.
   * - **Registry**
     - ``schemas/registry.py``
     - Maps ``(domain, step_type)`` to a step parameter model.

Workflow models describe the outer structure, step models describe individual
parameter blocks, and the registry connects the two.

Parameter registry and validation
---------------------------------

.. code-block:: python

   from habit.schemas.registry import ParamSchemaRegistry

   ParamSchemaRegistry.register("preprocessing", "my_step", MyStepParams)
   model = ParamSchemaRegistry.get("preprocessing", "my_step")
   ParamSchemaRegistry.ensure_initialized()

While parsing a workflow, ``validate_step_params()`` resolves each method name,
validates its parameters, and raises ``ConfigValidationError`` before
execution when the input is invalid.

The same metadata can drive GUI forms, keeping CLI YAML and GUI fields aligned.

Domain configurators
--------------------

After validation, each subsystem configurator assembles the runtime object:

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - Configurator
     - Location
     - Runtime object
   * - ``PreprocessingConfigurator``
     - ``preprocessing/configurator.py``
     - ``BatchProcessor``
   * - ``HabitatConfigurator``
     - ``habitat_analysis/configurator.py``
     - ``HabitatAnalysis``, ``HabitatMapAnalyzer``, or radiomics services
   * - ``MLConfigurator``
     - ``machine_learning/configurator.py``
     - ``HoldoutWorkflow``, ``KFoldWorkflow``, or ``ModelComparison``

Summary
-------

.. code-block:: text

   workflow schema       root structure and cross-field validation
   step schema            individual parameter types
   ParamSchemaRegistry    method name -> parameter model
   validate_step_params   validate each step
   BaseConfig / loader    load, resolve paths, and validate
   Domain Configurator    validated config -> executable object
   Runtime Registry       method name -> algorithm instance

Schema definitions live under ``schemas/``. Compatibility modules such as
``preprocessing/config_schemas.py`` may re-export them, but new fields should
be added to the canonical schema files.
