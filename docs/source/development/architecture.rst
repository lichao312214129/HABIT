Architecture
============

HABIT combines configuration-driven execution, lazy assembly, and explicit
contracts. This separates algorithm implementations from runtime parameters
while providing a consistent interface for multi-subject medical-image
workflows.

Design principles
-----------------

1. **Configuration as interface**: workflows are described by validated
   configuration objects.
2. **Typed schemas**: Pydantic catches unknown fields and invalid types before
   computation.
3. **Registry pattern**: algorithms are registered by name instead of being
   hard-coded into workflow logic.
4. **Unified contracts**: registries and orchestrators expose predictable
   methods.
5. **Lazy imports**: optional and heavy dependencies load only when needed.

Layered architecture
--------------------

.. mermaid::

   flowchart TD
     subgraph Entry["Entry points"]
       CLI["CLI"]
       PY["Python API"]
       GUI["Web GUI"]
     end
     CMD["Command layer<br/>load config and delegate"]
     CORE["Core workflows<br/>habit/core/"]
     INFRA["Schemas, registries,<br/>configurators"]
     DOMAIN["Preprocessing, habitat,<br/>machine learning"]
     UTIL["Shared utilities<br/>habit/utils/"]
     CLI --> CMD
     GUI --> CMD
     PY --> CORE
     CMD --> INFRA
     INFRA --> DOMAIN
     DOMAIN --> UTIL

Configuration-to-execution chain
--------------------------------

.. mermaid::

   flowchart TD
     Y["Configuration"] --> S["Pydantic workflow schema"]
     S --> V["Step parameter validation"]
     V --> C["Domain Configurator"]
     C --> R["Factory / Registry"]
     R --> A["Algorithm instances"]
     C --> O["Orchestrator"]
     O --> X["run / fit / predict"]

Key components
--------------

* ``ParamSchemaRegistry`` maps configuration method names to Pydantic
  parameter models.
* Domain configurators assemble executable orchestrators and keep heavy
  imports lazy.
* ``ClassRegistry`` and ``CallableRegistry`` provide the shared registration
  interface.
* Orchestrators execute complete workflows such as ``BatchProcessor``,
  ``HabitatAnalysis``, and ``KFoldWorkflow``.

Subsystems
----------

**Preprocessing** uses ``BatchProcessor`` and ``PreprocessorFactory`` to apply
configured steps to each subject.

**Habitat analysis** uses a pipeline of explicit steps and services. Its
supported strategies are ``two_step``, ``one_step``, and ``direct_pooling``.
Fitted state is persisted for reproducible prediction.

**Machine learning** separates workflow orchestration from runner execution.
``PipelineBuilder`` keeps feature selection, resampling, and the model inside
one sklearn Pipeline to prevent data leakage.

CLI-to-core mapping
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 26 24 30

   * - CLI command
     - Configuration
     - Core entry
     - Orchestrator
   * - ``preprocess``
     - ``PreprocessingConfig``
     - ``run_preprocess_from_config``
     - ``BatchProcessor``
   * - ``get-habitat``
     - ``HabitatAnalysisConfig``
     - ``run_habitat_analysis_from_config``
     - ``HabitatAnalysis``
   * - ``extract``
     - ``FeatureExtractionConfig``
     - ``run_feature_extraction_from_config``
     - ``HabitatMapAnalyzer``
   * - ``model``
     - ``MLConfig``
     - ``run_ml_from_config``
     - ``HoldoutWorkflow``
   * - ``cv``
     - ``MLConfig``
     - ``run_kfold_from_config``
     - ``KFoldWorkflow``

See :doc:`invariants` for the contracts enforced by architecture tests.
