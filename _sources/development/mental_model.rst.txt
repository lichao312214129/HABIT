Core Concepts and Mental Model
==============================

Establish a shared vocabulary before reading HABIT's implementation. The
following terms are used throughout the developer documentation.

Global mental model
-------------------

HABIT connects domain concepts to engineering roles through a configuration
pipeline:

.. mermaid::

   flowchart TD
     V["Voxel feature"] --> SV["Supervoxel"] --> H["Habitat"]
     H --> F["Habitat feature"] --> M["Machine-learning model"]
     C["Configurator"] --> O["Orchestrator"]
     R["Registry"] --> O
     K["Contract"] -.-> O
     O --> V

Domain concepts
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Term
     - Meaning
   * - **Voxel**
     - The smallest unit of a 3D medical image. Habitat analysis starts with
       voxel-level features.
   * - **Voxel feature**
     - A feature vector calculated for each voxel, such as raw intensity,
       kinetic measurements, or local radiomics.
   * - **Supervoxel**
     - A local group of similar voxels within one subject, used as an
       intermediate product by the ``two_step`` strategy.
   * - **Habitat**
     - An image-phenotype region inside a tumor, represented by an integer
       label image.
   * - **Habitat feature**
     - A downstream feature calculated after habitat maps are generated, such
       as radiomics, MSI, or ITH.
   * - **Clustering mode**
     - One of ``two_step``, ``one_step``, or ``direct_pooling``.

Engineering roles
-----------------

The v1 stack (L0–L5, see :doc:`architecture`) has its own vocabulary:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Role
     - Responsibility
     - Representative symbol
   * - **Spec**
     - Declares an analysis as immutable, fingerprinted data: which
       components, with which parameters.
     - ``HabitatSpec``, ``MLSpec``, ``Spec``
   * - **Recipe**
     - A standard study design: assembles domain components per the spec,
       executes them, and returns a typed result.
     - ``recipes.Study(...).fit_predict``, ``recipes.train_model``
   * - **Contract**
     - The typed data model that travels between layers.
     - ``Subject``, ``Cohort``, ``FeatureTable``, ``HabitatModel``
   * - **Pipeline**
     - The executable object a recipe assembles from domain components;
       fitted state travels inside it.
     - ``SubjectPipeline``, ``TablePipeline``
   * - **DataSource / Sink**
     - L1 adapters; the only place files are read.
     - ``DirectoryDataSource``
   * - **Component registry**
     - Maps ``Spec("name", params)`` references to implementations.
     - the v2 capability packages registries, ``ComponentRegistry``

The classic v0.1 roles below still describe the compat engines under
``habit/compat/engines/`` (and the YAML parsing layer in
``habit/schemas/``), which several CLI workflows continue to route through:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Role
     - Responsibility
     - Representative symbol
   * - **Config / Schema**
     - Typed representation of configuration and its validation rules.
     - ``BaseConfig``, ``MLConfig``
   * - **Configurator**
     - Assembles validated configuration into an executable object; it does
       not execute the workflow.
     - ``MLConfigurator``
   * - **Orchestrator**
     - Executes a complete workflow through ``run()`` or
       ``fit()``/``predict()``.
     - ``BatchProcessor``, ``HabitatAnalysis``
   * - **Registry / Factory**
     - Maps names to classes or functions and creates the selected algorithm.
     - ``ModelFactory``, ``PreprocessorFactory``
   * - **Contract**
     - Shared interface rules protected by architecture tests.
     - ``ClassRegistry``, ``ORCHESTRATOR_CONTRACT``

.. tip::

   Remember the three most easily confused roles:
   **Configurator assembles, Orchestrator executes, and Registry resolves
   names into objects.** On the v1 path the recipe plays all three roles at
   once, driven by a spec instead of a configurator.

Workflow and Runner
-------------------

The machine-learning subsystem separates orchestration from execution:

* **Workflow**, such as ``HoldoutWorkflow``, decides what should happen:
  orchestration, data splitting, and result organization.
* **Runner**, such as ``HoldoutRunner``, decides how it happens: concrete
  training and inference operations.

This separation allows the two concerns to evolve and be tested independently.

Configuration to execution
--------------------------

The v0.1 subsystems follow the same high-level chain:

.. mermaid::

   flowchart LR
     Y["Configuration"] --> S["Schema<br/>validate"]
     S --> C["Configurator<br/>assemble"]
     C --> O["Orchestrator<br/>execute"]
     R["Registry"] -.->|create by name| O

The v1 chain replaces the middle two stages with a single recipe call:

.. mermaid::

   flowchart LR
     SP["Spec<br/>(immutable)"] --> RE["Recipe<br/>assemble + execute"]
     CT["Cohort / FeatureTable"] --> RE
     RE --> RS["Typed result<br/>+ RunManifest"]
     RG["Domain registry"] -.->|"resolve Spec('name')"| RE

See :doc:`request_lifecycle` for a command-level walkthrough and
:doc:`repo_layout` for the implementation locations.
