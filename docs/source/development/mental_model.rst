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
   names into objects.**

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

Every subsystem follows the same high-level chain:

.. mermaid::

   flowchart LR
     Y["Configuration"] --> S["Schema<br/>validate"]
     S --> C["Configurator<br/>assemble"]
     C --> O["Orchestrator<br/>execute"]
     R["Registry"] -.->|create by name| O

See :doc:`request_lifecycle` for a command-level walkthrough and
:doc:`repo_layout` for the implementation locations.
