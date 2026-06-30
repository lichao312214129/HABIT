habitat_analysis module
=========================

.. automodule:: habit.core.habitat_analysis
   :no-members:

Core analysis
-------------

``HabitatAnalysis`` is the main entry class for habitat analysis.

.. automodule:: habit.core.habitat_analysis.habitat_analysis
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

These classes define habitat analysis config structures; essential for customizing pipelines.

.. automodule:: habit.core.habitat_analysis.config_schemas
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.habitat_analysis.configurator
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline and steps
------------------

V1 removed the legacy ``strategies/`` subpackage; ``clustering_mode`` dispatch lives in a recipe dict inside ``HabitatAnalysis``. Read the pipeline base class and concrete steps for the current structure.

.. automodule:: habit.core.habitat_analysis.pipelines.base_pipeline
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.habitat_analysis.pipelines.habitat_subject_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.habitat_analysis.pipelines.steps
   :members:
   :undoc-members:
   :show-inheritance:

Domain services
---------------

``HabitatConfigurator`` constructs the implementations below and injects them into ``HabitatAnalysis``, which wires them into ``HabitatPipeline`` steps on the ``predict`` path (whitelist ``_PIPELINE_SERVICE_ATTRS``).

.. automodule:: habit.core.habitat_analysis.services.feature_service
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.habitat_analysis.services.clustering_service
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.habitat_analysis.services.habitat_image_writer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.habitat_analysis.services.result_publisher
   :members:
   :undoc-members:
   :show-inheritance:

Analyzers and extractors
------------------------

.. automodule:: habit.core.habitat_analysis.habitat_features.habitat_analyzer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.habitat_analysis.clustering_features.voxel_radiomics_extractor
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.habitat_analysis.clustering_features.supervoxel_radiomics_extractor
   :members:
   :undoc-members:
   :show-inheritance:
