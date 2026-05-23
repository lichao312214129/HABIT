habitat_analysis 模块
========================

.. automodule:: habit.core.habitat_analysis
   :no-members:

核心分析类 (Core Analysis)
---------------------------

`HabitatAnalysis` 是执行生境分析的主要入口类。

.. automodule:: habit.core.habitat_analysis.habitat_analysis
   :members:
   :undoc-members:
   :show-inheritance:

配置管理 (Configuration)
-------------------------

这些类定义了生境分析的配置结构，了解它们对于自定义分析流程至关重要。

.. automodule:: habit.core.habitat_analysis.config_schemas
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.habitat_analysis.configurator
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline 与步骤 (Pipeline & Steps)
-----------------------------------

V1 已删除旧的 ``strategies/`` 子包；``clustering_mode`` 的分发集中在
``HabitatAnalysis`` 内部的 recipe 字典。开发者应从 pipeline 基类和具体 step
理解当前执行结构。

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

领域服务 (Domain services)
---------------------------

``HabitatConfigurator`` 创建下列实现类并注入 ``HabitatAnalysis`` ，再由
``HabitatAnalysis`` 在 ``predict`` 路径写入 ``HabitatPipeline`` 各步骤
（白名单 ``_PIPELINE_SERVICE_ATTRS``）。

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

分析器与提取器 (Analyzers & Extractors)
----------------------------------------

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
