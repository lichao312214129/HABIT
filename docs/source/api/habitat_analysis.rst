habitat_analysis 模块
========================

.. automodule:: habit.core.habitat_analysis
   :members:
   :undoc-members:
   :show-inheritance:

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

分析策略 (Analysis Strategies)
-------------------------------

不同的策略决定了如何从 ROI 中提取生境特征。

.. automodule:: habit.core.habitat_analysis.strategies.two_step_strategy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.habitat_analysis.strategies.one_step_strategy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.habitat_analysis.strategies.direct_pooling_strategy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.habitat_analysis.strategies.base_strategy
   :members:
   :undoc-members:
   :show-inheritance:

流程管理器 (Managers)
----------------------

这些管理器负责协调具体的分析步骤，如特征提取、聚类和结果汇总。

.. automodule:: habit.core.habitat_analysis.managers.feature_manager
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.habitat_analysis.managers.clustering_manager
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.habitat_analysis.managers.result_manager
   :members:
   :undoc-members:
   :show-inheritance:

分析器与提取器 (Analyzers & Extractors)
----------------------------------------

.. automodule:: habit.core.habitat_analysis.analyzers.habitat_analyzer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.habitat_analysis.extractors.voxel_radiomics_extractor
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.habitat_analysis.extractors.supervoxel_radiomics_extractor
   :members:
   :undoc-members:
   :show-inheritance:
