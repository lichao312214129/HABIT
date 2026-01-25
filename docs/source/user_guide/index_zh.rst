用户指南
========

本指南按照生境分析的逻辑顺序组织，帮助您从数据准备到模型部署完成整个工作流程。

.. toctree::
   :maxdepth: 2

   roi_preparation_zh
   image_preprocessing_zh
   habitat_segmentation_zh
   habitat_feature_extraction_zh
   machine_learning_modeling_zh

实用工具
--------

HABIT 提供了一些辅助工具来简化您的工作流程：

.. toctree::
   :maxdepth: 1

   ../app_dicom_info_zh
   ../app_model_comparison_zh
   ../app_dice_calculator_zh
   ../app_habitat_test_retest_zh

工作流程概述
------------

HABIT 的完整工作流程包括以下步骤：

1. **数据准备**: 准备医学图像数据和 ROI 掩码
2. **图像预处理**: 对原始图像进行预处理（重采样、配准、标准化等）
3. **生境分割**: 使用聚类算法将肿瘤分割为多个生境
4. **生境特征提取**: 从生境图中提取各种特征
5. **机器学习建模**: 使用提取的特征进行机器学习建模

**自定义扩展**

HABIT 支持高度的自定义扩展，您可以：

- **自定义预处理器**: 添加自定义的图像预处理方法
- **自定义特征提取器**: 添加自定义的聚类特征提取方法
- **自定义聚类算法**: 添加自定义的聚类算法
- **自定义策略**: 添加自定义的生境分割策略
- **自定义模型**: 添加自定义的机器学习模型
- **自定义特征选择器**: 添加自定义的特征选择方法

参考 :doc:`../customization/index_zh` 了解详细的扩展指南。

**配置文件**

HABIT 使用 YAML 配置文件来控制所有参数。每个步骤都有对应的配置文件模板：

- **预处理配置**: `config_preprocessing.yaml`
- **生境分析配置**: `config_habitat.yaml`
- **特征提取配置**: `config_extract_features.yaml`
- **机器学习配置**: `config_machine_learning.yaml`

参考 :doc:`../configuration_zh` 了解配置文件的详细说明。
