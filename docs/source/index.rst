HABIT: Habitat Analysis: Biomedical Imaging Toolkit
==================================================

.. image:: https://img.shields.io/badge/Python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://opensource.org/licenses/MIT

**HABIT** 是一个综合性的肿瘤"生境"（Habitat）分析工具包，旨在通过医疗影像挖掘肿瘤内部的异质性。它涵盖了从影像预处理、特征提取、聚类分析到机器学习建模的完整放射组学（Radiomics）流水线。

本项目的核心开发者为 **黎超** 和 **董梦实** (Core developers: Li Chao, Dong Mengshi)。

核心工作流
----------

HABIT 识别并表征具有不同影像表型的肿瘤亚区域（即 Habitat）。

**影像输入 → 像素级特征提取 → 超像素分割 (可选) → Habitat 聚类 → Habitat 特征提取 → 临床预测模型 (可选)**

主要功能
--------

*   **影像预处理**: DICOM 转换、重采样、配准、归一化、N4 偏置场校正。
*   **特征工程**: 提供像素级、超像素级、Habitat 级的特征提取。
*   **聚类策略**: 支持 One-Step (个性化)、Two-Step (群组一致性) 和 Direct Pooling 三种策略。
*   **机器学习**: 内置特征选择、模型训练（XGBoost, AutoGluon 等）及验证体系。
*   **统计验证**: 提供 ICC 分析、Test-Retest 重复性验证及可视化工具。

.. toctree::
   :maxdepth: 2
   :caption: 开始使用
   :hidden:

   getting_started/index_zh
   data_structure_zh

.. toctree::
   :maxdepth: 2
   :caption: 用户指南
   :hidden:

   user_guide/index_zh

.. toctree::
   :maxdepth: 1
   :caption: 实用工具
   :hidden:

   app_dicom_info_zh
   app_icc_analysis_zh
   app_merge_csv_zh
   app_habitat_test_retest_zh
   app_model_comparison_zh
   app_dice_calculator_zh

.. toctree::
   :maxdepth: 2
   :caption: 开发指南
   :hidden:

   import_robustness_guide_zh
   python_subprocess_methods_zh

.. toctree::
   :maxdepth: 2
   :caption: 设计哲学
   :hidden:

   design_philosophy_zh

.. toctree::
   :maxdepth: 2
   :caption: 自定义扩展
   :hidden:

   customization/index_zh

.. toctree::
   :maxdepth: 2
   :caption: 配置参考
   :hidden:

   configuration_zh

.. toctree::
   :maxdepth: 2
   :caption: 命令行工具
   :hidden:

   cli_zh

.. toctree::
   :maxdepth: 2
   :caption: API 参考
   :hidden:

   api/index

.. toctree::
   :maxdepth: 2
   :caption: 开发与架构
   :hidden:

   development/index
   development/metrics_optimization
   algorithms/index
   changelog
   acknowledgments

索引与表格
----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

项目链接
--------

*   **GitHub**: https://github.com/lichao312214129/HABIT
*   **文档**: https://habit-docs.readthedocs.io/ (示例)
