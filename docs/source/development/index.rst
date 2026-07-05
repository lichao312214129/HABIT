开发者指南
==========

面向 HABIT 的贡献者与二次开发者。仅想使用软件的研究者请从 :doc:`../tutorial/installation` 开始。

本章帮助你理解 HABIT 的整体设计、代码组织、配置驱动机制与扩展方式，并给出贡献流程。

.. toctree::
   :maxdepth: 2

   architecture
   repo_layout
   configuration_system
   extension_points
   subsystems
   dev_workflow
   contributing

阅读顺序建议
------------

1. :doc:`architecture` —— 先建立整体心智模型（分层、四条主线、数据流、跨子系统契约图）。
2. :doc:`repo_layout` —— 知道每类代码放在哪、如何按功能定位文件。
3. :doc:`configuration_system` —— 理解 "YAML → Schema → Configurator → 运行时对象" 这条主链。
4. :doc:`extension_points` —— 所有可扩展点（工厂/注册表）一览。
5. :doc:`subsystems` —— 深入生境分析与机器学习子系统。
6. :doc:`dev_workflow` —— 环境搭建、测试、如何新增命令 / 配置项 / 组件。

.. note::

   "如何编写自定义组件" 的完整代码模板见 :doc:`../customization/index`。
   本章侧重 **架构与机制**，扩展章侧重 **动手模板**，两者互补。

环境速览：Python 3.10 + ``pip install -e ".[dev]"`` + ``pytest tests/``。
API 与第三方库参考见左侧 **Developer** 侧边栏。
