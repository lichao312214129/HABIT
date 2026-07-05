开发者指南
==========

面向 HABIT 的贡献者与二次开发者，也是从**高层与架构**深入理解 HABIT 的入口。
仅想使用软件的研究者请从 :doc:`../tutorial/installation` 开始。

本章按"理解 → 定位 → 扩展"三段式组织，帮你先建立心智模型，再学会在代码里找东西，
最后动手扩展与贡献。

.. toctree::
   :maxdepth: 2
   :caption: 一、理解 HABIT（Why + 心智模型）

   philosophy
   mental_model
   architecture
   request_lifecycle

.. toctree::
   :maxdepth: 2
   :caption: 二、代码组织（在哪找）

   repo_layout
   configuration_system
   subsystems

.. toctree::
   :maxdepth: 2
   :caption: 三、扩展与贡献（怎么做）

   extension_points
   invariants
   dev_workflow
   contributing

阅读路线建议
------------

**第一次接触 HABIT（建立心智模型）**

1. :doc:`philosophy` —— HABIT 为什么这样设计：五大设计支柱与权衡。
2. :doc:`mental_model` —— 核心概念术语表与一张全局心智地图。
3. :doc:`architecture` —— 分层结构、配置装配主链、CLI↔Core 映射。
4. :doc:`request_lifecycle` —— 跟读一条真实命令走完全链路（理解代码最有效的一页）。

**要动手改代码（定位与落地）**

5. :doc:`repo_layout` —— 每类代码放在哪、"改 X 去哪找"。
6. :doc:`configuration_system` —— "YAML → Schema → Configurator → 运行时对象"主链细节。
7. :doc:`subsystems` —— 生境分析与机器学习子系统的内部执行流。

**要扩展或提交贡献**

8. :doc:`extension_points` —— 全部 8 个扩展点（注册表/工厂）一览与实战。
9. :doc:`invariants` —— **绝不能破坏的规则**，改代码前必读的护栏清单。
10. :doc:`dev_workflow` —— 环境、测试、新增命令 / 配置项 / 组件的落地步骤。
11. :doc:`contributing` —— PR 与代码风格规范。

.. note::

   本章侧重 **架构与机制**；完整可复制的组件代码模板见 :doc:`../customization/index`，两者互补。

环境速览：Python 3.10 + ``pip install -e ".[dev]"`` + ``pytest tests/``。
API 与第三方库参考见左侧 **Developer** 侧边栏。
