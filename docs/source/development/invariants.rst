不变量与架构契约
================

本页列出 HABIT 中**绝不能破坏的规则**——它们是保证结果可信、代码一致的护栏。
无论是你手改还是让大模型改代码，改动都必须守住这些不变量；其中大部分由
``tests/test_architecture_contracts.py`` 自动检查。

.. important::

   把这一页当作"红线清单"。提交前跑一遍架构契约测试：

   .. code-block:: bash

      pytest tests/test_architecture_contracts.py -m unit

科学正确性类
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 不变量
     - 为什么
   * - **防数据穿越**
     - 特征选择、标准化、重采样必须封进**一条** sklearn Pipeline，在交叉验证里只在训练折上
       ``fit``。任何"先在全量数据上选特征再划分"都是穿越，会高估性能。落点：``PipelineBuilder``。
   * - **训练/预测严格对称**
     - ``fit`` 阶段固化的处理（聚类中心、scaler 参数、选中特征等）必须在 ``predict`` 阶段原样复用。
       生境侧靠序列化 ``habitat_pipeline.pkl`` 保证；ML 侧靠同一条 Pipeline 保证。
   * - **随机性可控**
     - 涉及随机的步骤必须接受并透传随机种子，保证同配置可复现。

配置类
------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 不变量
     - 为什么
   * - **``extra='forbid'``**
     - 所有配置根模型继承 ``BaseConfig``，禁止未知字段。新增字段**必须**同步声明到 Schema，
       否则用户写该字段会报"未知字段"。这是"尽早失败"的基石（见 :doc:`philosophy`）。
   * - **Schema 是唯一事实来源**
     - 字段定义只写在 ``habit/core/schemas/``。各子系统的 ``config_schemas.py`` 仅做 re-export，
       不得在别处另立定义。
   * - **有参数就配 Schema**
     - 新组件若带可配置 ``params``，必须定义 Pydantic 参数模型并注册到 ``ParamSchemaRegistry``，
       从而同时获得类型校验与 GUI 表单。

架构契约类（由测试自动守护）
----------------------------

这些规则写死在 ``tests/test_architecture_contracts.py`` 里，破坏即红：

.. mermaid::

   flowchart TD
     classDef t fill:#f9f2f4,stroke:#b85450,stroke-width:2px,color:#000;
     classDef r fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000;
     classDef o fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px,color:#000;

     TST["test_architecture_contracts.py"]:::t
     TST --> C1["all registries subclass _BaseRegistry"]:::r
     TST --> C2["class factories add create();<br/>callable ones add entries()"]:::r
     TST --> C3["uniform surface:<br/>register/get/available/<br/>register_params_model/get_params_model"]:::r
     TST --> C4["registries do NOT share _registry dict"]:::r
     TST --> C5["every orchestrator exposes its<br/>terminal method(s): run OR fit+predict"]:::o

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 不变量
     - 说明
   * - **统一注册表基类**
     - 每个注册表都继承 ``_BaseRegistry``；类式工厂再继承 ``ClassRegistry``（提供 ``create()``），
       函数式注册表继承 ``CallableRegistry``（提供 ``get_entry()`` / ``entries()``）。
   * - **统一注册表接口**
     - 每个注册表都暴露 ``register`` / ``get`` / ``available`` / ``register_params_model`` /
       ``get_params_model``；``available()`` 返回 list。
   * - **注册表互不共享存储**
     - 不同注册表必须各自持有独立的 ``_registry`` 字典，绝不能共用一个 dict。
   * - **编排器终端方法固定**
     - 每个顶层编排器必须暴露 ``ORCHESTRATOR_CONTRACT`` 中为它声明的终端方法：
       批处理/工作流用 ``run()``，生境分析用 ``fit()`` + ``predict()``。

当前受契约测试覆盖的组件
~~~~~~~~~~~~~~~~~~~~~~~~

新增同类组件后，需把它登记进对应清单，测试才会保护它：

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - 类别
     - 成员
     - 登记位置
   * - 类式工厂
     - ``PreprocessorFactory`` / ``ModelFactory`` / ``ClusteringAlgorithmFactory`` /
       ``FeatureExtractorRegistry`` / ``PreprocessingMethodFactory`` / ``HabitatFeatureRegistry``
     - 测试文件的 ``CLASS_REGISTRIES``
   * - 函数式注册表
     - ``SelectorRegistry`` / ``MetricRegistry``
     - 测试文件的 ``CALLABLE_REGISTRIES``
   * - 编排器
     - ``BatchProcessor`` / ``HabitatAnalysis`` / ``HabitatMapAnalyzer`` /
       ``TraditionalRadiomicsExtractor`` / ``HoldoutWorkflow`` / ``KFoldWorkflow`` / ``ModelComparison``
     - ``common/orchestrator.py`` 的 ``ORCHESTRATOR_CONTRACT``

工程约定类
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 不变量
     - 为什么
   * - **进度条统一**
     - 一律用 ``habit/utils/progress_utils.py``，禁止各处自造 tqdm。
   * - **工具集中**
     - 跨子系统工具统一放 ``habit/utils/``，不在子系统里重复造轮子。
   * - **图内一律英文**
     - 所有由代码生成的图（ROC、校准、生境图例等）文字必须英文（文档正文不受此限）。
   * - **延迟导入**
     - CLI 命令体、Configurator 工厂方法内才导入重型依赖，保持启动快、可选依赖不缺就不崩。
   * - **命令薄、核心厚**
     - 业务逻辑放 ``core/*/run.py``，命令层只做加载配置 + 转发 + 异常收敛。
   * - **类型注解 + 英文注释**
     - 函数显式标注输入/输出类型；注释用英文并讲清意图。

.. seealso::

   - 这些护栏背后的设计意图见 :doc:`philosophy`。
   - 新增组件/编排器的完整步骤见 :doc:`extension_points` 与 :doc:`dev_workflow`。
