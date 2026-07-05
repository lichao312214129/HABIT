设计理念
========

本页回答一个问题：**HABIT 为什么是现在这个样子？** 架构页讲"是什么"，扩展页讲"怎么做"，
而理解这些设计背后的**意图与权衡**，才能让你（以及协助你开发的大模型）做出与整体一致的决策。

从问题出发
----------

生境分析（habitat analysis）这类医学影像研究有几个绕不开的现实痛点：

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - 痛点
     - 含义
   * - **可复现**
     - 论文要求同一份数据、同一套参数能复跑出同一结果；参数散落在脚本里难以复现。
   * - **多受试者批处理**
     - 动辄数百个受试者，需要并行、断点续跑、单例失败不拖垮全局。
   * - **算法要可插拔**
     - 聚类算法、特征、模型、特征选择器经常要换着试，不该每次都改主流程。
   * - **训练/预测必须一致**
     - 推断时的每一步都要和训练时严格相同，否则结果不可信。
   * - **CLI 与 GUI 行为一致**
     - 命令行跑通的东西，图形界面点出来必须是同一套逻辑，不能两套实现。

HABIT 的全部设计，都是为了系统性地消解这五个痛点。

五大设计支柱
------------

.. mermaid::

   flowchart LR
     classDef pain fill:#f9f2f4,stroke:#b85450,stroke-width:2px,color:#000;
     classDef pillar fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px,color:#000;

     P1["Reproducible"]:::pain
     P2["Batch subjects"]:::pain
     P3["Pluggable algos"]:::pain
     P4["Train == Predict"]:::pain
     P5["CLI == GUI"]:::pain

     S1["Config as interface"]:::pillar
     S2["Typed schema<br/>(fail fast)"]:::pillar
     S3["Registry decoupling"]:::pillar
     S4["Unified contracts"]:::pillar
     S5["Thin command,<br/>thick core"]:::pillar

     P1 --> S1 & S2
     P2 --> S5
     P3 --> S3
     P4 --> S4
     P5 --> S5

支柱一 · 配置即接口（Config as Interface）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**做法**：每个工作流的全部参数都写在一份 YAML 里，代码不接受"隐藏的默认行为"。

**解决什么**：一份 YAML 就是一次实验的完整、可版本化的快照——把它交给别人就能复跑。

**权衡**：牺牲了少量灵活性（不能在代码里临时改参数），换来强可复现性与可审计性。

支柱二 · 强类型 Schema，尽早失败（Fail Fast）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**做法**：YAML 解析后立刻用 Pydantic 校验，且根模型设 ``extra='forbid'``——多写一个未知字段就报错。

**解决什么**：把"参数拼错/类型不对"这类错误挡在算法运行**之前**，而不是跑了两小时才崩、或更糟地被静默忽略产出错误结果。

**权衡**：新增字段必须同步声明到 Schema，略增维护成本；换来的是配置错误无处遁形。

支柱三 · 注册表解耦（Registry Decoupling）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**做法**：预处理器、模型、聚类算法、特征选择器等都用装饰器按名注册到全局工厂，主流程只按 YAML 里的名字去取。

**解决什么**：新增一个算法 = 写一个类 + 一行注册装饰器，**不碰主流程**。换算法只改 YAML 里的一个名字。

**为什么不直接用 sklearn 的 Estimator？** 因为 HABIT 要统一管理"名字 ↔ 类 ↔ 参数 Schema ↔ GUI 表单"四者的映射，注册表是承载这层映射的统一入口（详见 :doc:`extension_points`）。

支柱四 · 统一契约（Unified Contracts）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**做法**：所有类式工厂继承同一套 ``ClassRegistry``/``CallableRegistry`` 基类；所有顶层执行引擎（Orchestrator）
遵守固定的终端方法约定（``run()`` 或 ``fit()``/``predict()``）；这些约定由 ``tests/test_architecture_contracts.py`` 自动守护。

**解决什么**：任何子系统"长得都一样"——学会一个就会用全部；训练/预测天然对称（``fit`` 存管线、``predict`` 加载同一条管线），杜绝推断漂移。

**权衡**：约束了实现自由度，换来跨子系统的一致性与可测试性。绝不能破坏的具体规则见 :doc:`invariants`。

支柱五 · 命令薄、核心厚（Thin Command, Thick Core）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**做法**：CLI 命令体只做"加载配置 + 转发"，真正逻辑全在 ``core/*/run.py``。CLI、Web GUI、Python API 三种入口共用同一核心。

**解决什么**：一套逻辑，三种用法，永不分叉。GUI 通过子进程调用同一个 CLI，因此"命令行能跑的，界面点出来一定一样"。

.. mermaid::

   flowchart TD
     classDef entry fill:#f9f2f4,stroke:#b85450,stroke-width:2px,color:#000;
     classDef core fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px,color:#000;

     CLI["CLI<br/>habit &lt;cmd&gt; -c cfg.yaml"]:::entry
     GUI["Web GUI"]:::entry
     PY["Python API"]:::entry
     CORE["core/*/run.py<br/>(the single source of behavior)"]:::core

     CLI --> CORE
     GUI -->|subprocess calls CLI| CLI
     PY --> CORE

一图看懂五者关系
----------------

把五大支柱叠起来，就是 HABIT 那条贯穿始终的主链：

.. mermaid::

   flowchart LR
     classDef a fill:#fff2cc,stroke:#d6b656,stroke-width:2px,color:#000;
     classDef b fill:#ffe6cc,stroke:#d79b00,stroke-width:2px,color:#000;
     classDef c fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000;
     classDef d fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px,color:#000;

     Y["YAML<br/>(config as interface)"]:::a
     V["Pydantic schema<br/>(fail fast)"]:::b
     R["Registry<br/>(pluggable algos)"]:::c
     O["Orchestrator<br/>(unified contract)"]:::d

     Y --> V --> R --> O

.. seealso::

   - 这条主链的机制细节见 :doc:`configuration_system`。
   - 想跟读一次真实运行如何走完全链路，见 :doc:`request_lifecycle`。
   - 每个概念的准确定义见 :doc:`mental_model`。
