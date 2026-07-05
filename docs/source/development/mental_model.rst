核心概念与心智模型
==================

读 HABIT 代码前，先建立一套共同语言。本页给出**术语表**和**一张全局心智地图**，
之后所有开发者文档都沿用这里的词汇。

一张图：全局心智地图
--------------------

HABIT 的世界可以分成"业务概念"（研究者关心的东西）和"工程角色"（代码里的抽象）两层，
它们通过配置主链连接：

.. mermaid::

   flowchart TD
     classDef dom fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000;
     classDef eng fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px,color:#000;

     subgraph Domain["Domain concepts (what researchers care about)"]
       V["Voxel feature"]:::dom --> SV["Supervoxel"]:::dom --> H["Habitat"]:::dom --> F["Habitat feature"]:::dom --> M["ML model"]:::dom
     end

     subgraph Eng["Engineering roles (how the code is organized)"]
       CFG["Configurator<br/>assembles"]:::eng --> ORC["Orchestrator<br/>executes"]:::eng
       REG["Registry<br/>names -> classes"]:::eng --> ORC
       CON["Contract<br/>the rules all must obey"]:::eng -.-> ORC
     end

     Eng -->|produces| Domain

业务概念
--------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 术语
     - 含义
   * - **Voxel（体素）**
     - 影像的最小单元（3D 像素）。生境分析从体素级特征出发。
   * - **Voxel feature（体素特征）**
     - 每个体素上算出的特征向量（原始强度、动力学、局部 radiomics 等）。
   * - **Supervoxel（超体素）**
     - 单个受试者内，把相似体素聚成的局部小块（``two_step`` 策略的中间产物）。
   * - **Habitat（生境）**
     - 肿瘤内部影像表型相近的子区域，是本工具箱的核心产物；生境图是一张整数标签影像。
   * - **Habitat feature（生境特征）**
     - 生境图生成**之后**在其上计算的下游特征（传统 radiomics、MSI、ITH 等），用于后续建模。
   * - **Clustering mode（聚类策略）**
     - 体素如何聚合成生境的三种路线：``two_step`` / ``one_step`` / ``direct_pooling``\ （见 :doc:`subsystems`）。

工程角色
--------

这几个词是读代码时最常撞见的抽象，务必分清：

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - 角色
     - 职责（一句话）
     - 代表符号
   * - **Config / Schema**
     - 一份 YAML 的强类型表示，负责校验。
     - ``BaseConfig`` ``MLConfig``
   * - **Configurator（装配器）**
     - 把校验后的配置**装配**成一个可执行对象。它只组装，不执行。
     - ``MLConfigurator``
   * - **Orchestrator（编排器）**
     - 顶层**执行**引擎，跑完整个工作流。终端方法固定为 ``run()`` 或 ``fit()``/``predict()``。
     - ``BatchProcessor`` ``HabitatAnalysis`` ``HoldoutWorkflow``
   * - **Registry / Factory（注册表/工厂）**
     - 维护"名字 → 类/函数"的映射，按 YAML 里的名字实例化具体算法。
     - ``ModelFactory`` ``PreprocessorFactory``
   * - **Contract（契约）**
     - 所有工厂、编排器必须遵守的统一接口约定，由架构契约测试守护。
     - ``ClassRegistry`` ``ORCHESTRATOR_CONTRACT``

.. tip::

   一句话记住三个最易混的角色：**Configurator 组装、Orchestrator 执行、Registry 查名建对象。**

Workflow 与 Runner（机器学习专属）
----------------------------------

机器学习子系统里还有一对容易混淆的角色：

- **Workflow**\ （如 ``HoldoutWorkflow``）：负责"做什么"——编排流程、划分数据、组织产出结构。
- **Runner**\ （如 ``HoldoutRunner``）：负责"怎么做"——具体的训练/推理执行细节。

分离的目的：编排逻辑与执行细节各自独立演化、独立测试（详见 :doc:`subsystems`）。

一份配置如何变成一次运行
------------------------

把上面的角色按时间顺序串起来，就是每个子系统都遵循的同一条主链：

.. mermaid::

   flowchart LR
     classDef a fill:#fff2cc,stroke:#d6b656,stroke-width:2px,color:#000;
     classDef b fill:#ffe6cc,stroke:#d79b00,stroke-width:2px,color:#000;
     classDef c fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000;
     classDef d fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px,color:#000;

     Y["YAML"]:::a --> S["Schema<br/>(validate)"]:::b
     S --> C["Configurator<br/>(assemble)"]:::c
     C --> O["Orchestrator<br/>(execute)"]:::d
     R["Registry"]:::c -.->|create by name| O

.. seealso::

   - 想看这条链在真实命令里逐行怎么走，见 :doc:`request_lifecycle`。
   - 这些角色的代码落点见 :doc:`repo_layout`；机制细节见 :doc:`configuration_system`。
