总体架构
========

本页给出 HABIT 的整体设计心智模型：分层结构、四条业务主线、统一的数据流，以及贯穿全局的设计理念。

设计理念
--------

HABIT 的核心是一套 **配置驱动（config-driven）+ 工厂/注册表（factory & registry）** 的架构。它把 "算法逻辑" 与 "运行参数" 彻底解耦：

- **配置即接口**：每条工作流都由一份 YAML 描述。研究者改 YAML，不改代码。
- **Schema 校验前置**：YAML 先经 Pydantic Schema 结构化与校验，尽早在运行前暴露配置错误。
- **工厂装配**：算法（预处理器、聚类器、模型、特征选择器等）通过注册表按名字实例化，新增算法即 "注册即可用"。
- **子系统同构**：预处理、生境分割、特征提取、机器学习四大子系统遵循 **同一套模式**，学会一个即可类推其余。
- **CLI / GUI / Python API 三入口共用一套核心**：三者最终都调用 ``habit/core`` 下相同的 ``run_*`` 入口，行为一致。

分层结构
--------

从上到下分为四层：命令入口层、命令实现层、核心业务层、共享工具层。

.. mermaid::

   flowchart TD
     subgraph Entry["Entry points"]
       CLI["CLI: habit/cli.py (Click)"]
       PY["Python API: core run_* functions"]
       GUI["Web GUI: habit-gui (FastAPI + React)"]
     end

     subgraph Cmd["Command layer: habit/commands/cmd_*.py"]
       CMD["Load config, call core, print result"]
     end

     subgraph Core["Core layer: habit/core"]
       PRE["preprocessing"]
       HAB["habitat_analysis"]
       ML["machine_learning"]
       DCM["dicom_sort"]
       COM["common (configs, configurators)"]
       SCH["schemas (workflows, steps, registry)"]
     end

     subgraph Utils["Shared utilities: habit/utils"]
       U["progress, yaml, io, parallel, logging, visualization"]
     end

     CLI --> CMD
     GUI -->|subprocess: habit cli| CMD
     GUI -->|bridge: schema reflect| SCH
     CMD --> Core
     PY --> Core
     PRE --> Utils
     HAB --> Utils
     ML --> Utils
     Core --> COM
     Core --> SCH

各层职责：

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 层
     - 职责
   * - **命令入口层** (``habit/cli.py``)
     - 用 Click 定义所有子命令（``preprocess`` / ``get-habitat`` / ``extract`` / ``model`` …），只做参数解析并转发到命令实现层。
   * - **命令实现层** (``habit/commands/cmd_*.py``)
     - 加载并校验配置、调用核心 ``run_*`` 函数、打印结果与退出码。共享工具在 ``habit/commands/common.py``。
   * - **核心业务层** (``habit/core/*``)
     - 全部业务逻辑。四个业务子系统 + ``common``\ （配置基建）+ ``schemas``\ （Schema 与注册表）。
   * - **共享工具层** (``habit/utils/*``)
     - 跨子系统复用的工具：统一进度条、YAML 读写、并行、日志、绘图等。

四条业务主线
------------

HABIT 的功能围绕一条典型研究流水线展开，每一段都对应一个 CLI 命令和一个核心子系统：

.. mermaid::

   flowchart LR
     A["Raw images / DICOM"] --> B["Preprocessing<br/>habit preprocess"]
     B --> C["Habitat segmentation<br/>habit get-habitat"]
     C --> D["Feature extraction<br/>habit extract / radiomics"]
     D --> E["Machine learning<br/>habit model / cv / compare"]
     E --> F["Models, metrics, plots, reports"]

.. list-table::
   :header-rows: 1
   :widths: 20 22 32 26

   * - 阶段
     - CLI 命令
     - 核心入口（``habit/core/.../run.py``）
     - 编排核心
   * - 预处理
     - ``habit preprocess``
     - ``run_preprocess_from_config``
     - ``BatchProcessor``
   * - 生境分割
     - ``habit get-habitat``
     - ``run_habitat_analysis_from_config``
     - ``HabitatAnalysis``
   * - 特征提取
     - ``habit extract`` / ``habit radiomics``
     - ``run_feature_extraction_from_config`` / ``run_radiomics_from_config``
     - ``HabitatMapAnalyzer`` / ``TraditionalRadiomicsExtractor``
   * - 机器学习
     - ``habit model`` / ``cv`` / ``compare``
     - ``run_ml_from_config`` / ``run_kfold_from_config`` / ``run_model_comparison_from_config``
     - ``HoldoutWorkflow`` / ``KFoldWorkflow`` / ``ModelComparison``

除主线外，还有若干辅助命令：``sort-dicom``\ （DICOM 排序）、``icc``\ （组内相关系数）、``retest``\ （重测信度）、``dice``、``dicom-info``、``merge-csv``、``gui``。

统一数据流
----------

四大子系统虽然算法不同，但都遵循 **同一条装配链路**。理解这一条，就理解了整个 HABIT 的运行方式：

.. mermaid::

   flowchart LR
     Y["YAML config file"] --> L["load_config + path resolve<br/>(common/configs)"]
     L --> S["Pydantic workflow schema<br/>(schemas/workflows)"]
     S --> V["step params validation<br/>(schemas/validation + registry)"]
     V --> CFG["Domain Configurator<br/>(preprocessing / habitat / ml)"]
     CFG --> OBJ["Runtime object<br/>(BatchProcessor / HabitatAnalysis / Workflow)"]
     OBJ --> RUN["run() / fit() / transform()"]
     RUN --> OUT["Outputs on disk"]

     subgraph Factories["Runtime factories & registries"]
       F["PreprocessorFactory / ClusteringFactory<br/>ModelFactory / selector registry ..."]
     end
     OBJ -.->|instantiate algorithms by name| Factories

要点：

1. **配置加载与路径解析**：``habit/core/common/configs/`` 负责读取 YAML、把相对路径解析为绝对路径。
2. **结构化为 Schema**：``habit/core/schemas/workflows/`` 定义整份配置的 Pydantic 根模型。
3. **步骤参数校验**：``habit/core/schemas/`` 的注册表把每个步骤名映射到其 ``params`` 模型并逐一校验。
4. **领域 Configurator 装配**：各子系统的 ``configurator.py`` 把校验后的配置装配成可执行对象。
5. **工厂按名实例化算法**：运行时通过注册表把配置里的 ``method`` / 模型名变成具体算法实例。

.. seealso::

   - 配置系统的完整机制见 :doc:`configuration_system`。
   - 所有工厂/注册表的清单见 :doc:`extension_points`。
   - 目录与文件定位见 :doc:`repo_layout`。

训练 / 预测的对称性
-------------------

生境分割与机器学习都区分 **train** 与 **predict** 两种模式，并共享同一套配置与代码路径：

- **train**：拟合流水线（聚类中心、标准化参数、模型权重…）并 **序列化保存**\ （如生境的 ``habitat_pipeline.pkl``、ML 的模型文件）。
- **predict**：加载已保存的流水线，对新数据 ``transform`` / ``predict``，保证与训练阶段完全一致的处理。

这种对称设计使 "论文里训练好的模型" 可以稳定地复用到新队列上。生境分割三种聚类策略下的具体步骤见 :doc:`subsystems`。
