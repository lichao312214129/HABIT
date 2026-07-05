总体架构
========

本页从底层代码实现的角度，剖析 HABIT 的整体架构设计。HABIT 的核心是一套 **配置驱动（Config-driven）** 、 **延迟装配（Lazy Assembly）** 与 **契约化运行（Contract-based Execution）** 相结合的系统，旨在将算法逻辑与运行参数解耦，并提供与 scikit-learn 类似但更适合多受试者医学影像工作流的统一契约。

核心设计原则
------------

1. **配置即接口**：所有工作流均由 YAML 描述。
2. **Schema 强类型校验**：YAML 解析后，立即通过 Pydantic 转换为强类型 Schema。参数拼写错误、类型不匹配在实例化前就会被拦截。
3. **注册表模式（Registry Pattern）**：算法组件（预处理器、模型、特征选择器等）不硬编码在主流程中，而是通过装饰器按名注册。
4. **统一契约（Unified Contracts）**：
   * 所有基于类的组件工厂继承自 ``habit.core.common.registry.ClassRegistry``。
   * 所有基于函数的组件工厂继承自 ``habit.core.common.registry.CallableRegistry``。
   * 所有执行引擎（Orchestrators）遵循固定的终端方法约定（``run()`` 或 ``fit()``/``predict()``）。

分层架构
--------

HABIT 的代码结构自上而下分为四层：

.. mermaid::

   flowchart TD
     classDef ui fill:#f9f2f4,stroke:#b85450,stroke-width:2px;
     classDef core fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px;
     classDef util fill:#d5e8d4,stroke:#82b366,stroke-width:2px;

     subgraph Entry["1. Entry Points"]
       CLI["CLI (habit/cli.py)"]:::ui
       PY["Python API (habit/core/*/run.py)"]:::ui
       GUI["Web GUI (FastAPI)"]:::ui
     end

     subgraph Cmd["2. Command Execution"]
       CMD["habit/commands/cmd_*.py<br/>(Load YAML -> Call Core)"]:::core
     end

     subgraph Core["3. Core Business Logic (habit/core/)"]
       subgraph Infra["Infrastructure (common/ & schemas/)"]
         CFG["Config Loader"]:::core
         REG["ClassRegistry / CallableRegistry"]:::core
         SCH["ParamSchemaRegistry (Pydantic)"]:::core
       end
       subgraph Domain["Domain Subsystems"]
         PRE["preprocessing/"]:::core
         HAB["habitat_analysis/"]:::core
         ML["machine_learning/"]:::core
       end
     end

     subgraph Utils["4. Shared Utilities (habit/utils/)"]
       UTILS["Logging, Parallel, Random State, IO"]:::util
     end

     CLI --> CMD
     GUI --> CMD
     PY --> Domain
     CMD --> Infra
     Infra --> Domain
     Domain --> Utils

统一数据流与对象装配链路
------------------------

HABIT 最核心的机制是**“配置如何转化为可执行的算法对象”**。以下链路贯穿了预处理、生境分割和机器学习四大子系统：

.. mermaid::

   flowchart TD
     classDef file fill:#fff2cc,stroke:#d6b656,stroke-width:2px;
     classDef schema fill:#ffe6cc,stroke:#d79b00,stroke-width:2px;
     classDef factory fill:#d5e8d4,stroke:#82b366,stroke-width:2px;
     classDef run fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px;

     Y["1. YAML Config"]:::file --> L["Config Loader"]:::run
     L --> S["2. Pydantic Workflow Schema"]:::schema
     S --> V["3. ParamSchemaRegistry<br/>(Validates specific step params)"]:::schema
     V --> CFG["4. Domain Configurator"]:::run
     CFG --> FAC["5. Factory/Registry<br/>(e.g., ModelFactory)"]:::factory
     FAC -.->|"create(name, params)"| ALG["6. Algorithm Instance"]:::factory
     CFG --> OBJ["7. Orchestrator<br/>(e.g., KFoldWorkflow)"]:::run
     OBJ --> RUN["8. Execution<br/>(run / fit / predict)"]:::run

**关键组件解析：**

1. **ParamSchemaRegistry** (``habit/core/schemas/registry.py``)：
   这是 HABIT 校验机制的核心。它将 YAML 中的 ``method`` 名字映射到对应的 Pydantic 模型。
2. **Domain Configurator** (如 ``MLConfigurator``)：
   负责将校验后的配置转化为可执行的 Orchestrator，并巧妙地实现了**延迟导入**（避免了在不需要时加载 torch/sklearn 等重型依赖）。
3. **ClassRegistry / CallableRegistry** (``habit/core/common/registry.py``)：
   底层注册表实现。提供统一的 ``register()``、``create()`` 和 ``get()`` 接口。
4. **Orchestrator**：
   组装完毕的顶层执行引擎。在机器学习模块中，它可能是 ``HoldoutWorkflow``；在预处理模块中，它是 ``BatchProcessor``。

核心子系统深度剖析
------------------

1. 预处理 (Preprocessing)
^^^^^^^^^^^^^^^^^^^^^^^^^

* **核心引擎**：``BatchProcessor``。
* **执行机制**：基于多进程（由 ``processes`` 参数控制），对每个受试者按配置顺序依次调用 ``PreprocessorFactory`` 中的插件（如 resample, registration）。支持 ``save_intermediate`` 保存中间态 NIfTI。

2. 生境分析 (Habitat Analysis)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

这是 HABIT 最复杂的子系统，采用 **Pipeline + Step + Service** 三层结构。

* **数据载荷**：定义了显式的 ``HabitatSubjectData``，在 Pipeline 的各个 Step 之间显式传递特征和掩膜信息。
* **聚类策略**：底层支持三种 ``clustering_mode``：
  * ``two_step``（默认）：个体体素聚类 -> 超体素 -> 合并所有受试者 -> 群体超体素聚类 -> 生境。
  * ``one_step``：个体体素直接聚成生境（常用于单肿瘤探索）。
  * ``direct_pooling``：跨受试者体素池化后统一聚类。
* **Train/Predict 严格对称**：``fit()`` 阶段构建并拟合完整的 ``HabitatPipeline``，然后序列化为 ``habitat_pipeline.pkl``；``predict()`` 阶段直接加载该 pkl 并注入 Service 执行 ``transform``，保证了推断时的绝对一致性。

3. 机器学习 (Machine Learning)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **编排与执行解耦**：Workflow（如 ``HoldoutWorkflow``, ``KFoldWorkflow``）负责流程编排与数据划分，Runner（如 ``HoldoutRunner``）负责具体执行。
* **防数据穿越设计**：``PipelineBuilder`` 负责将特征选择（``selector_before/after``）、重采样（``resampling``）和模型（``model``）严格封装为单条 ``sklearn.pipeline.Pipeline``。在交叉验证中，该 Pipeline 确保特征选择仅在训练折上进行，彻底杜绝数据穿越。

CLI 与 Core 映射关系
--------------------

CLI 命令通过薄封装将控制权移交给 Core 层的 Python API：

.. list-table::
   :widths: 15 20 25 25 15
   :header-rows: 1

   * - CLI 命令
     - 命令模块
     - 配置 Schema
     - Core 入口函数
     - Orchestrator
   * - ``preprocess``
     - ``cmd_preprocess.py``
     - ``PreprocessingConfig``
     - ``run_preprocess_from_config``
     - ``BatchProcessor``
   * - ``get-habitat``
     - ``cmd_habitat.py``
     - ``HabitatAnalysisConfig``
     - ``run_habitat_analysis_from_config``
     - ``HabitatAnalysis``
   * - ``extract``
     - ``cmd_extract_features.py``
     - ``FeatureExtractionConfig``
     - ``run_feature_extraction_from_config``
     - ``HabitatMapAnalyzer``
   * - ``model``
     - ``cmd_ml.py``
     - ``MLConfig``
     - ``run_ml_from_config``
     - ``HoldoutWorkflow``
   * - ``cv``
     - ``cmd_ml.py``
     - ``MLConfig``
     - ``run_kfold_from_config``
     - ``KFoldWorkflow``