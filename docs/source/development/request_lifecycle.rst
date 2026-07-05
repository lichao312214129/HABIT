一次运行的全生命周期
====================

这是理解 HABIT 最有效的一页：我们跟读**一条真实命令**从你按下回车到产出文件，逐层穿过所有抽象。
读完你就把 :doc:`philosophy` 的理念和 :doc:`mental_model` 的角色，串成了一个完整的故事。

示例命令：

.. code-block:: bash

   habit cv -c config/machine_learning/config_machine_learning_kfold_demo.yaml

它做的事：读一份 ML 配置，跑 K 折交叉验证，输出模型、指标与图表。

全景：七个阶段
--------------

.. mermaid::

   flowchart TD
     classDef entry fill:#f9f2f4,stroke:#b85450,stroke-width:2px,color:#000;
     classDef cmd fill:#ffe6cc,stroke:#d79b00,stroke-width:2px,color:#000;
     classDef cfg fill:#fff2cc,stroke:#d6b656,stroke-width:2px,color:#000;
     classDef asm fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000;
     classDef run fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px,color:#000;

     S1["1. CLI entry<br/>habit/cli.py :: cv()"]:::entry
     S2["2. Command layer<br/>commands/cmd_ml.py :: run_kfold()"]:::cmd
     S3["3. Load + validate<br/>MLConfig.from_file()"]:::cfg
     S4["4. Core API<br/>run_kfold_from_config()"]:::run
     S5["5. Assemble<br/>MLConfigurator.create_kfold_workflow()"]:::asm
     S6["6. Execute<br/>KFoldWorkflow.run() -> KFoldRunner"]:::run
     S7["7. Report<br/>models + metrics + plots"]:::run

     S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7

阶段一 · CLI 入口
-----------------

``habit/cli.py`` 里的 ``cv`` 命令只做两件事：声明它接受 ``-c`` 配置参数，然后**延迟导入**并转发。

.. code-block:: python

   @cli.command('cv')
   @config_option()
   def cv(config):
       """Run K-fold cross-validation for model evaluation"""
       from habit.commands.cmd_ml import run_kfold   # imported only when called
       run_kfold(config)

.. note::

   **为什么在函数体里 import？** 保持 ``habit --help`` 秒开——不会因为某个命令依赖 torch/sklearn
   就在启动时全量加载。这是"命令薄"的一个具体体现（见 :doc:`philosophy`）。

阶段二 · 命令层
---------------

``commands/cmd_ml.py`` 的 ``run_kfold()`` 承担"胶水"职责：加载配置、建输出目录、配日志、调用核心、统一处理异常。

.. code-block:: python

   def run_kfold(config_file: str) -> None:
       config = load_config_or_exit(MLConfig, config_file)     # -> 阶段三
       output_dir = Path(config.output); output_dir.mkdir(parents=True, exist_ok=True)
       logger = setup_logger(name="cli.kfold", output_dir=output_dir,
                             log_filename="kfold_cv.log")
       try:
           run_kfold_from_config(config, logger=logger,        # -> 阶段四
                                 output_dir=str(output_dir))
       except Exception as exc:
           exit_with_error(f"Error: {exc}")
       echo_success("K-fold cross-validation completed successfully!")

要点：日志统一落到 ``{output}/kfold_cv.log``；任何异常都被收敛成一次干净的退出。业务逻辑一行都没有。

阶段三 · 加载与校验（fail fast 在此发生）
-----------------------------------------

``load_config_or_exit(MLConfig, path)`` 内部走的是配置主链的前半段：

.. mermaid::

   flowchart LR
     Y["YAML file"] --> L["load_config()<br/>read + PathResolver"]
     L --> P["MLConfig(**data)<br/>Pydantic validate"]
     P -->|ok| OK["typed MLConfig object"]
     P -->|bad field/type| ERR["ConfigValidationError<br/>(stop before any compute)"]

- 相对路径按 YAML 所在目录解析为绝对路径（``PathResolver``）。
- ``MLConfig`` 继承 ``BaseConfig``（``extra='forbid'``），拼错字段当场报错。
- 每个步骤的 ``params`` 还会经 ``validate_step_params()`` 按 ``ParamSchemaRegistry`` 逐一校验。

这一步跑完，后面拿到的就是一个**保证合法**的强类型配置对象。机制细节见 :doc:`configuration_system`。

阶段四 · 核心 API
-----------------

``run_kfold_from_config()`` 是 Python API 用户也能直接调用的入口，它是"薄命令、厚核心"的边界：

.. code-block:: python

   def run_kfold_from_config(config, *, logger=None, output_dir=None):
       if config.run_mode != "train":
           raise ValueError("K-fold cross-validation requires run_mode='train'.")
       configurator = MLConfigurator(config=config, logger=log, output_dir=out)  # 阶段五
       workflow = configurator.create_kfold_workflow()                           # 阶段五
       workflow.run()                                                            # 阶段六

短短几行，正好体现主链后半段：**Configurator 装配 → Orchestrator 执行**。

阶段五 · 装配
-------------

``MLConfigurator``（继承 ``BaseConfigurator``）把校验后的配置翻译成一个可执行的 Orchestrator。
``create_kfold_workflow()`` 在此按配置里的名字，通过各个 Registry 备好模型、特征选择器、评估器等组件，
组装出一个 ``KFoldWorkflow``。**装配器只组装，不执行。**

阶段六 · 执行（Workflow → Runner）
----------------------------------

``KFoldWorkflow.run()`` 负责编排，真正的每折训练交给 Runner：

.. mermaid::

   flowchart TD
     WF["KFoldWorkflow.run()<br/>(what to do)"] --> PLAN["WorkflowPlan<br/>(frozen config snapshot)"]
     PLAN --> RUN["KFoldRunner<br/>(how to do it)"]
     RUN --> DM["DataManager<br/>load tables, split folds"]
     DM --> PB["PipelineBuilder<br/>selector -> scaler -> resampler -> model"]
     PB --> FIT["fit on train fold,<br/>evaluate on val fold"]
     FIT --> RES["KFoldRunResult"]

**关键护栏**：``PipelineBuilder`` 把特征选择、标准化、重采样、模型封装成**一条** sklearn Pipeline，
在交叉验证里只在训练折上 ``fit``，从而彻底杜绝数据穿越（见 :doc:`invariants`）。

阶段七 · 产出
-------------

Runner 返回的 ``KFoldRunResult`` 交给报告与可视化层，落盘为模型文件、指标表和图（ROC/校准/DCA 等，**图内英文**），
全部写入 ``{output}`` 目录。控制权逐层返回，命令层打印 ``... completed successfully!``。

回到主链
--------

把七个阶段抽象掉细节，就是那条贯穿所有子系统的同一主链——换成 ``preprocess`` / ``get-habitat`` / ``model``，
只是把 ``MLConfig``/``MLConfigurator``/``KFoldWorkflow`` 换成各自子系统的对应物，**骨架完全一样**：

.. mermaid::

   flowchart LR
     classDef a fill:#fff2cc,stroke:#d6b656,stroke-width:2px,color:#000;
     classDef b fill:#ffe6cc,stroke:#d79b00,stroke-width:2px,color:#000;
     classDef c fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000;
     classDef d fill:#dae8fc,stroke:#6c8ebf,stroke-width:2px,color:#000;

     Y["YAML"]:::a --> V["Schema<br/>validate"]:::b --> C["Configurator<br/>assemble"]:::c --> O["Orchestrator<br/>execute"]:::d

.. seealso::

   - 各命令与其 Schema/核心函数/编排器的对照表见 :doc:`architecture` 的"CLI 与 Core 映射"。
   - 生境与 ML 子系统内部的执行细节见 :doc:`subsystems`。
