代码地图
========

本页帮助你快速定位代码：仓库顶层结构、``habit/`` 包各子目录职责，以及 "我想改 X，该去哪个文件" 的对照表。

仓库顶层
--------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - 路径
     - 内容
   * - ``habit/``
     - Python 主包（全部业务代码）。
   * - ``config/``
     - 所有示例 / 生产 YAML 配置，按子系统分目录（``preprocessing/`` ``habitat/`` ``machine_learning/`` …）。
   * - ``demo_data/``
     - 演示数据与运行产物（影像、ML 表格、配准可执行文件等）。
   * - ``tests/``
     - pytest 测试 + 可执行 demo 脚本（见 :doc:`dev_workflow`）。
   * - ``docs/``
     - 本文档（Sphinx 源码在 ``docs/source/``）。
   * - ``habit-gui/``
     - 可选的独立 Web GUI 仓库/目录（FastAPI + React + bridge）；默认不在本仓库 checkout 中（见 ``.gitignore``）；``habit gui`` 会在 sibling ``habit-gui/`` 或 bundled ``habit/_gui_bundle`` 中查找。
   * - ``pyproject.toml``
     - 打包与入口点定义（``habit = "habit.cli:cli"``）。

``habit/`` 包结构
-----------------

.. mermaid::

   flowchart TD
     ROOT["habit/"]
     ROOT --> CLI["cli.py — Click command group"]
     ROOT --> CMD["commands/ — cmd_*.py (active CLI impl)"]
     ROOT --> CORE["core/ — business logic"]
     ROOT --> UTILS["utils/ — shared utilities"]

     CORE --> COM["common/ — configs, configurators, contracts"]
     CORE --> SCH["schemas/ — workflow & step schemas, registry, reflect"]
     CORE --> PRE["preprocessing/"]
     CORE --> HAB["habitat_analysis/"]
     CORE --> MLC["machine_learning/"]
     CORE --> DCM["dicom_sort/"]

     COM --> CFG["configs/"]
     COM --> CON["configurators/"]
     COM --> REG["registry.py"]
     COM --> ORC["orchestrator.py"]

顶层子包职责
------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - 子包
     - 职责
   * - ``habit/cli.py``
     - Click 根命令组，所有子命令在此声明；命令体只做 "延迟导入 + 转发"。
   * - ``habit/commands/``
     - **当前生效的命令实现层**：每个 ``cmd_*.py`` 负责加载配置、调用核心、输出结果。共享 helper 在 ``common.py``。
   * - ``habit/core/common/``
     - 跨域基建：YAML 加载与路径解析（``configs/``）、Configurator 基类（``configurators/``）、
       统一注册表基类（``registry.py`` → :class:`~habit.core.common.registry.ClassRegistry`）、
       编排器契约表（``orchestrator.py`` → :data:`~habit.core.common.orchestrator.ORCHESTRATOR_CONTRACT`）。
   * - ``habit/core/schemas/``
     - Pydantic 配置模型：整份工作流（``workflows/``）、单步参数（``steps/``）、参数注册表（``registry.py``）、校验（``validation.py``）、GUI 反射（``reflect.py`` / ``field_reflect.py``）。
   * - ``habit/core/preprocessing/``
     - 影像预处理批流水线：``BatchProcessor``、``BasePreprocessor``、``PreprocessorFactory`` 及各步骤实现。
   * - ``habit/core/habitat_analysis/``
     - 生境分割 + 聚类特征 + 分割后特征提取 + 传统 radiomics（见 :doc:`subsystems`）。
   * - ``habit/core/machine_learning/``
     - 表格机器学习：数据装配、特征选择、建模、评估、报告、可视化、统计检验。
   * - ``habit/core/dicom_sort/``
     - 独立的 DICOM 排序（基于 dcm2niix），不走 ``BatchProcessor``。
   * - ``habit/utils/``
     - 跨子系统共享工具（见下节）。

共享工具 ``habit/utils/``
-------------------------

按开发约定，所有跨子系统复用工具集中于此，其中 **进度条必须统一使用** ``progress_utils.py``。常用模块：

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 文件
     - 用途
   * - ``progress_utils.py``
     - **统一进度条**\ （全包标准，禁止各处自造 tqdm）。
   * - ``yaml_utils.py``
     - YAML 读写封装。
   * - ``log_utils.py``
     - 日志与 ``LoggerManager`` / ``setup_logger``。
   * - ``io_utils.py`` / ``file_system_utils.py``
     - 影像/掩膜路径解析、路径转换（含 Windows/WSL）。
   * - ``parallel_utils.py`` / ``parallel_gpu_utils.py``
     - 通用并行与 GPU 槽位分配（torch radiomics）。
   * - ``visualization_utils.py`` / ``font_config.py``
     - 绘图与字体配置（**图内一律英文**）。
   * - ``habitats_results_io.py`` / ``habitat_postprocess_utils.py``
     - 生境结果读写与后处理。
   * - ``radiomics_params_utils.py`` / ``torch_radiomics_utils.py``
     - Radiomics 参数与工具。
   * - ``job_cancel.py``
     - 任务取消检测（供长任务与 GUI 使用）。

跨子系统契约文件
----------------

以下文件定义全包共享的接口约定，新增工厂或编排器时必须同步更新并跑契约测试：

.. mermaid::

   flowchart LR
     REG["common/registry.py<br/>ClassRegistry"] --> PF["PreprocessorFactory"]
     REG --> MF["ModelFactory"]
     REG --> CF["ClusteringAlgorithmFactory"]
     REG --> EF["FeatureExtractorRegistry"]
     REG --> PP["PreprocessingMethodFactory"]
     REG --> HF["HabitatFeatureRegistry"]

     ORC["common/orchestrator.py<br/>ORCHESTRATOR_CONTRACT"] --> BP["BatchProcessor"]
     ORC --> HA["HabitatAnalysis"]
     ORC --> HW["HoldoutWorkflow / ..."]

     TST["tests/test_architecture_contracts.py"] -.-> REG
     TST -.-> ORC

"我想改 X，去哪找"
------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 目标
     - 起点文件
   * - 新增 / 修改一个 CLI 命令
     - ``habit/cli.py`` + ``habit/commands/cmd_*.py``
   * - 新增一个预处理步骤
     - ``habit/core/preprocessing/`` + ``PreprocessorFactory``
   * - 新增一个聚类算法
     - ``habit/core/habitat_analysis/clustering/base_clustering.py``
   * - 新增一个聚类特征提取器
     - ``habit/core/habitat_analysis/clustering_features/base_extractor.py``
   * - 新增一个机器学习模型
     - ``habit/core/machine_learning/models/factory.py``
   * - 新增一个特征选择方法
     - ``habit/core/machine_learning/feature_selectors/selector_registry.py``
   * - 改动配置字段 / 校验规则
     - ``habit/core/schemas/workflows/`` 与 ``schemas/steps/``
   * - 生境三种策略的流水线步骤
     - ``habit/core/habitat_analysis/habitat_analysis.py``\ （``_PIPELINE_RECIPES``）+ ``pipelines/steps/``
   * - ML 训练 / 预测执行流
     - ``habit/core/machine_learning/workflows/`` 与 ``runners/``
   * - 新增类式工厂（扩展算法组件）
     - 继承 ``habit/core/common/registry.py`` 的 ``ClassRegistry``；参考同域已有工厂
   * - 新增顶层编排器（新 CLI 流水线）
     - 实现类 + 更新 ``common/orchestrator.py`` 的 ``ORCHESTRATOR_CONTRACT``
       （``tests/test_architecture_contracts.py`` 自动读取该表校验终端方法）

.. seealso::

   扩展点的完整清单与注册方式见 :doc:`extension_points`；动手模板见 :doc:`../customization/index`。
