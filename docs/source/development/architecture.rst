HABIT 整体架构（V1）- 整合文档
================================

本文档面向 **开发者**，整合了 HABIT V1 代码库的整体架构、模块边界、
数据流、设计模式与扩展点。是维护本仓库时的**唯一权威参考**。

.. note::
   V1 已废弃旧 ``strategies/`` 子包与 ``pipelines/pipeline_builder.py``
   的"控制器→策略→构造器"三层结构，统一收敛到一个深模块
   :py:class:`habit.core.habitat_analysis.HabitatAnalysis`。
   本文档反映重构后的事实，不再描述旧路径。

   **本文档已整合以下旧文件**：
   - ``architecture.rst`` (顶层架构)
   - ``module_architecture.rst`` (模块细节)
   - ``design_patterns.rst`` (设计模式)
   - ``testing.rst`` (测试指南)
   - ``metrics_optimization.rst`` (优化记录)
   - ``habit/core/habitat_analysis/{README,ARCHITECTURE,PIPELINE_DESIGN}.md``
     (子包内部文档；已合并到 ②-a 节与 :doc:`module_architecture`)


定位与边界
----------

HABIT（**H**\ abitat **A**\ nalysis & **B**\ iomarker **I**\ dentification **T**\ oolkit）
是一个面向影像组学（radiomics）研究的 Python 包，覆盖以下三块独立但可拼接的能力：

1. **影像预处理**：把原始 DICOM/NIfTI 数据转成可建模输入。
2. **生境分析**：包含两个子部分

   - **②-a 生境获取（Habitat Generation）**：把影像内体素聚为亚区（habitat），输出每个 subject
     的生境图（NRRD）与标签表（CSV）。
   - **②-b 生境特征提取（Habitat Feature Extraction）**：在已生成的生境图上进行多进程特征抽取，
     输出 habitat 级别的影像组学特征表。
3. **机器学习**：在生境/影像组学特征上训练分类/回归模型，做评估、对比、
   可视化、报告。

三块能力之间通过 **数据契约** 解耦（一方写文件，另一方读文件），
而非互相 ``import`` 业务代码。这是理解整个仓库依赖方向的钥匙。


五层架构总览
-------------

.. graphviz::

   digraph architecture {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fillcolor="#f6f8fa", fontname="Helvetica"];
       edge [color="#888"];

       subgraph cluster_l1 {
           label="Layer 1: 用户入口层";
           style=filled; color="#e8f4f8";
           CLI [label="CLI\nhabit/cli.py (Click)"];
           API [label="Python API\nhabit/__init__.py"];
       }

       subgraph cluster_l2 {
           label="Layer 2: 服务装配层";
           style=filled; color="#fff8e8";
           BaseCfg [label="BaseConfigurator\n(logger/cache/output_dir)"];
           HabitatCfg [label="HabitatConfigurator"];
           MLCfg [label="MLConfigurator"];
           PreCfg [label="PreprocessingConfigurator"];
       }

       subgraph cluster_l3 {
           label="Layer 3: 业务核心层（三域解耦）";
           style=filled; color="#f0f8e8";
           Pre [label="① preprocessing\n图像预处理"];
           subgraph cluster_habitat {
               label="② habitat_analysis\n生境分析";
               style=filled; color="#e8f0e8";
               HabitatGen [label="②-a 生境获取\n(Habitat Generation)"];
               HabitatFeat [label="②-b 生境特征提取\n(Habitat Feature Extraction)"];
           }
           ML [label="③ machine_learning\n机器学习"];
       }

       subgraph cluster_l4 {
           label="Layer 4: 域内架构";
           style=dashed; color="#bbb";
           PreArch [label="BatchProcessor\n+ PreprocessorFactory"];
           subgraph cluster_habitat_arch {
               label="habitat_analysis 域内架构";
               style=dashed;
               GenArch [label="HabitatAnalysis (deep)\n+ Pipeline + Managers\n生境获取", fillcolor="#fff7e6"];
               FeatArch [label="HabitatMapAnalyzer (deep)\n+ FeatureExtractor\n特征提取", fillcolor="#fff7e6"];
           }
           MLArch [label="MLWorkflow\n+ PipelineBuilder + Models"];
       }

       subgraph cluster_l5 {
           label="Layer 5: 基础设施层";
           style=filled; color="#f8e8f8";
           Configs [label="BaseConfig + Pydantic schemas"];
           Registries [label="Algorithm/Extractor registries"];
           Utils [label="habit.utils\nio/log/parallel/viz"];
       }

       CLI -> HabitatCfg;
       CLI -> MLCfg;
       CLI -> PreCfg;
       API -> BaseCfg;

       BaseCfg -> HabitatCfg;
       BaseCfg -> MLCfg;
       BaseCfg -> PreCfg;

       PreCfg -> Pre;
       HabitatCfg -> HabitatGen;
       HabitatCfg -> HabitatFeat;
       MLCfg -> ML;

       Pre -> PreArch;
       HabitatGen -> GenArch;
       HabitatFeat -> FeatArch;
       ML -> MLArch;

       PreArch -> Configs;
       GenArch -> Configs;
       FeatArch -> Configs;
       MLArch -> Configs;

       PreArch -> Utils;
       GenArch -> Utils;
       FeatArch -> Utils;
       MLArch -> Utils;
   }


依赖方向硬规则
^^^^^^^^^^^^^^^

* ``habit.utils`` 不依赖任何 ``core`` 子包；任何 core 子包都可以用 utils。
* ``habit.core.{habitat_analysis, machine_learning, preprocessing}`` **互不依赖**。
* ``habit.core.common`` 不在模块顶层 import 任何业务子包；
  装配通过 :py:mod:`habit.core.common.configurators` 下的三个域专用
  configurator 完成，且业务 import 全部 **延迟到 factory 调用时**。
* CLI / API 不应直接 import 业务子包的内部实现；统一经对应的
  configurator 拿装配好的对象。


Layer 1：用户入口层
--------------------

HABIT 暴露两套入口，覆盖不同使用场景：

.. list-table::
   :header-rows: 1
   :widths: 18 24 58

   * - 入口
     - 文件
     - 用途与定位
   * - **Python API**
     - ``habit/__init__.py``
     - 给 Notebook / 下游脚本用。``__getattr__`` 懒加载，把
       ``HabitatAnalysis`` / ``HabitatFeatureExtractor`` / ``Modeling``
       推迟到首次访问；导入 ``habit`` 不会触发整个 ``core`` 加载。
   * - **CLI（推荐主干路径）**
     - ``habit/cli.py`` + ``habit/cli_commands/commands/``
     - 通过 ``habit <subcommand>`` 调用；每个子命令薄封装：加载 YAML →
       构造 ``BaseConfig`` 子类 → 经对应的域 configurator
       装配 → 调业务对象。这是 V1 的"标准用法"。

CLI 子命令清单（节选自 ``habit/cli.py``）：

.. code-block:: text

   habit preprocess     -> cmd_preprocess.run     (BatchProcessor)
   habit get-habitat    -> cmd_habitat.run        (HabitatAnalysis.fit/predict)
   habit extract        -> cmd_extract.run        (habitat 特征表)
   habit model          -> cmd_ml.run             (Holdout 训练/预测)
   habit cv             -> cmd_cv.run             (KFold)
   habit compare        -> cmd_compare.run        (ModelComparison)
   habit icc            -> cmd_icc.run            (ICC 特征筛选)
   habit radiomics      -> cmd_radiomics.run      (传统 radiomics)
   habit retest         -> cmd_retest.run         (test-retest 分析)
   habit dice           -> utils.dice_calculator  (Dice 分数批算)
   habit dicom-info     -> cmd_dicom_info.run     (DICOM 元数据扫描)
   habit merge-csv      -> cmd_merge_csv.run      (合并 CSV)

标准调用链：

.. code-block:: text

   habit <command> -c config.yaml
     -> habit/cli.py
     -> cmd_<domain>.run_*
     -> ConfigClass.from_file(...)
     -> DomainConfigurator(...)
     -> create_<service>()
     -> service.run / fit / predict


Layer 2：服务装配层（Domain Configurators）
-------------------------------------------

V1 起，仓库不再有"一个上帝类"装配三个域。``habit/core/common/configurators/``
按业务域拆成三个并列的 configurator，全部继承同一个抽象基类
:py:class:`habit.core.common.configurators.base.BaseConfigurator`：

* :py:class:`habit.core.common.configurators.habitat.HabitatConfigurator`
  —— habitat 域的工厂方法：
  
  - ``create_feature_service()``
  - ``create_clustering_service()``
  - ``create_result_writer()``
  - ``create_habitat_analysis()`` —— 完整装配（注入三个 managers + logger）
  - ``create_habitat_map_analyzer()``
  - ``create_feature_extractor()``
  - ``create_radiomics_extractor()``
  - ``create_test_retest_analyzer()``

* :py:class:`habit.core.common.configurators.ml.MLConfigurator`
  —— ML 域的工厂方法：

  - ``create_evaluator()``
  - ``create_reporter()``
  - ``create_threshold_manager()``
  - ``create_plot_manager()``
  - ``create_metrics_store()``
  - ``create_model_comparison()``
  - ``create_ml_workflow()`` （同时覆盖 train + predict）
  - ``create_kfold_workflow()``

* :py:class:`habit.core.common.configurators.preprocessing.PreprocessingConfigurator`
  —— preprocessing 域的工厂方法：

  - ``create_batch_processor()``

设计要点：

* **域内深、域间隔**。三个 configurator 互不 import；CLI 子命令只
  挑自己需要的那一个。
* **共享只放在基类**。日志接管、``output_dir`` 处理、轻量服务缓存
  留在 ``BaseConfigurator``，避免三个子类重复实现。
* **延迟导入**。所有业务 import 在 factory 方法内部，避免
  ``common`` 模块顶层重型依赖。

.. note::
   ``habit.core.common.dependency_injection.DIContainer`` 是更通用的 DI 容器，
   但当前仓库内 **几乎未被使用**。V1 装配的实际事实是上面三个域专用的
   configurator 类。


配置体系：``BaseConfig`` + Pydantic
-----------------------------------

所有需要从 YAML 加载的配置都继承 ``habit.core.common.config_base.BaseConfig``：

.. graphviz::

   digraph configs {
       rankdir=LR;
       node [shape=box, style=rounded, fontname="Helvetica"];
       BaseConfig [shape=box, style="rounded,filled", fillcolor="#e7f0fa"];
       BaseConfig -> HabitatAnalysisConfig;
       BaseConfig -> FeatureExtractionConfig;
       BaseConfig -> RadiomicsConfig;
       BaseConfig -> PreprocessingConfig;
       BaseConfig -> MLConfig;
       BaseConfig -> ICCConfig;
       BaseConfig -> ModelComparisonConfig;
       BaseConfig -> TestRetestConfig;
   }

* ``BaseConfig.from_file(path)`` 是统一的 YAML 入口。
* 模型字段、嵌套结构与跨字段约束用 Pydantic 表达；例如
  :py:class:`HabitatAnalysisConfig` 在 ``run_mode == 'predict'`` 时强制要求
  ``pipeline_path``，并禁止 ``two_step`` 与 subject 级丢特征方法的冲突组合。
* 路径字段在加载阶段统一解析为绝对路径。


Layer 3：业务核心层（三域解耦）
-------------------------------

本层包含三个独立的业务域，它们之间**不直接 import 彼此的实现**，
仅通过**文件产物**（CSV/NRRD/PKL）衔接。

### ① ``habit.core.preprocessing`` —— 图像预处理

**定位**：把原始 DICOM/NIfTI 跑过一串可配置预处理步骤（重采样、配准、
Z-score、DICOM→NIfTI 等），输出标准化影像。

**关键模块**：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 模块
     - 职责
   * - ``image_processor_pipeline.py``
     - ``BatchProcessor`` (deep)：按 subject 并行调度整条预处理链。
   * - ``base_preprocessor.py``
     - ``BasePreprocessor``：所有具体预处理器的统一接口。
   * - ``preprocessor_factory.py``
     - ``PreprocessorFactory``：注册名 → 具体实例。
   * - ``config_schemas.py``
     - ``PreprocessingConfig``：步骤名 → 参数的有序映射。

**数据流**：

.. code-block:: text

   raw images (per subject)
     -> LoadImagePreprocessor (隐式前置步骤)
     -> [step_1] -> [step_2] -> ... -> [step_n]
     -> standardized images on disk

**扩展点：新增预处理步骤**

1. 新建继承 ``BasePreprocessor`` 的类。
2. 实现 ``__call__(self, data: Dict[str, Any]) -> Dict[str, Any]``。
3. 用 ``PreprocessorFactory.register("step_name")`` 注册。
4. 确保模块会被 ``habit.core.preprocessing`` 导入。

### ② ``habitat.core.habitat_analysis`` —— 生境分析

生域分析包含**两个独立的子部分**，通过文件产物（NRRD/CSV）衔接：

#### ②-a **生境获取（Habitat Generation）**

**定位**：把多模态影像在 ROI 内的体素聚为生境（habitat），输出 NRRD 生境图与
CSV 标签表，并把训练状态持久化为 ``habitat_pipeline.pkl`` 以供后续预测。

**子包目录布局**：

.. code-block:: text

   habit/core/habitat_analysis/
   ├── __init__.py                 # Public API
   ├── habitat_analysis.py         # HabitatAnalysis (deep) + _PIPELINE_RECIPES
   ├── config_schemas.py           # Pydantic configuration models
   │
   ├── services/                   # Service layer (called by steps)
   │   ├── feature_service.py      # FeatureService
   │   ├── clustering_service.py   # ClusteringService
   │   └── result_writer.py        # ResultWriter
   │
   ├── pipelines/
   │   ├── base_pipeline.py        # HabitatPipeline + BasePipelineStep
   │   │                           # + IndividualLevelStep / GroupLevelStep markers
   │   └── steps/                  # Concrete pipeline steps (10 active + 1 deprecated)
   │       ├── voxel_feature_extractor.py
   │       ├── subject_preprocessing.py
   │       ├── individual_clustering.py
   │       ├── calculate_mean_voxel_features.py
   │       ├── supervoxel_feature_extraction.py
   │       ├── merge_supervoxel_features.py
   │       ├── combine_supervoxels.py
   │       ├── concatenate_voxels.py
   │       ├── group_preprocessing.py
   │       └── population_clustering.py
   │
   ├── clustering_features/        # Pre-clustering feature extractors
   ├── clustering/                 # Clustering algorithms (KMeans / GMM / SLIC / ...)
   ├── feature_preprocessing/      # Subject- / group-level preprocessing methods
   ├── habitat_features/           # Post-clustering analyzers (HabitatMapAnalyzer)
   └── utils/                      # Subpackage-local utilities

.. note::
   V1 已删除旧 ``strategies/`` 子包与 ``pipelines/pipeline_builder.py``。
   所有 mode 分发逻辑收敛到 ``habitat_analysis.py`` 中的
   ``_PIPELINE_RECIPES`` 字典；不再有 ``BaseClusteringStrategy`` 类树
   和 ``build_habitat_pipeline()`` 工厂。

**关键模块**：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 模块
     - 职责
   * - ``habitat_analysis.py`` (**deep**)
     - V1 的 **唯一编排入口** 。build → fit/predict → 持久化 → 结果后处理。
       内部用 ``_PIPELINE_RECIPES`` 按 ``clustering_mode`` 分发 step 列表，
       用 ``_PIPELINE_SERVICE_ATTRS`` 显式白名单注入 service。
   * - ``services/{feature_service,clustering_service,result_writer}.py``
     - 三类领域职责：特征抽取与预处理、聚类训练/选择、结果落盘与可视化。
       构造时注入到 ``HabitatAnalysis``，由后者再注入到 pipeline 步。
   * - ``pipelines/base_pipeline.py``
     - sklearn 风格的 ``HabitatPipeline`` + ``BasePipelineStep`` +
       ``IndividualLevelStep`` / ``GroupLevelStep`` 标记类。
       ``fit_transform`` / ``transform`` / ``save`` / ``load`` 接口。
   * - ``pipelines/steps/*.py``
     - 10 个具体步骤；按继承的标记类自动分入个体级 / 群体级两类。
   * - ``clustering/``
     - K-Means / GMM / SLIC / Hierarchical 等 **聚类算法**
       （此处的 ``BaseClusteringStrategy`` 是 *算法* 接口，**不是** 已删除的策略层）。
   * - ``clustering_features/``
     - 聚类前的体素 / supervoxel 级特征抽取实现，由 factory 按配置选择。
   * - ``habitat_features/``
     - 已生成 habitat map 上的特征抽取（HabitatMapAnalyzer 等）。
   * - ``config_schemas.py``
     - ``HabitatAnalysisConfig`` / ``FeatureConstructionConfig`` /
       ``HabitatsSegmentionConfig`` / ``ResultColumns``。

公共 API
^^^^^^^^

``HabitatAnalysis`` 暴露三个入口，全部返回 ``pd.DataFrame``：

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - 方法
     - 用途
     - 副作用
   * - ``fit(subjects=None, save_results_csv=None)``
     - 训练 + 持久化。按 recipe 构造 pipeline、调用
       ``pipeline.fit_transform``、保存 ``habitat_pipeline.pkl``、
       做结果后处理与 CSV/NRRD 输出。
     - 写入 ``.pkl``、``habitats.csv``、``habitat_*.nrrd``、可视化。
   * - ``predict(pipeline_path, subjects=None, save_results_csv=None)``
     - 加载 + transform。读取已保存 pipeline、与当前 config 调和、
       通过白名单注入当前 services、强制 ``plot_curves=False``、
       调用 ``pipeline.transform``。
     - 写入 ``habitats.csv``、``habitat_*.nrrd``（不写 ``.pkl``）。
   * - ``run(subjects=None, save_results_csv=None, load_from=None)``
     - 分发器。根据 ``load_from`` 或 ``config.run_mode`` 路由到
       ``fit`` / ``predict``，作为 CLI / 脚本的稳定入口。
     - 同被分发到的方法。

CLI ``habit get-habitat`` 是 V1 的唯一外部入口，封装上述方法；旧的
``scripts/run_habitat_analysis.py`` 双轨执行器已删除。

Pipeline Recipe 分发
^^^^^^^^^^^^^^^^^^^^

``_PIPELINE_RECIPES`` 是 mode 分发的 **唯一真相**：

.. code-block:: python

   _PIPELINE_RECIPES = {
       'two_step':       _build_two_step_steps,
       'one_step':       _build_one_step_steps,
       'direct_pooling': _build_pooling_steps,
   }

``HabitatAnalysis._build_pipeline()`` 仅做一次字典查表 → 调用 builder →
把返回的 step 列表封装成 ``HabitatPipeline``。新增 mode 等价于「新增一个
builder 函数 + 字典加一行」，不需要新建文件或类。

三种 recipe 的步骤组成（与 ``habitat_analysis.py`` 中的
``_build_*_steps`` 完全一致）：

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Mode
     - 步骤序列
   * - ``two_step``
     - ``voxel_features`` → ``subject_preprocessing`` →
       ``individual_clustering`` (target=supervoxel) →
       ``calculate_mean_voxel_features`` →
       *(可选)* ``supervoxel_advanced_features`` →
       ``merge_supervoxel_features`` → ``combine_supervoxels`` →
       *(可选)* ``group_preprocessing`` →
       ``population_clustering``
   * - ``one_step``
     - ``voxel_features`` → ``subject_preprocessing`` →
       ``individual_clustering`` (target=habitat, find_optimal=True) →
       ``calculate_mean_voxel_features`` →
       ``merge_supervoxel_features`` → ``combine_supervoxels``
   * - ``direct_pooling``
     - ``voxel_features`` → ``subject_preprocessing`` →
       ``concatenate_voxels`` →
       *(可选)* ``group_preprocessing`` →
       ``population_clustering``

可选步骤的开关：

* ``supervoxel_advanced_features``: 仅当
  ``config.FeatureConstruction.supervoxel_level.method`` 不是
  ``mean_voxel_features`` 时插入。
* ``group_preprocessing``: 仅当
  ``config.FeatureConstruction.preprocessing_for_group_level.methods``
  非空时插入。

Service 注入与 ``_PIPELINE_SERVICE_ATTRS``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

训练 pipeline 时，service 引用在构造时绑定到每个 step。预测时从 ``.pkl``
反序列化的 step 仍持有训练时的 service 引用——这些必须替换为当前运行的
service 实例。V1 用 **显式白名单** 完成：

.. code-block:: python

   _PIPELINE_SERVICE_ATTRS: Tuple[str, ...] = (
       'feature_service',
       'clustering_service',
       'result_writer',
   )

``_inject_services_into_pipeline(pipeline)`` 遍历每个 step，把白名单中
step 持有的属性替换为当前 service 实例，并同步更新 ``pipeline.config``。
对 ``feature_service``，只 *同步路径与日志目标*，保留训练时的 fitted 状态
（避免把已训练的特征状态丢掉）。

.. note::
   V1 故意删除了之前基于 ``dir(self)`` 的反射注入。任何未来命名为
   ``*_service`` 的属性都不会被悄悄注入；新增 service 必须显式
   修改这个白名单——这是设计意图。

训练数据流（two-step 为例）
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. graphviz::

   digraph fit_flow {
       rankdir=LR;
       node [shape=box, style=rounded, fontname="Helvetica"];

       Imgs [label="images + masks\n(per subject)"];
       VF [label="VoxelFeatureExtractor"];
       SP [label="SubjectPreprocessing"];
       IC [label="IndividualClustering\nvoxel -> supervoxel"];
       MVF [label="CalculateMeanVoxelFeatures"];
       SF [label="SupervoxelFeatureExtraction\n(optional)", style="dashed,rounded"];
       MS [label="MergeSupervoxelFeatures"];
       CS [label="CombineSupervoxels\n(group)"];
       GP [label="GroupPreprocessing\n(optional, group, stateful)", style="dashed,rounded"];
       PC [label="PopulationClustering\n(group, stateful)"];
       Out [label="habitat maps (.nrrd)\nhabitats.csv\nhabitat_pipeline.pkl",
            shape=note, fillcolor="#fff7e6", style="filled,rounded"];

       Imgs -> VF -> SP -> IC -> MVF -> SF -> MS -> CS -> GP -> PC -> Out;
   }

数据形状演化（two-step）::

   Per subject: N voxels x F features
     -> per subject: K supervoxel labels + voxel features
     -> per subject: K supervoxels x F mean features
     -> per subject: 'supervoxel_df' (K x F + Subject + Supervoxel + Count)
     -> all subjects concatenated: M x (F + metadata)
     -> all subjects: M x (F' + metadata)         (after group preprocessing)
     -> all subjects: M x (F' + metadata + Habitats)

预测数据流
^^^^^^^^^^

预测路径与训练共用同一个 ``HabitatPipeline``，只是：

1. 从磁盘 ``HabitatPipeline.load(pipeline_path)`` 反序列化训练好的 step。
2. 调用 ``HabitatAnalysis._inject_services_into_pipeline`` 用
   ``_PIPELINE_SERVICE_ATTRS`` 白名单把当前运行时的 service 注入到 step。
3. 强制 ``pipeline.config.plot_curves = False`` —— 预测路径下没有 cluster
   selection 的中间分数，绘图代码会在那些分数上抛 ``TypeError``，关闭
   ``plot_curves`` 是把这个 bug 关在源头。
4. 只调 ``transform``，不再 ``fit``。

外部产物
^^^^^^^^

* ``<out_dir>/habitats.csv`` —— habitat 标签表（含 Subject、Supervoxel、
  Habitats、Count + 特征列）。
* ``<out_dir>/{subject}_supervoxel.nrrd`` —— two-step 模式的超体素图。
* ``<out_dir>/{subject}_habitats.nrrd`` —— 最终生境图（one-step 由
  IndividualClustering 直接写出；其它模式由 ResultWriter 在结果后处理时写出）。
* ``<out_dir>/habitat_pipeline.pkl`` —— joblib 序列化的训练 pipeline，
  预测仅依赖此一个产物。
* ``<out_dir>/habitat_analysis.log`` —— 当前运行日志。
* ``<out_dir>/visualizations/`` —— 当 ``plot_curves`` 启用时的聚类验证
  曲线和 2D/3D 可视化。

Pipeline 持久化格式
^^^^^^^^^^^^^^^^^^^

``HabitatPipeline.save(path)`` 用 joblib 写一个文件，里面包含：

* 完整的 step 列表（每个 step 仍保留它在 ``fit()`` 学到的参数）；
* 训练时使用的 ``config``（预测时会被当前 config 替换）。

默认位置是 ``<config.out_dir>/habitat_pipeline.pkl``。

deep / shallow 划分
^^^^^^^^^^^^^^^^^^^

* **Deep**：``HabitatAnalysis`` + ``HabitatPipeline``。接口表面少，实现表面大。
* **Shallow**：``services/``、单个 step、单个 algorithm/extractor。
  这些都是「插件式」的薄壳，遵循固定接口。

扩展点
^^^^^^

**新增 clustering mode**

1. 在 ``habitat_analysis.py`` 实现一个
   ``_build_<mode>_steps(config, feature_service, clustering_service, result_writer)``
   函数，返回 ``List[Tuple[str, BasePipelineStep]]``。
2. 在 ``_PIPELINE_RECIPES`` 字典中加入 ``'<mode>': _build_<mode>_steps``。
3. *(可选)* 输出列差异（如 one-step 把 ``Supervoxel`` 镜像为 ``Habitats``）
   放在对应的 step 内（参见 ``MergeSupervoxelFeaturesStep``）；只有 *跟随
   pipeline 之后* 的副作用（额外文件、日志、条件保存）才进
   ``HabitatAnalysis._save_results``。

不需要新建类、文件或注册表条目。recipe 字典就是注册表。

**新增 pipeline step**

1. 继承 ``IndividualLevelStep`` 或 ``GroupLevelStep``（根据是否需要按 subject
   独立处理决定），实现 ``fit(X, y=None)`` 与 ``transform(X)``。
2. 把 step 注入到一条或多条 recipe 中。``HabitatPipeline`` 会按继承的标记
   类自动把 step 分入「个体级（可并行）」与「群体级（顺序）」两个阶段。
3. 在 :doc:`module_architecture` 的 *Pipeline 步契约* 表中追加该 step 的
   I/O 字段。

**新增 service**

1. 在 ``services/`` 下新建 service 类。
2. 在 ``HabitatAnalysis.__init__`` 注入并保存到 ``self``。
3. **必须** 把它的属性名加入 ``_PIPELINE_SERVICE_ATTRS``——否则 predict 路径
   不会重新注入它。
4. 在 ``habit/core/common/configurators/habitat.py`` 的
   ``HabitatConfigurator.create_habitat_analysis`` 中构造并传入新 service。

**新增聚类算法**

1. 在 ``clustering/`` 下继承 ``BaseClusteringStrategy``（这里的 strategy
   是 *算法接口*，不是已删除的旧顶层策略层）。
2. 用注册装饰器 ``@register_clustering('algo_name')`` 注册，``ClusteringService``
   会按配置 ``algorithm`` 字段解析。

设计决策
^^^^^^^^

1. **三层折叠为一个 deep 模块**：原来的 ``HabitatAnalysis`` 控制器、
   ``strategies/*Strategy`` 子类树、``pipeline_builder.py`` 工厂层各自只做
   转发，没有真实行为。V1 把三者收敛到 ``habitat_analysis.py``，认知形状
   贴合实际。
2. **Recipe 字典代替类层级**：每个 *strategy* 的本质区别只在 step 列表，
   一个返回列表的函数加一行字典就够了。
3. **显式注入白名单**：``_PIPELINE_SERVICE_ATTRS`` 让"哪些属性允许在 load
   时被重新绑定"成为可读、可测的契约，避免反射的脆弱性。
4. **predict 强制 ``plot_curves=False``**：在源头把"画 selection 曲线时
   消费 ``None``"的 bug 关掉，比逐处补丁更稳。
5. **fit/predict 共用一个 pipeline 类**：避免两套并行执行栈带来的
   plumbing 重复。
6. **sklearn 风格步骤接口**：``fit / transform / fit_transform`` 对 ML 用户
   零学习成本，也方便给新 step 一个明确契约。

#### ②-b **生境特征提取（Habitat Feature Extraction）**

**定位**：在已生成的生境图上进行多进程特征抽取，输出 habitat 级别的影像组学特征表（CSV），供下游机器学习使用。

**输入依赖**：

* ②-a 生境获取阶段生成的 ``<out_dir>/habitat_*.nrrd`` 文件
* 原始影像和 mask 文件（用于在每个 habitat 区域内计算影像组学特征）

**关键模块**：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 模块
     - 职责
   * - ``analyzers/habitat_analyzer.py`` (**deep**)
     - ``HabitatMapAnalyzer``：核心协调器，负责多进程调度、特征汇总、结果持久化。
   * - ``managers/feature_service.py``
     - ``FeatureService``：管理特征提取器的注册、选择、执行
       （复用自生境获取阶段的同一组件）。
   * - ``extractors/``
     - 体素级 / supervoxel 级特征抽取实现（如 radiomics 特征、纹理特征等）。
   * - ``config_schemas.py``
     - ``FeatureExtractionConfig`` / ``RadiomicsConfig``：定义提取参数
       （特征类别、归一化方式等）。

***** 数据流

.. graphviz::

   digraph feature_flow {
       rankdir=LR;
       node [shape=box, style=rounded, fontname="Helvetica"];

       Input [label="输入\nhabitat_*.nrrd\n+ 原始影像/mask"];
       HMA [label="HabitatMapAnalyzer\n(多进程调度)"];
       FM [label="FeatureService\n(特征提取器管理)"];
       Ext [label="Extractors\n(radiomics/texture/etc.)"];
       Output [label="输出\nhabitat_features.csv\n(per subject)",
               shape=note, fillcolor="#fff7e6", style="filled,rounded"];

       Input -> HMA -> FM -> Ext -> Output;
   }

***** 工作流程

1. **加载生境图**：读取 ②-a 阶段生成的 ``.nrrd`` 文件，获取每个体素的 habitat 标签
2. **区域分割**：根据 habitat 标签将 ROI 划分为多个子区域
3. **并行提取**：对每个 habitat 区域并行执行特征提取（利用 ``parallel_utils``）
4. **特征聚合**：汇总所有 habitat 的特征为表格格式
5. **结果保存**：输出 ``<out_dir>/habitat_features.csv``，每行一个 subject × habitat 组合

***** 外部产物

* ``<out_dir>/habitat_features.csv`` —— habitat 级别特征表（行：subject×habitat，列：特征名）
* 可选：``<out_dir>/features_per_habitat/`` —— 每个 habitat 的详细特征报告

***** 与生境获取的关系

::

   ②-a 生境获取                          ②-b 生境特征提取
   ┌─────────────────────┐              ┌─────────────────────────┐
   │ images + masks      │              │ habitat_*.nrrd          │
   │       ↓             │              │ images + masks          │
   │ HabitatAnalysis.fit │──产出 NRRD──▶│       ↓                 │
   │       ↓             │              │ HabitatMapAnalyzer.run  │
   │ habitat_*.nrrd      │              │       ↓                 │
   │ habitats.csv        │              │ habitat_features.csv    │
   └─────────────────────┘              └─────────────────────────┘
      训练/预测                              仅预测（无需训练）

**注意**：两个子部分可以独立调用：

* 用户可以只运行 ②-a 获取 habitat maps
* 或者在已有 habitat maps 的基础上只运行 ②-b 提取特征
* 也可以通过 CLI 一次性完成两步（``habit get-habitat`` + ``habit extract``）

***** 扩展点

* **新增特征类型**：实现新的 extractor 类并注册到 ``FeatureService``
* **新增聚合策略**：修改 ``HabitatMapAnalyzer`` 中的特征汇总逻辑
* **优化并行性能**：调整进程数、内存映射策略

### ③ ``habit.core.machine_learning`` —— 机器学习

**定位**：读 CSV 特征表 → sklearn ``Pipeline`` 训练 / 交叉验证 / 预测 / 
多模型对比 → 图、报告、模型 .pkl。**不依赖** habitat_analysis：
habitat 表也好、传统 radiomics 表也好，都是同一份 CSV 契约。

**关键模块**：

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 模块
     - 职责
   * - ``workflows.base.BaseWorkflow`` (**deep**)
     - workflow 公共骨架：配置校验 + DataManager +
       PlotManager + PipelineBuilder + CallbackList。
   * - ``workflows/holdout_workflow.py``
     - ``MachineLearningWorkflow``：训练 + 预测统一执行体（训练路径内部委托 runner）。
   * - ``runners/holdout.py``
     - ``HoldoutRunner``：封装 holdout 训练、评估与汇总逻辑。
   * - ``workflows/kfold_workflow.py``
     - ``MachineLearningKFoldWorkflow``（对外入口，内部委托 runner）。
   * - ``runners/kfold.py``
     - ``KFoldRunner``：封装 K-Fold 折循环、模型训练与聚合逻辑。
   * - ``workflows/comparison_workflow.py``
     - ``ModelComparison``：多模型评估 + 可视化 + 报告。
   * - ``pipeline_utils.PipelineBuilder``
     - 把 YAML 配置串成 sklearn ``Pipeline``。
   * - ``models/factory.ModelFactory``
     - 用装饰器 ``@register`` 注册具体模型。
   * - ``feature_selectors/selector_registry``
     - selector 元数据注册与执行。
   * - ``evaluation/`` / ``visualization/`` / ``reporting/``
     - 评估、绘图和报告（回调模块已移除）。

.. note::
   2026-05 起，``MachineLearningKFoldWorkflow`` 采用“workflow 薄壳 + runner 执行器”模式：
   workflow 负责入口与生命周期，``KFoldRunner`` 负责折内计算细节。所有输出步骤
   （保存模型、写报表、画图）都改为显式调用 reporting 组件，不再依赖 callbacks。

**数据流（训练 / 预测 / 对比）**：

.. graphviz::

   digraph ml_flow {
       rankdir=LR;
       node [shape=box, style=rounded, fontname="Helvetica"];

       CSV [label="features CSV\n(habitat or radiomics)"];

       subgraph cluster_train {
           label="train";
           style=dashed; color="#bbb";
           DM [label="DataManager"];
           PB [label="PipelineBuilder\n-> sklearn Pipeline"];
           CB [label="CallbackList"];
           Mods [label="<model>_final_pipeline.pkl",
                 shape=note, fillcolor="#fff7e6"];
           Rep [label="reports",
                shape=note, fillcolor="#fff7e6"];
       }

       subgraph cluster_predict {
           label="predict";
           style=dashed; color="#bbb";
           Load [label="load *.pkl"];
           Pred [label="predict on new CSV"];
           POut [label="predictions",
                 shape=note, fillcolor="#fff7e6"];
       }

       subgraph cluster_compare {
           label="compare";
           style=dashed; color="#bbb";
           ME [label="MultifileEvaluator"];
           MC [label="ModelComparison"];
           COut [label="comparison report",
                 shape=note, fillcolor="#fff7e6"];
       }

       CSV -> DM -> PB -> CB;
       CB -> Mods;
       CB -> Rep;
       CSV -> Load -> Pred -> POut;
       Rep -> ME -> MC -> COut;
   }

**扩展点**：

* **新增模型**：继承基类或兼容 sklearn estimator → ``ModelFactory.register()``。
* **新增特征选择**：实现 selector → 加入 registry → 明确 scaling 前/后位置。


Layer 4：基础设施层
-------------------

### ``habit.core.common``

业务无关的「装配 + 配置 + 横切」工具：

* ``config_base.py`` —— Pydantic ``BaseConfig`` 基类与 ``from_file``。
* ``configurators/`` —— 服务装配（见 Layer 2）。
* ``dependency_injection.DIContainer`` —— 通用 DI 容器（当前非主路径）。
* ``dataframe_utils.py`` —— DataFrame/数组横切清洗。

### ``habit.utils``

业务零依赖的横切工具集合：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 模块
     - 职责
   * - ``io_utils``
     - 路径扫描、SITK/YAML/JSON/CSV 读写。
   * - ``log_utils``
     - 日志初始化（与 configurators 协作）。
   * - ``progress_utils``
     - 统一进度条 ``CustomTqdm``。
   * - ``parallel_utils``
     - 并行 map，整合进度条与错误收集。
   * - ``visualization_utils``
     - 绘图风格与可视化辅助。
   * - ``habitat_postprocess_utils``
     - habitat map 连通域等后处理。


设计模式
--------

V1 重构后，项目使用的设计模式如下（**已移除过时的策略模式**）：

工厂模式 (Factory Pattern)
^^^^^^^^^^^^^^^^^^^^^^^^^^

用于动态创建特征提取器和聚类算法。

**实现位置**：

* ``extractors/base_extractor.py`` —— ``get_feature_extractor(name)``
* ``algorithms/base_clustering.py`` —— ``get_clustering_algorithm(name)``
* ``extractors/feature_extractor_factory.py`` —— ``FeatureExtractorFactory``
* ``models/factory.py`` —— ``ModelFactory``

**注册机制示例**：

.. code-block:: python

   # 聚类算法注册
   @register_clustering('kmeans')
   class KMeansClustering(BaseClustering):
       ...

   # 使用
   algo = get_clustering_algorithm('kmeans', n_clusters=5)

依赖注入 (Dependency Injection)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

通过域专用 configurator 管理依赖关系，提高可测试性。

**实际签名**（V1 当前实现）：

.. code-block:: python

   class HabitatAnalysis:
       def __init__(
           self,
           config,
           feature_service,      # FeatureService 实例
           clustering_service,   # ClusteringService 实例
           result_writer,       # ResultWriter 实例
           logger,
       ):
           self.feature_service = feature_service
           self.clustering_service = clustering_service
           self.result_writer = result_writer

   # 使用 HabitatConfigurator 装配
   from habit.core.common.configurators import HabitatConfigurator
   configurator = HabitatConfigurator(config=config)
   analysis = configurator.create_habitat_analysis()

模板方法模式 (Template Method Pattern)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

用于定义算法骨架，子类实现具体步骤。

**应用场景**：

* ``BaseClustering.fit()`` 定义 _validate → _initialize → _iterate → _finalize 骨架
* ``BaseWorkflow`` 定义 workflow 执行骨架
* ``BasePipelineStep`` 定义 fit → transform → fit_transform 骨架

.. code-block:: python

   class BaseClustering(ABC):
       def fit(self, X):
           self._validate_input(X)
           self._initialize(X)
           self._iterate(X)
           self._finalize()
       
       @abstractmethod
       def _initialize(self, X): ...
       
       @abstractmethod
       def _iterate(self, X): ...

注册表模式 (Registry Pattern)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

用于插件式扩展：

* ``_CLUSTERING_REGISTRY`` —— 聚类算法注册表
* ``_EXTRACTOR_REGISTRY`` —— 特征提取器注册表
* Model Factory 的 ``@register`` 装饰器 —— 模型注册表

观察者/回调模式 (Observer/Callback Pattern)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

用于日志和进度通知：

* Logger 注入到各 manager
* ``CallbackList`` 在 ML workflow 中（checkpoint/report/visualization）

.. note::
   **已废弃的模式（V1 前）**：
   
   ❌ **策略模式（Strategy Pattern）** 用于 clustering mode 分发 —— 
   已被 ``_PIPELINE_RECIPES`` 字典分发替代。
   旧的 ``strategies/`` 子包和 ``BaseClusteringStrategy`` 接口已删除。


测试指南
--------

HABIT 使用 pytest 进行测试。

运行测试
~~~~~~~~

.. code-block:: bash

   pytest tests/
   pytest tests/test_habitat.py
   pytest tests/test_habitat.py::test_feature_extraction
   pytest --cov=habit --cov-report=html tests/

编写测试
~~~~~~~~

测试文件应该放在 ``tests/`` 目录下，并以 ``test_`` 开头。

**修正后的 API 示例**（匹配 V1 实际接口）：

.. code-block:: python

   import pytest
   import numpy as np
   import SimpleITK as sitk
   from habit.core.habitat_analysis.clustering_features.raw_feature_extractor import RawFeatureExtractor
   
   def test_raw_feature_extractor():
       image = sitk.GetImageFromArray(np.random.rand(10, 10, 10))
       mask = sitk.GetImageFromArray(np.ones((10, 10, 10)))
       
       extractor = RawFeatureExtractor()
       
       # 正确的方法名和参数
       features = extractor.extract_features(
           image_data=image,
           mask_data=mask,
           subject='test_subject',
           image='t1'
       )
       
       assert features is not None
       assert isinstance(features, (np.ndarray, pd.DataFrame))

使用 Fixtures
~~~~~~~~~~~~~

.. code-block:: python

   @pytest.fixture
   def sample_image():
       return sitk.GetImageFromArray(np.random.rand(10, 10, 10))
   
   @pytest.fixture
   def sample_mask():
       return sitk.GetImageFromArray(np.ones((10, 10, 10)))
   
   def test_with_fixtures(sample_image, sample_mask):
       extractor = RawFeatureExtractor()
       features = extractor.extract_features(
           image_data=sample_image,
           mask_data=sample_mask,
           subject='test'
       )
       assert features is not None

测试最佳实践
~~~~~~~~~~~~

* 每个测试应该独立运行
* 使用描述性的测试名称
* 测试应该快速（< 1 秒）
* 使用 fixtures 来共享测试数据
* 测试应该覆盖正常情况和边界情况
* **重要**：使用正确的 API 方法名 ``extract_features()`` 而非 ``extract()``


Metrics 模块优化记录
---------------------

本次优化对 ``habit/core/machine_learning/evaluation/metrics.py`` 进行了全面改进：

主要改进
~~~~~~~~

1. **混淆矩阵缓存** 🚀 —— 引入 ``MetricsCache`` 类，约 **8倍** 速度提升
2. **扩展 Target Metrics** 💡 —— 新增 PPV/NPV/F1-score/Accuracy 支持
3. **Fallback 机制** 🎯 —— 自动寻找"最接近"阈值
4. **智能阈值选择** 🧠 —— First/Youden/Pareto+Youden 三种策略
5. **类别筛选功能** 📋 —— 按 categories 参数选择计算范围

使用示例
~~~~~~~~

.. code-block:: python

   from habit.core.machine_learning.evaluation.metrics import (
       calculate_metrics_at_target,
       calculate_metrics
   )

   # 训练集：找最优阈值
   train_result = calculate_metrics_at_target(
       y_train_true, y_train_prob,
       targets={'sensitivity': 0.91, 'specificity': 0.91},
       threshold_selection='pareto+youden',
       fallback_to_closest=True
   )

   # 测试集：应用阈值
   test_metrics = calculate_metrics(
       y_test_true, y_test_pred, y_test_prob,
       use_cache=True,
       categories=['basic']
   )

向后兼容性 ✅
~~~~~~~~~~~~~

所有现有代码无需修改，新功能通过可选参数启用。


架构摩擦点与技术债务（2026 年识别）
----------------------------------

以下是通过架构审查识别出的 **深化机会**，按优先级排序：

高优先级
~~~~~~~~

1. **消除浅层 Pipeline Steps**
   
   **问题**：部分 Step 类只是简单委托给 Manager：
   
   - ``VoxelFeatureExtractor.transform()`` → ``FeatureService.extract_voxel_features()``
   - ``SubjectPreprocessingStep.transform()`` → ``FeatureService.apply_preprocessing()``
   
   **影响文件**：``pipelines/steps/voxel_feature_extractor.py`` 等
   
   **建议**：将 Manager 方法直接提升为 Step，或让 Step 包含更多实质逻辑。

2. **提取 ImageIO 工具类**
   
   **问题**：5 个 Extractor 重复相同的 ``sitk.ReadImage`` + ``sitk.GetArrayFromImage`` 逻辑。
   
   **影响文件**：``extractors/voxel_radiomics_extractor.py``、
   ``raw_feature_extractor.py`` 等
   
   **建议**：在 ``base_extractor.py`` 中添加统一的 ``load_image()`` / ``load_mask()`` 方法。

中优先级
~~~~~~~~

3. **拆分 FeatureService 多职责**
   
   **问题**：FeatureService (400+ 行, 15个方法) 承担 4 个不同职责：
   特征提取协调、预处理执行、配置解析、数据路径管理。
   
   **建议**：拆分为 ``FeatureExtractionOrchestrator`` + ``Preprocessor`` + ``ConfigResolver``。

4. **统一 Extractor 返回类型**
   
   **问题**：不同 Extractor 返回 ``pd.DataFrame`` 或 ``np.ndarray``，下游需做类型检查。
   
   **建议**：强制基类接口返回 ``pd.DataFrame``，在基类提供转换工具。

5. **增加 Pipeline Steps 测试覆盖**
   
   **问题**：11 个 Pipeline Step 类完全没有单元测试。
   
   **建议**：为核心 Steps（VoxelFeatureExtractor、IndividualClusteringStep、
   PopulationClusteringStep、GroupPreprocessingStep）添加测试。

低优先级
~~~~~~~~

6. **明确 HabitatMapAnalyzer 定位**
   
   **问题**：``HabitatMapAnalyzer`` 是独立工具但导出为公共 API，与 ResultWriter 有职责重叠。
   
   **建议**：在文档中明确定位，或考虑整合进 ResultWriter。


维护注意事项
------------

通用规则
~~~~~~~~

- **新增业务命令**：优先新增 ``cmd_<name>.py``，再在 ``habit/cli.py`` 注册。
- **新增 YAML 服务**：先定义 ``BaseConfig`` 子类，再在对应 domain configurator 添加 ``create_*``。
- **Configurator 内部的业务 import 应放在 factory 方法内部**，避免 ``common`` 顶层 import 业务子包。
- **绘图文字必须用英文**（项目约定）。

Per-Domain 注意事项
~~~~~~~~~~~~~~~~~~~

**preprocessing**：

- ``LoadImagePreprocessor`` 常作为隐式前置步骤，用户 YAML 通常不需要写 ``load_image``。
- 预处理器应尽量只读写自己声明处理的键。
- 进度条统一使用 ``CustomTqdm`` 或经 ``parallel_utils`` 间接使用。

**habitat_analysis**：

- 旧 ``strategies/`` 与 ``pipeline_builder.py`` 已删除，不要沿用旧三层结构。
- Service 注入使用显式白名单 ``_PIPELINE_SERVICE_ATTRS``；新增 service 必须同步修改白名单。
- 子包细节、Pipeline 步契约与状态管理见 :doc:`module_architecture` 的
  ``habit.core.habitat_analysis`` 节；用户向使用见
  :doc:`../user_guide/habitat_segmentation_zh`。

**machine_learning**：

- 训练与预测共用 ``MLConfig`` 与 ``MachineLearningWorkflow``，不要恢复独立 ``PredictionWorkflow``。
- 评估/绘图/报告尽量放在对应子包或 callback，workflow 只做编排。

**utils**：

- 开发 ``habit`` 包时，通用工具统一放在 ``habit/utils``。
- ``utils`` 不要 ``import`` 业务 domain；若工具出现领域耦合，迁回对应子包。


开发者阅读顺序
---------------

1. 读本文档 **Layer 1-3**，掌握全局架构和数据流。
2. 按要改的功能选 domain：
   
   - 图像预处理：``BatchProcessor``、``PreprocessorFactory``
   - habitat 分割：``HabitatAnalysis``、``_PIPELINE_RECIPES``
   - ML 建模：``MachineLearningWorkflow``、``BaseWorkflow``、``PipelineBuilder``
3. 再深入具体 step、model、selector、extractor。
4. 最后看配置 schema 与配置模板，确认用户可配置面。
5. 查看 **架构摩擦点** 章节，了解已知技术债务。

**核心原则**：保持「入口薄、domain 内聚、domain 间通过文件契约解耦」。


索引与参考
----------

子包内部细节集中在 docs 自身：

* :doc:`module_architecture` —— 模块代码组织、Pipeline 步契约、状态管理。
* :doc:`../user_guide/habitat_segmentation_zh` —— 用户向使用、配置、API。
* :doc:`../algorithms/clustering` —— 聚类算法实现细节。

版本信息
--------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 项目
     - 内容
   * - 文档版本
     - V2.1 (整合 habitat_analysis MDs)
   * - 代码版本
     - V1 Architecture
   * - 最后更新
     - 2026-05-02
   * - 整合来源
     - architecture.rst + module_architecture.rst + design_patterns.rst +
       testing.rst + metrics_optimization.rst +
       habitat_analysis/{README,ARCHITECTURE,PIPELINE_DESIGN}.md
