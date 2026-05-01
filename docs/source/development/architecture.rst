HABIT 整体架构（V1）
=====================

本文档面向 **开发者** ，描述 HABIT V1 代码库的整体架构、模块边界、
数据流与扩展点，是维护本仓库时的首要参考。

本文档覆盖整个 ``habit`` 包；habitat_analysis 子包内部的细节
（pipeline 步、manager 协作、产物字段）见
``habit/core/habitat_analysis/ARCHITECTURE.md`` 。

.. note::
   V1 已废弃旧 ``strategies/`` 子包与 ``pipelines/pipeline_builder.py``
   的"控制器→策略→构造器"三层结构，统一收敛到一个深模块
   :py:class:`habit.core.habitat_analysis.HabitatAnalysis`。
   本文档反映重构后的事实，不再描述旧路径。


定位与边界
----------

HABIT（**H**\ abitat **A**\ nalysis & **B**\ iomarker **I**\ dentification **T**\ oolkit）
是一个面向影像组学（radiomics）研究的 Python 包，覆盖以下三块独立但可拼接的能力：

1. **影像预处理**：把原始 DICOM/NIfTI 数据转成可建模输入。
2. **生境分析**：把影像内体素聚为亚区（habitat），输出每个 subject
   的生境图（NRRD）与表格（CSV）。
3. **机器学习**：在生境/影像组学特征上训练分类/回归模型，做评估、对比、
   可视化、报告。

三块能力之间通过 **数据契约** 解耦（一方写文件，另一方读文件），
而非互相 ``import`` 业务代码。这是理解整个仓库依赖方向的钥匙。


顶层包拓扑
----------

.. graphviz::

   digraph topology {
       rankdir=TB;
       node [shape=box, style="rounded,filled", fillcolor="#f6f8fa", fontname="Helvetica"];
       edge [color="#888"];

       subgraph cluster_entry {
           label="Entry Layer";
           style=dashed; color="#bbb";
           CLI    [label="habit.cli\n(Click root)"];
           API    [label="habit.__init__\n(lazy attribute API)"];
       }

       subgraph cluster_core {
           label="habit.core";
           style=dashed; color="#bbb";
           Common      [label="common\nBaseConfig + configurators"];
           HabitatCfg  [label="HabitatConfigurator"];
           MLCfg       [label="MLConfigurator"];
           PreCfg      [label="PreprocessingConfigurator"];
           Habitat     [label="habitat_analysis"];
           ML          [label="machine_learning"];
           Pre         [label="preprocessing"];
       }

       Utils [label="habit.utils\nio / log / progress\nparallel / viz"];

       CLI -> HabitatCfg [style=dotted];
       CLI -> MLCfg      [style=dotted];
       CLI -> PreCfg     [style=dotted];
       API -> Common;

       Common -> HabitatCfg;
       Common -> MLCfg;
       Common -> PreCfg;

       HabitatCfg -> Habitat;
       MLCfg      -> ML;
       PreCfg     -> Pre;

       Habitat -> Utils;
       ML      -> Utils;
       Pre     -> Utils;
       Common  -> Utils;
   }

依赖方向有四条硬规则：

* ``habit.utils`` 不依赖任何 ``core`` 子包；任何 core 子包都可以用 utils。
* ``habit.core.{habitat_analysis, machine_learning, preprocessing}`` **互不依赖**。
* ``habit.core.common`` 不在模块顶层 import 任何业务子包；
  装配通过 :py:mod:`habit.core.common.configurators` 下的三个域专用
  configurator (``HabitatConfigurator`` / ``MLConfigurator`` /
  ``PreprocessingConfigurator``) 完成，且它们的业务 import 全部 **延迟到
  factory 调用时**。这样 ``common`` 的 import 面只随真正使用的域增长。
* CLI / API 不应直接 import 业务子包的内部实现；统一经对应的
  configurator 拿装配好的对象。


入口三轨
--------

HABIT 同时暴露三套入口，覆盖不同使用场景：

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
       (``HabitatConfigurator`` / ``MLConfigurator`` /
       ``PreprocessingConfigurator``) 装配 → 调业务对象。这是 V1 的"标准用法"。
   * - **Scripts**
     - ``scripts/`` 下少量独立工具（``anonymize_dicom.py`` /
       ``app_dilation_or_erosion.py`` / ``app_km_survival.py`` /
       ``get_supervoxel.py`` / ``image2array.py`` /
       ``organize_image_data.py`` 等）
     - 与 CLI **不重叠** 的辅助脚本。V1 已删除所有与 CLI 重复的
       ``scripts/app_*.py`` 与 ``scripts/run_habitat_analysis.py``。
       新功能不应再加到这一层；除非真的与 CLI 能力不重叠。

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


服务装配：域专用 configurators
---------------------------------

V1 起，仓库不再有"一个上帝类"装配三个域。``habit/core/common/configurators/``
按业务域拆成三个并列的 configurator，全部继承同一个抽象基类
:py:class:`habit.core.common.configurators.base.BaseConfigurator`：

* :py:class:`habit.core.common.configurators.habitat.HabitatConfigurator`
  —— habitat 域的 ``create_feature_manager`` /
  ``create_clustering_manager`` / ``create_result_manager`` /
  ``create_habitat_analysis`` / ``create_habitat_map_analyzer`` /
  ``create_feature_extractor`` / ``create_radiomics_extractor`` /
  ``create_test_retest_analyzer``。
* :py:class:`habit.core.common.configurators.ml.MLConfigurator`
  —— ML 域的 ``create_evaluator`` / ``create_reporter`` /
  ``create_threshold_manager`` / ``create_plot_manager`` /
  ``create_metrics_store`` / ``create_model_comparison`` /
  ``create_ml_workflow``（同时覆盖 train + predict） /
  ``create_kfold_workflow``。
* :py:class:`habit.core.common.configurators.preprocessing.PreprocessingConfigurator`
  —— preprocessing 域的 ``create_batch_processor``。

设计要点：

* **域内深、域间隔**。三个 configurator 互不 import；CLI 子命令只
  挑自己需要的那一个。``common`` 不再在模块顶层 import 业务子包，
  业务 import 全部 **延迟到 factory 调用** 内部。
* **共享只放在基类**。日志接管、``output_dir`` 处理、轻量服务缓存
  (``get_service`` / ``register_service`` / ``clear_cache``) 留在
  ``BaseConfigurator``，避免三个子类重复实现同一份样板。
* **接口稳定**。每个 ``create_*`` 输入是已校验的 ``BaseConfig`` 子类
  实例，输出是同一个域对象，与旧 ``ServiceConfigurator.create_*``
  对应的方法名一致；CLI 替换工作量集中在 import 行。

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
* 模型字段、嵌套结构与跨字段约束用 Pydantic 的 ``@field_validator`` /
  ``@model_validator``  表达；例如
  :py:class:`HabitatAnalysisConfig` 在 ``run_mode == 'predict'`` 时强制要求
  ``pipeline_path``，并禁止 ``two_step`` 与 subject 级丢特征方法的冲突组合。
  :py:class:`MLConfig` 同样在 ``run_mode == 'predict'`` 时要求 ``pipeline_path``，
  在 ``run_mode == 'train'`` 时要求非空 ``models``。
* 路径字段在加载阶段统一解析为绝对路径，避免下游业务码做 ``os.path.abspath``。

历史例外（**已修复**）：``feature_selectors/icc/config.py`` 在 V1 已经迁到
``ICCConfig(BaseConfig)``，与其它配置走同一套加载路径
（``ICCConfig.from_file(yaml_path)``）。


核心子包：``habit.core.habitat_analysis``
------------------------------------------

定位
^^^^

把多模态影像在 ROI 内的体素聚为生境（habitat），输出 NRRD 生境图与
CSV 标签表，并把训练状态（聚类模型 + 预处理状态）持久化为
``habitat_pipeline.pkl`` 以供后续预测。

关键模块
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 模块
     - 职责
   * - ``habitat_analysis.py`` (deep)
     - V1 的 **唯一编排入口** 。一个类完成 build → fit/predict → 持久化 →
       结果后处理。内部用 ``_PIPELINE_RECIPES`` 字典按
       ``clustering_mode`` 分发 step 列表（``two_step`` / ``one_step``
       / ``direct_pooling``），用 ``_PIPELINE_MANAGER_ATTRS`` 显式
       白名单注入 manager 到加载后的 pipeline，杜绝反射。
   * - ``managers/{feature,clustering,result}_manager.py``
     - 三类领域职责：特征抽取与预处理、聚类训练/选择、结果落盘与可视化。
       通过构造时注入到 ``HabitatAnalysis``，再由后者注入到 pipeline 步。
   * - ``pipelines/base_pipeline.py``
     - sklearn 风格的 ``HabitatPipeline`` + ``BasePipelineStep``。
       ``fit_transform`` / ``transform`` / ``save`` / ``load`` 接口
       与 ``.pkl`` 格式是 V1 的稳定数据契约。
   * - ``pipelines/steps/*.py``
     - 7 个具体步：体素特征抽取、subject 预处理、个体聚类、
       supervoxel 特征/聚合、group 预处理、群体聚类、连接体素等。
   * - ``algorithms/``
     - K-Means / GMM / DBSCAN / SLIC / Hierarchical / MeanShift / Spectral
       共用的 ``BaseClusteringStrategy`` 接口（注意：此处的 strategy
       是 **聚类算法** 接口，**不是** 已废弃的旧 ``strategies/`` 子包）。
   * - ``extractors/``
     - 体素级 / supervoxel 级特征抽取实现，由 ``feature_extractor_factory``
       按配置选择。
   * - ``analyzers/habitat_analyzer.py``
     - ``HabitatMapAnalyzer``：在已生成的生境图上做 basic / radiomics /
       MSI / ITH 多进程特征抽取，写出 CSV，**不 import ML workflow**，
       与 ML 通过文件契约连接。
   * - ``config_schemas.py``
     - ``HabitatAnalysisConfig`` / ``FeatureExtractionConfig`` /
       ``RadiomicsConfig``。

deep / shallow 划分
^^^^^^^^^^^^^^^^^^^

* **Deep**：``HabitatAnalysis`` + ``HabitatPipeline``。这两块是子包对外的
  全部抽象，内部接口表面少（``fit``/``predict``/``run``、
  ``fit_transform``/``transform``/``save``/``load``），实现表面大。
* **Shallow**：``managers``、单个 step、单个 algorithm/extractor。
  这些都是「插件式」的薄壳，遵循固定接口。

数据流（训练）
^^^^^^^^^^^^^^

.. graphviz::

   digraph fit_flow {
       rankdir=LR;
       node [shape=box, style=rounded, fontname="Helvetica"];

       Imgs [label="images + masks\n(per subject)"];
       VF   [label="VoxelFeatureExtractor"];
       SP   [label="SubjectPreprocessing"];
       IC   [label="IndividualClustering\nvoxel -> supervoxel"];
       SA   [label="SupervoxelAggregation"];
       GP   [label="GroupPreprocessing"];
       PC   [label="PopulationClustering\nsupervoxel -> habitat"];
       Out  [label="habitat maps (.nrrd)\nhabitats.csv\nhabitat_pipeline.pkl",
             shape=note, fillcolor="#fff7e6", style="filled,rounded"];

       Imgs -> VF -> SP -> IC -> SA -> GP -> PC -> Out;
   }

数据流（预测）
^^^^^^^^^^^^^^

预测路径与训练共用同一个 ``HabitatPipeline``，只是：

1. 从磁盘 ``HabitatPipeline.load(pipeline_path)`` 反序列化训练好的 step。
2. ``HabitatAnalysis._inject_managers_into_pipeline`` 用
   ``_PIPELINE_MANAGER_ATTRS`` 白名单把当前运行时的 manager 注入到 step。
3. 强制把 ``pipeline.config.plot_curves = False``，避免预测期把 ``None``
   传给 cluster-selection 曲线绘制（V1 之前的已知 bug，已在重构中修复）。
4. 只调 ``transform``，不再 ``fit``。

外部产物
^^^^^^^^

* ``<out_dir>/habitats.csv`` —— habitat 标签表
* ``<out_dir>/habitat_*.nrrd`` —— 每 subject 生境图
* ``<out_dir>/habitat_pipeline.pkl`` —— joblib 序列化的训练 pipeline


核心子包：``habit.core.machine_learning``
------------------------------------------

定位
^^^^

读 CSV 特征表 → sklearn ``Pipeline`` 训练 / 交叉验证 / 预测 / 多模型对比 →
图、报告、模型 .pkl。**不依赖** habitat_analysis：habitat 表也好、传统
radiomics 表也好，都是同一份 CSV 契约。

关键模块
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 模块
     - 职责
   * - ``base_workflow.BaseWorkflow`` (deep)
     - workflow 的公共骨架：配置校验 + ``DataManager`` +
       ``PlotManager`` + ``PipelineBuilder`` + ``CallbackList``。
       子类（``MachineLearningWorkflow``、``MachineLearningKFoldWorkflow``）
       只覆写 ``run_pipeline``。
   * - ``workflows/holdout_workflow.MachineLearningWorkflow``
     - 训练 + 预测的统一执行体。``run_pipeline`` 按
       ``config_obj.run_mode`` 分发到 ``fit()`` / ``predict()``；
       ``predict`` 路径加载已训练 ``*_final_pipeline.pkl``，写
       ``prediction_results.csv``（开启 ``evaluate`` 时一并写
       ``evaluation_metrics.csv``）。**V1 已删除** 独立的
       ``PredictionWorkflow`` / ``PredictionConfig``。
   * - ``workflows/comparison_workflow.ModelComparison``
     - 多模型评估 + 可视化 + 报告生成。是组合
       ``MultifileEvaluator`` / ``Plotter`` / ``PlotManager`` /
       ``ThresholdManager`` / ``ReportExporter`` 的薄 facade，自身
       只承担 split 分组与 train -> test 阈值传递的编排逻辑。
   * - ``pipeline_utils.PipelineBuilder``
     - 把 YAML 中的 imputer / scaler / feature selector / model 配置
       串成 sklearn ``Pipeline``。``FeatureSelectTransformer`` 在此处
       把多段特征筛选嵌入到 sklearn 步里。
   * - ``models/factory.ModelFactory`` + ``models/*``
     - 用装饰器 ``@register`` 注册具体模型（LR/SVM/KNN/树系/XGBoost/
       MLP/NB/Ensemble/AutoGluon ...）。AutoGluon 走同一注册表，没有
       特殊路径。
   * - ``feature_selectors/selector_registry``
     - selector 元数据注册；``run_selector`` 按签名注入 ``X``/``y``
       /``selected_features``。
   * - ``evaluation/``
     - ``metrics`` 含 DeLong / Hosmer-Lemeshow / Spiegelhalter；
       ``prediction_container`` 统一预测容器；
       ``model_evaluation.MultifileEvaluator`` 多模型对比；
       ``threshold_manager`` 阈值策略。
   * - ``visualization/{plot_manager,plotting}.py``
     - ``PlotManager`` 按配置调度，``Plotter`` 画 ROC / PR / 校准等，
       底层用 ``habit.utils.visualization_utils``。
   * - ``reporting/report_exporter.py``
     - ``ReportExporter`` / ``MetricsStore`` 写 CSV/JSON 报告。
   * - ``callbacks/``
     - checkpoint（写 ``*_pipeline.pkl``）、report（写 summary CSV/JSON
       与 ``all_prediction_results.csv``）、visualization（挂钩 PlotManager）。

数据流（训练 / 预测 / 对比）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. graphviz::

   digraph ml_flow {
       rankdir=LR;
       node [shape=box, style=rounded, fontname="Helvetica"];

       CSV [label="features CSV\n(habitat or radiomics)"];

       subgraph cluster_train {
           label="train (Holdout / KFold)";
           style=dashed; color="#bbb";
           DM   [label="DataManager"];
           PB   [label="PipelineBuilder\n-> sklearn Pipeline"];
           CB   [label="CallbackList\nckpt + report + plot"];
           Mods [label="<model>_final_pipeline.pkl",
                 shape=note, fillcolor="#fff7e6", style="filled,rounded"];
           Rep  [label="*_summary.csv / *.json\nall_prediction_results.csv",
                 shape=note, fillcolor="#fff7e6", style="filled,rounded"];
       }

       subgraph cluster_predict {
           label="predict";
           style=dashed; color="#bbb";
           Load [label="load *_pipeline.pkl"];
           Pred [label="predict on new CSV"];
           POut [label="predictions JSON / CSV",
                 shape=note, fillcolor="#fff7e6", style="filled,rounded"];
       }

       subgraph cluster_compare {
           label="compare";
           style=dashed; color="#bbb";
           ME   [label="MultifileEvaluator"];
           MC   [label="ModelComparison\n+ Plotter"];
           COut [label="comparison report\n+ figures",
                 shape=note, fillcolor="#fff7e6", style="filled,rounded"];
       }

       CSV  -> DM -> PB -> CB;
       CB   -> Mods;
       CB   -> Rep;
       CSV  -> Load -> Pred -> POut;
       Mods -> Load;
       Rep  -> ME -> MC -> COut;
   }


核心子包：``habit.core.preprocessing``
---------------------------------------

定位
^^^^

把原始 DICOM/NIfTI 跑过一串可配置预处理步骤（重采样、配准、Z-score、
DICOM→NIfTI 等），输出标准化影像。

关键模块
^^^^^^^^

* ``image_processor_pipeline.BatchProcessor`` (deep)：按 subject 并行，
  调度 ``PreprocessorFactory`` 出来的步骤；负责日志、进度条与失败汇报。
* ``preprocessor_factory``：注册名 → 具体 preprocessor 实例。
* 各 preprocessor（``resample``、``registration``、``zscore``、
  ``dcm2niix``、``custom_preprocessor_template`` ...）共用 base 接口。
* ``config_schemas.PreprocessingConfig``：``Preprocessing`` 字段是
  「步骤名 → 步骤配置（含 images 列表）」的字典。

数据流极其线性：

.. code-block:: text

   raw images
     -> [step_1] -> [step_2] -> ... -> [step_n]
     -> standardized images on disk (per subject)


支撑包：``habit.core.common`` 与 ``habit.utils``
-------------------------------------------------

``common``
^^^^^^^^^^

业务无关的「装配 + 配置 + 横切」工具：

* ``configurators/`` —— 服务装配（见前文）。``base.py`` 给出抽象基类，
  ``habitat.py`` / ``ml.py`` / ``preprocessing.py`` 三个域专用子类。
* ``config_base.py`` —— Pydantic ``BaseConfig`` 基类与 ``from_file``。
* ``config_loader.py`` / ``config_validator.py`` —— YAML 加载与校验，
  早期非 ``BaseConfig`` 路径的回退；V1 起新加配置应直接继承
  ``BaseConfig``，少量历史脚本仍依赖这一对函数。
* ``dependency_injection.DIContainer`` —— 通用 DI 容器，目前仅定义、
  几乎未引用。
* ``dataframe_utils.py`` —— DataFrame/数组横切清洗。

``habit.utils``
^^^^^^^^^^^^^^^

业务零依赖的横切工具集合：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 模块
     - 职责
   * - ``io_utils``
     - 路径扫描、SITK / YAML / JSON / CSV 读写、配置 I/O。
   * - ``log_utils``
     - 日志初始化（与 ``configurators/`` 协作）。
   * - ``progress_utils``
     - 全仓库统一的 ``CustomTqdm`` 进度条。**所有业务代码必须用这个**，
       不要直接 ``from tqdm import tqdm``（用户规则）。
   * - ``parallel_utils``
     - 并行任务封装，自带进度条。
   * - ``visualization`` / ``visualization_utils``
     - 出版级绘图风格、SHAP 等可视化辅助；图上文字 **必须英文**
       （用户规则）。
   * - ``font_config``
     - 出版级字体配置。
   * - ``dicom_utils``
     - DICOM 扫描与元数据抽取（含并行）。
   * - ``dice_calculator``
     - Dice 分数批算（CLI ``habit dice`` 调用）。
   * - ``habitat_postprocess_utils``
     - habitat 图后处理（连通域等）。
   * - ``file_system_utils`` / ``import_utils`` / ``image_converter``
     - 文件系统、动态 import、影像格式转换的杂项辅助。

.. note::
   全仓进度条统一走 ``habit.utils.progress_utils.CustomTqdm``。
   ``habit/core/habitat_analysis/`` 之下不再有重复的 ``utils/progress_utils.py``。


序列化产物总览
--------------

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - 产物
     - 路径
     - 由谁写
   * - ``habitat_pipeline.pkl``
     - ``<habitat_out_dir>/``
     - ``HabitatAnalysis.fit``（joblib）
   * - ``habitats.csv``
     - ``<habitat_out_dir>/``
     - ``ResultManager``
   * - ``habitat_*.nrrd``
     - ``<habitat_out_dir>/``
     - ``ResultManager``
   * - ``<model>_final_pipeline.pkl``
     - ``<ml_out>/models/`` 或 fold 子目录
     - ``model_checkpoint`` callback（joblib）
   * - ``*_summary.csv`` / ``*_results.json`` / ``all_prediction_results.csv``
     - ``<ml_out>/``
     - ``report`` callback / ``ReportExporter``
   * - 预处理后影像
     - ``<preprocess_out>/<subject>/``
     - ``BatchProcessor``


设计原则与不变量
----------------

阅读和修改本仓库时，请遵守下列不变量；它们是 V1 重构沉淀下来的约定：

1. **三业务子包互不 import**。要拼接，从对应的域 configurator
   (``HabitatConfigurator`` / ``MLConfigurator`` /
   ``PreprocessingConfigurator``) 装配，或者通过文件契约
   （CSV / NRRD / .pkl）。
2. **配置一律走 Pydantic** 。新加配置请继承 ``BaseConfig``，不要再用
   裸 dict + ``validate_config``。
3. **进度条一律用** ``habit.utils.progress_utils.CustomTqdm``。
4. **图上文字一律用英文**（matplotlib / graphviz / mermaid 节点标签均如此）。
   文档正文可以是中文。
5. **manager 注入用显式白名单**（``_PIPELINE_MANAGER_ATTRS``），
   不再用 ``dir(self)`` 反射。新增 manager 必须刻意修改这个常量。
6. **训练 / 预测共用同一 pipeline 类**。预测路径只是「load + 注入 +
   transform」，不应复刻一套独立的执行栈。habitat 与 ML 两侧都已落地
   （``HabitatAnalysis.fit/predict`` / ``MachineLearningWorkflow.fit/predict``，
   各自由 ``run_mode`` 字段分发）。


已知架构关切
-------------

V1 架构重构已经按 ``docs/code_review/architecture_refactor_plan.md`` 中
列出的 7 项 deepening candidate 全部落地：

* ``habitat_analysis`` 三层（控制器 / 策略 / 构造器）合一，``strategies/``
  与 ``pipelines/pipeline_builder.py`` 已删除；mode 分支收敛到
  ``_PIPELINE_RECIPES`` 字典。
* ML 训练 / 预测共用同一 ``MLConfig`` 与 ``MachineLearningWorkflow``；
  ``PredictionWorkflow`` 与独立的 ``PredictionConfig`` 已删除。
* ``ModelComparison`` 已是薄 facade（修复了 ``setup`` 缩进缺陷、统一
  图字英文、清理 dead import）。
* ``scripts/app_*.py`` 与 ``scripts/run_habitat_analysis.py`` 等双轨入口
  全部删除；CLI ``habit <subcommand>`` 是唯一推荐入口。
* ``habit/core/__init__.py`` / ``habit/core/habitat_analysis/__init__.py``
  改为 fail-fast；可选依赖通过 ``habit.is_available(name)`` /
  ``habit.import_error(name)`` 显式查询。
* ICC 配置改为 ``ICCConfig(BaseConfig)``。
* ``habit/core/habitat_analysis/utils/progress_utils.py`` 误引用已修复。
* ``ServiceConfigurator`` 已拆为 ``HabitatConfigurator`` /
  ``MLConfigurator`` / ``PreprocessingConfigurator``（继承
  ``BaseConfigurator``）；旧单类已删除，每个 cmd_* 子命令只 import
  自己使用的域。

V1 架构层面无遗留关切。后续若再发现「域间隐式 import」「configurator
被业务子包反向 import」等问题，应记入新一轮 review 而非保留在这一节。


相关文档
--------

* 各模块代码架构与维护入口：``docs/source/development/module_architecture.rst``
* habitat_analysis 子包内部：``habit/core/habitat_analysis/ARCHITECTURE.md``
* habitat pipeline 步骤设计：``habit/core/habitat_analysis/PIPELINE_DESIGN.md``
* 旧三层 → 新深模块的重构记录：
  ``docs/code_review/habitat_analysis_review.md`` 与
  ``docs/code_review/habitat_analysis_refactor_step1.md``
* 用户文档（用法）：``docs/source/user_guide/``
