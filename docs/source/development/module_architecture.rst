模块代码架构说明
================

本文档面向 HABIT 开发者，用于快速理解各主要模块的代码组织、调用链、
数据契约和扩展点。整体依赖规则与包级拓扑见
:doc:`architecture`；本文更关注“读代码时从哪里开始、改功能时改哪里”。

总览
----

HABIT 的主干能力可以按下面的数据链理解：

.. code-block:: text

   CLI / Python API
     -> Domain Configurator
     -> Domain Workflow / Pipeline
     -> CSV / NRRD / PKL artifacts
     -> downstream workflow

三个业务域之间不直接 import 彼此的内部实现：

* ``habit.core.preprocessing`` 负责图像预处理，输出标准化影像。
* ``habit.core.habitat_analysis`` 负责 habitat 分割、habitat map 与特征表。
* ``habit.core.machine_learning`` 负责读取 CSV 特征并训练、预测、比较模型。

它们通过文件产物衔接，而不是通过跨域对象引用衔接。跨域共享能力放在
``habit.core.common`` 和 ``habit.utils``。


入口与装配层
------------

关键文件
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 文件
     - 职责
   * - ``habit/cli.py``
     - Click 根入口，集中注册所有 ``habit <subcommand>`` 命令。
   * - ``habit/cli_commands/commands/cmd_*.py``
     - 业务命令的薄封装：加载配置、创建 configurator、调用业务对象。
   * - ``habit/core/common/configurators/base.py``
     - configurator 公共基类，处理 logger、输出目录和轻量服务缓存。
   * - ``habit/core/common/configurators/habitat.py``
     - 装配 habitat 分析、habitat 特征抽取、传统 radiomics、test-retest。
   * - ``habit/core/common/configurators/ml.py``
     - 装配 ML workflow、KFold、模型比较、评估、报告、绘图组件。
   * - ``habit/core/common/configurators/preprocessing.py``
     - 装配图像预处理 ``BatchProcessor``。

标准调用链
^^^^^^^^^^

.. code-block:: text

   habit <command> -c config.yaml
     -> habit/cli.py
     -> cmd_<domain>.run_*
     -> ConfigClass.from_file(...)
     -> DomainConfigurator(...)
     -> create_<service>()
     -> service.run / fit / predict / process_batch

命令组织有两类：

* **业务流程命令**：``preprocess``、``get-habitat``、``extract``、``model``、
  ``cv``、``compare``、``icc``、``radiomics``、``retest``。这些命令通常
  走 YAML 配置和 domain configurator。
* **辅助工具命令**：``dicom-info``、``merge-csv``、``dice``。它们主要由命令行
  参数驱动，``dicom-info`` 和 ``merge-csv`` 有对应 ``cmd_*.py``，``dice``
  直接从 ``habit/cli.py`` 调用 ``habit.utils.dice_calculator``。

维护注意事项
^^^^^^^^^^^^

* 新增业务命令时，优先新增 ``cmd_<name>.py``，再在 ``habit/cli.py`` 注册。
* 新增需要 YAML 的业务服务时，先定义 ``BaseConfig`` 子类，再在对应
  domain configurator 添加 ``create_*`` 方法。
* configurator 内部的业务 import 应放在 factory 方法内部，避免
  ``habit.core.common`` 顶层 import 业务子包。


``habit.core.preprocessing``
----------------------------

模块定位
^^^^^^^^

``preprocessing`` 子包负责把原始 DICOM/NIfTI 数据变成后续 habitat 或
radiomics 可直接读取的标准化影像。它是图像层面的批处理 pipeline，
不要和 ``habitat_analysis`` 内部的 subject/group feature preprocessing 混淆。

关键文件
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 文件
     - 职责
   * - ``config_schemas.py``
     - ``PreprocessingConfig`` 和每个步骤配置的 Pydantic schema。
   * - ``image_processor_pipeline.py``
     - ``BatchProcessor``，按 subject 并行调度整条预处理链。
   * - ``base_preprocessor.py``
     - ``BasePreprocessor``，所有具体预处理器的统一接口。
   * - ``preprocessor_factory.py``
     - ``PreprocessorFactory``，通过注册名创建具体 preprocessor。
   * - ``load_image.py``
     - ``LoadImagePreprocessor``，把路径读取成 ``SimpleITK.Image``。
   * - ``dcm2niix_converter.py``
     - ``Dcm2niixConverter``，调用外部 dcm2niix 执行 DICOM 转换。
   * - ``resample.py`` / ``registration.py`` / ``n4_correction.py`` 等
     - 具体预处理步骤，均通过 factory 注册。

数据流
^^^^^^

.. code-block:: text

   PreprocessingConfig
     -> PreprocessingConfigurator.create_batch_processor()
     -> BatchProcessor.process_batch()
     -> LoadImagePreprocessor
     -> PreprocessorFactory.create(step_name)
     -> BasePreprocessor.__call__(subject_data)
     -> processed images on disk

``BatchProcessor`` 的核心循环是按 subject 处理。每个 subject 的数据放在
``subject_data`` 字典中，常见键包括：

* ``subj``：subject ID。
* 影像 modality 键，例如 ``delay2``、``t1``、``flair``。
* mask 键，例如 ``mask_delay2`` 或配置中指定的 mask 名。
* ``output_dirs``：当前 subject 的输出目录集合。

配置中的 ``Preprocessing`` 字段是有序的「步骤名 -> 参数」映射。步骤名必须
等于 ``PreprocessorFactory`` 中注册的名字；每个步骤通常通过 ``images``
字段指定要处理的 modality。

扩展点
^^^^^^

新增图像预处理步骤时：

1. 新建继承 ``BasePreprocessor`` 的类。
2. 实现 ``__call__(self, data: Dict[str, Any]) -> Dict[str, Any]``。
3. 用 ``PreprocessorFactory.register("step_name")`` 注册。
4. 确保模块会被 ``habit.core.preprocessing`` 导入，否则注册装饰器不会执行。
5. 在配置模板或用户文档中补充该 ``step_name`` 的参数说明。

维护注意事项
^^^^^^^^^^^^

* ``LoadImagePreprocessor`` 是 ``BatchProcessor`` 的隐式前置步骤，通常不需要
  用户在 YAML 中写 ``load_image``。
* 预处理器应尽量只读写 ``subject_data`` 中自己声明处理的键，避免隐式修改
  其它步骤依赖的数据。
* 进度条统一使用 ``habit.utils.progress_utils.CustomTqdm`` 或经
  ``parallel_utils`` 间接使用。


``habit.core.habitat_analysis``
-------------------------------

模块定位
^^^^^^^^

``habitat_analysis`` 子包负责从多模态影像和 ROI mask 中生成 habitat map、
habitat 标签表和可复用的训练 pipeline。它也包含 habitat map 后续特征抽取
和传统 radiomics 抽取的实现。

关键文件
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 文件
     - 职责
   * - ``habitat_analysis.py``
     - ``HabitatAnalysis`` deep module，负责 build、fit、predict、持久化和结果后处理。
   * - ``config_schemas.py``
     - ``HabitatAnalysisConfig``、``FeatureExtractionConfig``、``RadiomicsConfig`` 等。
   * - ``pipelines/base_pipeline.py``
     - ``HabitatPipeline``、``BasePipelineStep``、``IndividualLevelStep``、``GroupLevelStep``。
   * - ``pipelines/steps/*.py``
     - 体素特征、subject 预处理、个体聚类、supervoxel 聚合、群体聚类等步骤。
   * - ``managers/feature_manager.py``
     - 特征抽取、特征预处理、supervoxel 级特征计算。
   * - ``managers/clustering_manager.py``
     - 聚类算法创建、最佳聚类数选择、训练状态保存。
   * - ``managers/result_manager.py``
     - habitat/supervoxel 图像和 CSV 结果落盘。
   * - ``algorithms/``
     - KMeans、GMM、DBSCAN、SLIC 等聚类策略接口实现。
   * - ``extractors/``
     - voxel/supervoxel 级特征抽取器与表达式解析。
   * - ``analyzers/``
     - 已生成 habitat map 上的特征提取、传统 radiomics 等分析器。

Pipeline 结构
^^^^^^^^^^^^^

``HabitatAnalysis`` 通过 ``_PIPELINE_RECIPES`` 按
``HabitatsSegmention.clustering_mode`` 选择 recipe：

* ``two_step``：voxel -> supervoxel -> population habitat。
* ``one_step``：每个 subject 内直接 voxel -> habitat。
* ``direct_pooling``：跨 subject pooling 后统一聚类。

recipe 返回 ``(name, step)`` 列表，交给 ``HabitatPipeline``。pipeline 自动把
step 分成两类：

* ``IndividualLevelStep``：逐 subject 处理，可并行。
* ``GroupLevelStep``：所有 subject 汇总后处理。

训练数据流
^^^^^^^^^^

.. code-block:: text

   images + masks
     -> VoxelFeatureExtractor
     -> SubjectPreprocessingStep
     -> IndividualClusteringStep
     -> CalculateMeanVoxelFeaturesStep / SupervoxelFeatureExtractionStep
     -> MergeSupervoxelFeaturesStep
     -> CombineSupervoxelsStep / ConcatenateVoxelsStep
     -> GroupPreprocessingStep
     -> PopulationClusteringStep
     -> habitats.csv + habitat maps + habitat_pipeline.pkl

预测数据流
^^^^^^^^^^

预测不会重新 fit pipeline：

1. ``HabitatPipeline.load(pipeline_path)`` 读取训练产物。
2. ``HabitatAnalysis._inject_managers_into_pipeline`` 用
   ``_PIPELINE_MANAGER_ATTRS`` 白名单注入当前运行时 manager。
3. ``FeatureManager`` 只更新当前数据路径和日志目标，保留已训练状态。
4. 调用 ``pipeline.transform(...)`` 生成新数据结果。

扩展点
^^^^^^

新增 habitat 聚类模式时：

* 新增一个 recipe 函数，返回明确的 step 列表。
* 把模式名加入 ``_PIPELINE_RECIPES``。
* 如需新的输出后处理，集中放在 ``HabitatAnalysis`` 或 ``ResultManager``，
  不要在 CLI 层分叉。

新增 pipeline step 时：

* 继承 ``IndividualLevelStep`` 或 ``GroupLevelStep``。
* 明确 ``transform`` 的输入/输出结构。
* 需要共享服务时，通过构造函数接收 manager，不要在 step 内重新创建 manager。

维护注意事项
^^^^^^^^^^^^

* 旧 ``strategies/`` 子包和 ``pipeline_builder.py`` 已删除；不要按旧三层结构
  新增代码。
* manager 注入用显式白名单 ``_PIPELINE_MANAGER_ATTRS``；新增 manager 必须主动
  修改白名单和注入逻辑。
* 更细的 habitat 内部说明见 ``habit/core/habitat_analysis/ARCHITECTURE.md`` 和
  ``habit/core/habitat_analysis/PIPELINE_DESIGN.md``。


``habit.core.machine_learning``
-------------------------------

模块定位
^^^^^^^^

``machine_learning`` 子包读取 CSV 特征表，完成训练、预测、K 折验证、模型比较、
统计检验、绘图和报告输出。它不关心 CSV 来自 habitat、传统 radiomics 还是临床表。

关键文件
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 文件
     - 职责
   * - ``config_schemas.py``
     - ``MLConfig``、模型配置、特征选择配置、比较配置等 schema。
   * - ``base_workflow.py``
     - ``BaseWorkflow``，公共骨架：配置、数据管理、pipeline builder、callbacks。
   * - ``workflows/holdout_workflow.py``
     - ``MachineLearningWorkflow``，训练和预测的主 workflow。
   * - ``workflows/kfold_workflow.py``
     - ``MachineLearningKFoldWorkflow``，K 折交叉验证。
   * - ``workflows/comparison_workflow.py``
     - ``ModelComparison``，多模型结果评估、绘图、报告。
   * - ``data_manager.py``
     - 读取、合并、切分 CSV 数据。
   * - ``pipeline_utils.py``
     - ``PipelineBuilder`` 和 ``FeatureSelectTransformer``。
   * - ``models/factory.py`` / ``models/*.py``
     - ``ModelFactory`` 和具体模型实现。
   * - ``feature_selectors/``
     - selector 注册表、``run_selector``、ICC 等特征选择实现。
   * - ``evaluation/`` / ``visualization/`` / ``reporting/`` / ``callbacks/``
     - 评估、绘图、报告和训练过程回调。

训练调用链
^^^^^^^^^^

.. code-block:: text

   MLConfig
     -> MLConfigurator.create_ml_workflow()
     -> MachineLearningWorkflow.run_pipeline()
     -> DataManager
     -> PipelineBuilder
     -> FeatureSelectTransformer
     -> scaler / imputer
     -> ModelFactory.create_model()
     -> callbacks
     -> model PKL + reports + figures

``PipelineBuilder`` 负责把 YAML 配置转为 sklearn ``Pipeline``。常见顺序是：

.. code-block:: text

   imputer
     -> feature selection before scaling
     -> scaler
     -> feature selection after scaling
     -> model

``FeatureSelectTransformer`` 在 sklearn pipeline 内调用 ``run_selector``。
selector 注册表会按 selector 函数签名注入 ``X``、``y``、
``selected_features``、``outdir`` 等参数。

预测和比较
^^^^^^^^^^

* 预测模式仍使用 ``MachineLearningWorkflow``，由 ``MLConfig.run_mode`` 分发。
  预测时加载训练好的 ``*_final_pipeline.pkl``，输出预测结果和可选评估指标。
* ``ModelComparison`` 不训练模型，只读取多个模型的预测结果，组合
  ``MultifileEvaluator``、``Plotter``、``ThresholdManager``、``ReportExporter``
  生成比较图、统计检验和报告。

扩展点
^^^^^^

新增模型时：

1. 继承项目模型基类或兼容 sklearn estimator 接口。
2. 用 ``ModelFactory.register("model_name")`` 注册。
3. 在配置 schema 或模板中补充可用参数。
4. 若模型需要额外依赖，应保持可选依赖的 import 边界清楚。

新增特征选择方法时：

1. 在 ``feature_selectors`` 中实现 selector 函数。
2. 用注册机制加入 ``selector_registry``。
3. 明确 selector 属于 scaling 前还是 scaling 后。
4. 保持输入输出为特征名列表或可被 ``FeatureSelectTransformer`` 消费的结构。

维护注意事项
^^^^^^^^^^^^

* V1 中训练和预测共用 ``MLConfig`` 与 ``MachineLearningWorkflow``；
  不要恢复独立 ``PredictionWorkflow``。
* 绘图文字必须使用英文。
* workflow 之外的评估、绘图、报告逻辑尽量放入 ``evaluation``、
  ``visualization``、``reporting`` 或 callback，保持 workflow 只负责编排。


``habit.core.common``
---------------------

模块定位
^^^^^^^^

``common`` 是 core 内的共享基础设施层，主要处理配置、服务装配和少量横切工具。
它可以被业务域使用，但不应在模块顶层 import 任一业务域的重型实现。

关键文件
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 文件
     - 职责
   * - ``config_base.py``
     - ``BaseConfig``、统一 ``from_file`` / ``from_dict`` / ``to_dict``。
   * - ``config_loader.py``
     - YAML/JSON 读写和路径解析。
   * - ``config_validator.py``
     - 历史兼容的配置校验入口。
   * - ``config_accessor.py``
     - 兼容 dict 与 Pydantic model 的访问辅助。
   * - ``configurators/``
     - 域专用服务装配入口。
   * - ``dependency_injection.py``
     - 通用 DI 容器，目前不是主路径。
   * - ``dataframe_utils.py``
     - 表格和数组清洗辅助。

边界
^^^^

* ``config_loader`` 解决“配置文件如何读、路径如何解析”。
* ``BaseConfig`` 和各 domain schema 解决“配置结构是否合法”。
* ``configurators`` 解决“已验证配置如何组装成可运行服务”。

不要把业务运行逻辑放进 ``common``。如果一个函数需要理解 habitat 或 ML 的领域
语义，它应该留在对应 domain 子包。


``habit.utils``
---------------

模块定位
^^^^^^^^

``utils`` 是全包可用的横切工具层。它不依赖 ``habit.core``，因此可以被
preprocessing、habitat、ML 和 CLI 辅助工具共同使用。

常见工具
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 模块
     - 职责
   * - ``progress_utils.py``
     - 全项目统一进度条 ``CustomTqdm``。
   * - ``parallel_utils.py``
     - 并行 map 封装，整合进度条和错误收集。
   * - ``log_utils.py``
     - 日志初始化和子进程日志恢复。
   * - ``io_utils.py``
     - 图像/表格/配置的通用读写、image/mask 路径扫描。
   * - ``dicom_utils.py``
     - DICOM 元信息扫描与批量提取。
   * - ``dice_calculator.py``
     - Dice 系数批量计算。
   * - ``visualization_utils.py`` / ``visualization.py``
     - 绘图风格、SHAP 和通用可视化辅助。
   * - ``habitat_postprocess_utils.py``
     - habitat map 连通域等后处理工具。
   * - ``file_system_utils.py`` / ``import_utils.py`` / ``image_converter.py``
     - 文件、动态 import、影像格式转换辅助。

维护注意事项
^^^^^^^^^^^^

* 开发 ``habit`` 包时，通用工具统一放在 ``habit/utils``。
* 进度条必须使用 ``CustomTqdm`` 或 ``parallel_utils``，不要直接引入
  ``tqdm``。
* 绘图函数和图中标签必须使用英文。
* ``utils`` 中不要 import 业务 domain；如果工具开始依赖领域语义，应迁回
  对应 domain 子包。


辅助工具命令
------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 命令
     - 实现位置
     - 说明
   * - ``habit dicom-info``
     - ``cmd_dicom_info.py`` + ``habit/utils/dicom_utils.py``
     - 扫描 DICOM 文件、列出或导出指定 tag。
   * - ``habit merge-csv``
     - ``cmd_merge_csv.py``
     - 按索引列横向合并 CSV/Excel。
   * - ``habit dice``
     - ``habit/cli.py`` + ``habit/utils/dice_calculator.py``
     - 批量计算两个 mask 集合之间的 Dice 系数。

这些命令不走 domain configurator，因为它们没有复杂的业务服务装配需求。
如果未来某个辅助命令开始拥有长期配置、多个服务和稳定产物，应考虑迁到
``cmd_*.py`` + config schema + configurator 的标准业务命令模式。


开发者阅读顺序
--------------

建议按下面顺序阅读代码：

1. 先读 :doc:`architecture`，掌握全局依赖方向。
2. 根据要改的功能选择一个 domain：

   * 图像预处理：从 ``BatchProcessor`` 和 ``PreprocessorFactory`` 开始。
   * habitat 分割：从 ``HabitatAnalysis`` 和 ``_PIPELINE_RECIPES`` 开始。
   * ML 建模：从 ``MachineLearningWorkflow``、``BaseWorkflow``、
     ``PipelineBuilder`` 开始。

3. 再深入到具体 step、model、selector、extractor。
4. 最后查看对应配置 schema 和配置模板，确认用户可配置面。

维护时优先保持“入口薄、domain 内聚、domain 间通过文件契约解耦”的结构。
