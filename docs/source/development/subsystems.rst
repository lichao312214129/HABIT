子系统详解
==========

本页深入两个最复杂的子系统：生境分析（habitat_analysis）与机器学习（machine_learning），
重点讲它们的内部编排与执行流。其余子系统（预处理、DICOM 排序）结构较直接，可参考 :doc:`repo_layout`。

生境分析
--------

生境分析把体素级影像特征逐层聚合成 "生境（habitat）"——肿瘤内部影像表型相近的子区域。
它以 **流水线（pipeline）+ 步骤（step）** 的方式组织，编排核心在
``habitat_analysis/habitat_analysis.py`` 的 ``HabitatAnalysis`` 类。

子目录职责
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 目录
     - 职责
   * - ``clustering/``
     - 聚类算法（kmeans / gmm / slic）+ 最优簇数搜索与验证指标。
   * - ``clustering_features/``
     - 体素 / 超体素级特征提取器（raw / kinetic / radiomics / local_entropy 等）。
   * - ``feature_preprocessing/``
     - 组级特征表（DataFrame）预处理（归一化、方差/相关性过滤等）。
   * - ``habitat_features/``
     - **分割后** 特征：``HabitatMapAnalyzer`` 及 radiomics / MSI / ITH 等。
   * - ``pipelines/`` 与 ``pipelines/steps/``
     - ``HabitatPipeline`` 的步骤链与各步骤实现。
   * - ``services/``
     - 领域服务层（特征服务、聚类服务、生境图写出、结果发布），供各步骤调用。
   * - ``checkpoint/``
     - 训练断点续跑（个体级）。

流水线编排：三种聚类策略
~~~~~~~~~~~~~~~~~~~~~~~~

``HabitatAnalysis`` 依据配置 ``HabitatSegmentation.clustering_mode`` 选择不同的步骤序列。
三种策略的映射写在模块级 ``_PIPELINE_RECIPES``\ （单一事实来源），分别由
``_build_two_step_steps`` / ``_build_one_step_steps`` / ``_build_pooling_steps`` 构建。

**two_step（默认，多受试者研究）**：两层聚类——先在每个受试者内把体素聚成超体素，再在群体层面把超体素聚成生境。

.. mermaid::

   flowchart TD
     A["voxel_features"] --> B["individual_preprocessing"]
     B --> C["individual_clustering<br/>target = supervoxel (per subject)"]
     C --> D["calculate_mean_voxel_features"]
     D --> E{"advanced supervoxel<br/>features?"}
     E -->|yes| F["supervoxel_advanced_features"]
     E -->|no| G["merge_supervoxel_features"]
     F --> G
     G --> H["combine_supervoxels<br/>(all subjects)"]
     H --> I{"group preprocessing?"}
     I -->|yes| J["group_preprocessing"]
     I -->|no| K["group_clustering<br/>supervoxel -> habitat"]
     J --> K

**one_step（单肿瘤探索）**：不做群体聚类，直接在个体层把体素聚成生境。

.. mermaid::

   flowchart TD
     A["voxel_features"] --> B["individual_preprocessing"]
     B --> C["individual_clustering<br/>target = habitat (find optimal k)"]
     C --> D["calculate_mean_voxel_features"]
     D --> E["merge_supervoxel_features"]
     E --> F["combine_supervoxels"]

**direct_pooling（跨受试者体素池化）**：跳过个体超体素层，把所有受试者体素拼在一起统一聚类。

.. mermaid::

   flowchart TD
     A["voxel_features"] --> B["individual_preprocessing"]
     B --> C["concatenate_voxels<br/>(pool all subjects)"]
     C --> D{"group preprocessing?"}
     D -->|yes| E["group_preprocessing"]
     D -->|no| F["group_clustering<br/>on pooled voxels"]
     E --> F

训练与预测
~~~~~~~~~~

- **train**：``HabitatAnalysis.fit()`` 按策略构建 ``HabitatPipeline`` 并 ``fit_transform``，
  然后把整条流水线序列化为 ``habitat_pipeline.pkl``\ （聚类中心、预处理参数等）。
- **predict**：``predict()`` 加载 pkl，向其中注入白名单内的服务（特征服务、聚类服务、生境图写出器），
  再对新受试者 ``transform``，保证与训练完全一致的处理。
- **resume**：训练支持个体级断点续跑（``checkpoint/``），大队列中断后可继续。

特征提取（分割后）
~~~~~~~~~~~~~~~~~~

生境图生成后，``habit extract`` 通过 ``HabitatMapAnalyzer``\ （``habitat_features/``）计算下游特征；
``habit radiomics`` 通过 ``TraditionalRadiomicsExtractor`` 计算传统影像组学。内置特征类型见
:doc:`extension_points`；各特征的学术定义见 :doc:`../reference/features/index`。

机器学习
--------

机器学习子系统处理表格数据：装配特征表 → 划分 → 构建 sklearn 流水线 → 训练/评估 → 出报告与图。

模块地图
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - 模块
     - 职责
   * - ``data_manager.py``
     - ``DataManager``：读多张表、合并、train/test 划分（含分层）。
   * - ``pipeline_builder.py``
     - ``PipelineBuilder``：构建 sklearn Pipeline（特征选择 → 标准化 → 重采样 → 模型）。
   * - ``models/``
     - ``BaseModel`` + ``ModelFactory`` + 各模型实现（逻辑回归、SVM、XGBoost、AutoGluon、集成等）。
   * - ``feature_selectors/``
     - 各 ``@SelectorRegistry.register`` 选择器；``icc/`` 子包含 ICC / 重测分析。
   * - ``workflows/``
     - ``HoldoutWorkflow`` / ``KFoldWorkflow`` / ``ModelComparison`` —— 面向命令的高层流程。
   * - ``runners/``
     - ``HoldoutRunner`` / ``KFoldRunner`` / ``InferenceRunner`` —— 实际训练/推理执行。
   * - ``contracts/``
     - ``WorkflowPlan``\ （不可变配置快照）与 ``WorkflowResult`` 协议，统一结果结构。
   * - ``evaluation/``
     - 指标、阈值管理、预测容器、模型评估。
   * - ``reporting/`` / ``visualization/``
     - 报告导出、图表编排（ROC、校准、DCA、KM 等，**图内英文**）。
   * - ``statistics/``
     - DeLong、Hosmer-Lemeshow、Spiegelhalter-Z 等统计检验。
   * - ``resampling.py``
     - 过采样/欠采样/SMOTE。

执行流（Holdout 为例）
~~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   flowchart TD
     CFG["MLConfig (validated)"] --> WF["HoldoutWorkflow.run()"]
     WF --> PLAN["WorkflowPlan<br/>(immutable snapshot)"]
     PLAN --> RUN["HoldoutRunner / InferenceRunner"]
     RUN --> DM["DataManager<br/>assemble X/y, split"]
     DM --> PB["PipelineBuilder<br/>selector -> scaler -> resampler -> model"]
     PB --> FIT["fit / evaluate"]
     FIT --> RES["WorkflowResult"]
     RES --> REP["reporting + visualization<br/>models, metrics, plots"]

设计要点：

- **workflow 与 runner 分离**：workflow 负责 "做什么"（编排、产出结构），runner 负责 "怎么执行"（训练/推理细节）。
- **``WorkflowPlan`` 不可变**：一旦生成即冻结配置快照，保证运行期一致、可复现。
- **``WorkflowResult`` 协议统一**：holdout / k-fold / 推理三种结果都满足同一协议，报告层可统一消费。

辅助分析
~~~~~~~~

ICC（``habit icc``）与 test-retest（``habit retest``）复用特征选择子系统中的 ``feature_selectors/icc/``，
用于评估特征的可重复性，常作为建模前的特征筛选依据。

.. seealso::

   - 命令与配置的用户视角说明见 :doc:`../how_to/index` 与 :doc:`../configuration/index`。
   - 如何新增算法组件见 :doc:`extension_points` 与 :doc:`../customization/index`。
