生境分割
========

本节介绍如何使用 HABIT 进行生境分割，这是HABIT 的核心功能。

.. seealso::

   聚类算法、指标与 sklearn 参数详见 :doc:`../reference/upstream_libraries_zh`（**scikit-learn**）；预处理步骤中的ANTs / SimpleITK 见同页。

概述
----

生境分割是将肿瘤分割为多个具有相似特征的区域（生境）的过程。HABIT 提供了三种聚类策略，支持灵活的特征提取和自定义扩展。

**生境分析的核心思想：**

1. **肿瘤异质性**: 肿瘤不是均匀的组织，而是由多个具有不同特征的亚区域组成
2. **功能分区**: 不同的生境可能对应不同的生物学过程（如血管生成、坏死、炎症等）
3. **临床意义**: 生境特征可以提供比整体肿瘤特征更丰富的诊断信息

**三种聚类策略：**

1. **One-Step 策略**: 个体级别聚类，每个肿瘤独立进行体素到生境的聚类。
2. **Two-Step 策略（推荐）**: 两步聚类，先体素到超像素（个体级），再超像素到生境（群体级）。
3. **Direct Pooling 策略**: 直接池化策略。该策略将所有受试者的所有体素特征一次性拼接（Pooling）到一个巨大的特征矩阵中进行聚类。这种方法能够发现群体中最具代表性的组织模式，且由于生境发现过程通常是无监督的，在特征空间进行的 Pooling 操作并不一定意味着标签信息的泄露。

结果解读指南
------------

运行 `get-habitat` 后，您会得到一系列结果文件。以下是如何解读这些结果的临床指南。

1. 生境地图 (*_habitats.nrrd)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

这是最直观的结果。您可以使用 **ITK-SNAP** 或**3D Slicer** 打开原始 MRI 图像，然后将此文件拖入作个Segmentation（或 Overlay）。

*   **五颜六色的区域是什么？**
    每一个颜色代表一个“生境”（Habitat）。例如，红色可能代表 Cluster 1，蓝色代表 Cluster 2。
*   **如何赋予临床意义：**
    您需要结合原始影像的信号特征来解读。
    *   如果 Cluster 1 在T1 增强序列上信号很高，在ADC 图上信号很低，它可能代表 **活性肿瘤区**。
    *   如果 Cluster 2 在T2 序列上信号高，无强化，可能代表 **坏死区** 或**水肿区**。

2. 聚类验证图(visualizations/optimal_clusters/)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

该图用于所有需要自动选择聚类数的场景（包括One-Step 和Two-Step 的相应聚类阶段）。

*   **横轴**：尝试的聚类数量（k=2, 3, 4...）。
*   **纵轴**：聚类评价指标（如Inertia 或Silhouette Score）。
*   **如何看？**
*   **Inertia/Kneedle**：``inertia`` 是不同 k 下的簇内平方误差曲线；``kneedle`` 是在该曲线上自动寻找拐点的策略。当前实现中，选择 ``inertia`` 或 ``kneedle`` 都会基于 Inertia 曲线使用 Kneedle 选拐点。
*   **轮廓系数 (Silhouette)**：看哪个 k 的分数最高。

3. 聚类可视化图 (visualizations/*_clustering/)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

聚类可视化通常包括，

*   **one_step_clustering/**：One-Step 的生境聚类可视化
*   **supervoxel_clustering/**：Two-Step 的超像素聚类可视化
*   **habitat_clustering/**：Two-Step/Direct Pooling 的生境聚类可视化

生境分析中的特征处理
--------------------

在生境分析中，特征处理的目的是消除不同扫描设备或个体差异带来的噪声，同时保留具有生物学意义的组织差异。

**1. 个体级别处理 (Subject-level)**
   *   **目的**：消除个体内的异常值（如Winsorize 去极值）和进行初步的标准化（如Min-Max 归一化）。
   *   **意义**：确保每个受试者的影像特征在进入群体分析前具有可比的数值量级，避免某个病例因为信号强度极高而主导聚类结果。

**2. 群体级别处理 (Group-level)**
   *   **目的**：在所有受试者的特征池化后进行进一步处理（如Binning 离散化）。
   *   **意义**：通过离散化减少微小噪声的影响，使聚类算法更容易捕捉到稳定的组织模式（如“高强化区—vs “低强化区”），从而提高生境发现的鲁棒性。

CLI 使用方法
------------

**基本语法：**

.. code-block:: bash

   habit get-habitat --config <config_file> [--mode <mode>] [--pipeline <pipeline_path>] [--resume] [--debug]

**参数说明：**

- `--config`, `-c`: 配置文件路径（必需，
- `--mode`, `-m`: 运行模式（train 或predict），覆盖配置文件中的设置
- `--pipeline`: Pipeline 文件路径，用于predict 模式，覆盖配置文件中的设置
- `--resume`: 启用断点续训（train 模式）；跳过 ``manifest.json`` 中 ``completed_subjects``；曾失败被试默认不重试（与 YAML ``resume: true`` 等效，CLI 可覆盖 YAML）
- `--debug`: 启用调试模式

**使用示例：**

.. code-block:: bash

   # 训练模式
   habit get-habitat --config ./config_habitat_train.yaml --mode train

   # 预测模式
   habit get-habitat --config ./config/habitat/config_habitat_two_step.yaml --mode predict

   # 使用指定的Pipeline 文件
   habit get-habitat --config ./config/habitat/config_habitat_two_step.yaml --mode predict --pipeline ./custom_pipeline.pkl

   # 启用调试模式
   habit get-habitat --config ./config/habitat/config_habitat_two_step.yaml --debug

   # 断点续训（train；详见本文「断点续训详解」）
   habit get-habitat --config ./config/habitat/config_habitat_two_step.yaml --resume

**输出：**

生境图将保存在配置文件中指定的输出目录。

Python API 使用方法
------------------

**基本用法：**

.. code-block:: python

   from habit.core.habitat_analysis.configurator import HabitatConfigurator
   from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig

   # Load and validate YAML into HabitatAnalysisConfig
   config = HabitatAnalysisConfig.from_file('./config/habitat/config_habitat_two_step.yaml')

   # Wires collaborator services + logging (mirror: habit.cli habitat command)
   configurator = HabitatConfigurator(config=config)

   habitat_analysis = configurator.create_habitat_analysis()

   # Train path; predict: habitat_analysis.predict(pipeline_path=config.pipeline_path, ...)
   habitat_analysis.fit()

**详细示例：**

.. code-block:: python

   import logging
   from pathlib import Path
   from habit.core.habitat_analysis.configurator import HabitatConfigurator
   from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
   from habit.utils.log_utils import setup_logger

   # Logging
   output_dir = Path('./results/habitat')
   output_dir.mkdir(parents=True, exist_ok=True)
   logger = setup_logger(
       name='habitat_analysis',
       output_dir=output_dir,
       log_filename='habitat_analysis.log',
       level=logging.INFO
   )

   # Load and validate YAML into HabitatAnalysisConfig
   config = HabitatAnalysisConfig.from_file('./config/habitat/config_habitat_two_step.yaml')

   configurator = HabitatConfigurator(config=config, logger=logger, output_dir=str(output_dir))

   habitat_analysis = configurator.create_habitat_analysis()

   logger.info("开始生境分析")
   habitat_analysis.fit(save_results_csv=True)
   logger.info("生境分析完成。")

YAML 配置详解
--------------

**配置文件结构：**

.. code-block:: yaml

   # 运行模式和Pipeline 设置（首先检查这些！，
   run_mode: predict  # train（如果要训练新模型，设置个train）或 predict（如果要使用预训练模型，设置个predict，
   pipeline_path: ./results/habitat_pipeline.pkl  # predict 模式的Pipeline 路径（predict 模式必需，

   # 数据路径
   data_dir: ./file_habitat.yaml
   out_dir: ./results/habitat/predict

   # 特征提取设置（仅在train 模式需要）
   FeatureConstruction:
     voxel_level:
       method: concat(raw(delay2), raw(delay3), raw(delay5))
       params: {}

     supervoxel_level:
       supervoxel_file_keyword: '*_supervoxel.nrrd'
       method: mean_voxel_features()
       params:
         params_file: {}

     preprocessing_for_subject_level:
       methods:
         - method: winsorize
           winsor_limits: [0.05, 0.05]
           global_normalize: true
         - method: minmax
           global_normalize: true

     preprocessing_for_group_level:
       methods:
         - method: binning
           n_bins: 10
           bin_strategy: uniform
           global_normalize: false

   # 生境分割设置（仅在train 模式需要）
   HabitatSegmentation:
     # 聚类策略，one_step"。two_step" 或"direct_pooling"
     clustering_mode: two_step

     # 超像素聚类设置（第一步：个体级别聚类，
     supervoxel:
       algorithm: kmeans
       n_clusters: 50  # two_step 模式使用，或 one_step 模式的最大值
       random_state: 42
       max_iter: 300
       n_init: 10

       # one_step 模式设置：每个肿瘤的自动聚类数选择
       one_step_settings:
         min_clusters: 2           # 要测试的最小聚类数
         max_clusters: 10          # 要测试的最大聚类数
         selection_method: inertia  # 确定最佳聚类数的方法：silhouette、calinski_harabasz、davies_bouldin、inertia、kneedle
         plot_validation_curves: true  # 为每个肿瘤绘制验证曲级

     # 生境聚类设置（第二步：群体级别聚类，仅在 two_step 模式使用，
     habitat:
       algorithm: kmeans  # kmeans 或gmm
       max_clusters: 10
       # - 'silhouette' / 'calinski_harabasz': 选择最大分数（越高越好）
       # - 'davies_bouldin': 选择最小分数（越低越好）
       # - 'inertia' / 'kneedle': 都是在 Inertia 曲线上使用 Kneedle 选拐点
       # - 'aic' / 'bic': GMM 常用，选择最小分数（越低越好）
       habitat_cluster_selection_method:
         - inertia
         # - silhouette
         # - calinski_harabasz
         # - aic
         # - bic
         # - davies_bouldin

       fixed_n_clusters:    # 固定的聚类数（设置为 null 则自动选择，
       random_state: 42
       max_iter: 300
       n_init: 10

   # 通用设置
   processes: 2  # 并行进程数
   individual_subject_timeout_sec: 900  # 个体级 Stage 1 单被试墙钟上限（秒）；默认 15 分钟；null 表示不限时
   resume: true  # 断点续训：跳过 manifest 中已完成被试；曾失败被试默认不重试
   # checkpoint_dir: null  # 默认 <out_dir>/.habitat_checkpoint
   # force_rerun_subjects: []  # resume=true 时强制重跑列表中的被试
   clear_checkpoint_on_success: false  # 训练全部成功后是否删除 checkpoint（默认保留）
   plot_curves: true  # 是否生成和保存图行
   save_results_csv: true  # 是否将结果保存为 CSV 文件
   random_state: 42  # 可重复性的随机种子
   debug: false  # 启用详细日志的调试模式

**字段说明：**

**run_mode**: 运行模式

- `train`: 训练新模型
- `predict`: 使用预训练模型进行预测

**pipeline_path**: Pipeline 文件路径

- predict 模式必需
- 指定训练好的 Pipeline 文件路径

**data_dir**: 数据目录路径

- 可以是文件夹或YAML 配置文件
- 参见 :doc:`../data_structure_zh` 了解数据结构

**out_dir**: 输出目录路径

- 生境图和结果将保存在此目录

**processes**: 个体级步骤并行进程数

- 对应 ``HabitatAnalysisConfig.processes``，默认 ``2``，须为正整数。

**individual_subject_timeout_sec**: 个体级并行阶段单被试墙钟时间上限

- **类型**: 数值（秒）或 ``null``
- **默认**: ``900``（15 分钟）；在 YAML 中省略该键时与代码默认值一致。
- **说明**: 从任务提交（多进程）或子进程启动（单进程）起计时，超时则将该被试记为失败并继续其余被试；详见实现与并行行为说明。设为 ``null`` 表示**不启用**单被试超时（一直等到结束或报错）。
- **注意**: 多进程并行时无法可靠终止已卡住的子进程，超时后主流程不再等待，但子进程可能仍在后台运行直至退出。

**resume**: 是否启用断点续训（个体级 Stage 1）

- **类型**: 布尔值
- **默认**: ``true``
- **说明**: 为 ``true`` 时读取 ``checkpoint_dir``（默认 ``<out_dir>/.habitat_checkpoint``）中的 ``manifest.json``，**跳过** ``completed_subjects`` 中的被试（直接从 ``subjects/{id}.pkl`` 加载）；**曾失败被试**（``failed_subjects``）在续训时**不会自动重试**，除非列入 ``force_rerun_subjects`` 或清空 checkpoint。
- **CLI**: ``habit get-habitat --resume`` 等效于 YAML 中 ``resume: true``（CLI 优先覆盖 YAML）。
- **predict 模式**: 忽略 ``resume`` 及相关字段。

**checkpoint_dir**: 断点续训缓存根目录

- **类型**: 字符串或 ``null``
- **默认**: ``null``（使用 ``<out_dir>/.habitat_checkpoint``）
- **说明**: 续训时必须指向**同一目录**；更换 ``out_dir`` 而不改 ``checkpoint_dir`` 会找不到旧 checkpoint。

**force_rerun_subjects**: 强制重跑的被试 ID 列表

- **类型**: 字符串列表
- **默认**: ``[]``
- **说明**: 在 ``resume: true`` 时仍重新处理列表中的被试（从 manifest 的 completed/failed 中移除并重跑）。

**clear_checkpoint_on_success**: 训练成功后是否删除 checkpoint

- **类型**: 布尔值
- **默认**: ``false``
- **说明**: 为 ``true`` 时，Stage 1 + Stage 2 **全部成功**后删除整个 checkpoint 目录。大规模队列建议 ``false``，便于排查或保留中间结果。

**断点续训与配置一致性（config_hash）**

- 程序用 ``data_dir`` + ``FeatureConstruction`` + ``HabitatSegmentation`` 计算 ``config_hash``，写入 ``manifest.json``。
- 续训时若 hash **与 manifest 不一致**，日志警告并**清空 checkpoint**，全部重跑。
- **修改以下字段不影响 hash**，可安全续训：``processes``、``individual_subject_timeout_sec``、``plot_curves``、``save_results_csv``、``verbose``、``debug``、``on_subject_failure`` 等。
- **切换** ``clustering_mode``（``one_step`` / ``two_step`` / ``direct_pooling``）**会改变 hash**，不能共用同一 checkpoint。

断点续训详解
------------

**适用场景**

- ``run_mode: train`` 且被试数量多、个体级 Stage 1 耗时长（radiomics、聚类等）
- 运行中断（手动停止、超时、单被试失败、机器重启）后从中断处继续
- **不适用于** ``predict`` 模式（无 checkpoint）

**覆盖范围**

Pipeline 分两阶段；checkpoint **只覆盖 Stage 1（个体级）**：

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - clustering_mode
     - Stage 1（可 checkpoint）
     - Stage 2（不可 checkpoint，中断后需重跑）
   * - ``two_step``
     - voxel 特征 → 预处理 → supervoxel 聚类 → mean/merge → ``checkpoint_save``
     - combine → group 预处理 → group 聚类
   * - ``one_step``
     - voxel 特征 → 预处理 → habitat 聚类 → mean/merge → ``checkpoint_save``
     - combine（无 group 聚类）
   * - ``direct_pooling``
     - voxel 特征 → 预处理 → ``checkpoint_save``
     - concat 全体 voxel → group 预处理 → group 聚类

**磁盘结构**

.. code-block:: text

   <out_dir>/.habitat_checkpoint/
   ├── manifest.json
   └── subjects/
       ├── sub001.pkl
       ├── sub002.pkl
       └── ...

``manifest.json`` 主要字段：

.. code-block:: json

   {
     "version": 1,
     "config_hash": "4cb5616edf46aece",
     "clustering_mode": "two_step",
     "completed_subjects": ["sub001", "sub002"],
     "failed_subjects": ["sub706"],
     "stage": "individual"
   }

- ``completed_subjects``：父进程确认成功的被试（manifest 与 ``subjects/*.pkl`` 对应）
- ``failed_subjects``：超时、崩溃或 Python 异常的被试；续训时**跳过且不自动重试**
- ``stage``：``"individual"`` 表示 Stage 1 未完成或 Stage 2 未跑完；``"done"`` 表示训练已成功结束

**工作机制（写入分离）**

1. **Worker 子进程**：个体级最后一步 ``checkpoint_save`` 调用 ``save_subject_pkl``，写入 ``subjects/{id}.pkl``（**不**改 manifest）
2. **父进程**：每个被试并行任务结束后，成功则 ``record_success_manifest``（更新 manifest）；失败则 ``record_failure``（记入 ``failed_subjects`` 并删除对应 pkl）

因此：**只有父进程判定成功**的被试才会进入 ``completed_subjects``；子进程算完但超时/崩溃时，可能无 pkl 或 pkl 被 ``record_failure`` 删除。

**pkl 内容（按模式）**

- ``two_step`` / ``one_step``：``merge_supervoxel_features`` 之后的精简 ``HabitatSubjectData``（主要是 ``supervoxel_df``，体积较小）
- ``direct_pooling``：``individual_preprocessing`` 之后（``features`` + ``raw`` + ``mask_info``，体积较大）

**典型工作流**

**首次训练**

.. code-block:: yaml

   run_mode: train
   resume: false
   out_dir: ./results/habitat_two_step

**中断后续训**

.. code-block:: yaml

   resume: true
   # out_dir 与 checkpoint_dir 须与上次一致（或显式指定同一 checkpoint_dir）

.. code-block:: bash

   habit get-habitat -c config/habitat/config_habitat_two_step.yaml --resume

**重试失败被试**（例如 ``sub706``）

.. code-block:: yaml

   resume: true
   force_rerun_subjects:
     - sub706

或删除 ``.habitat_checkpoint`` 后 ``resume: false`` 全量重跑。

**验证配置是否一致**

续训前可对比 ``manifest.json`` 中的 ``config_hash`` 与当前 YAML 计算值是否相同。程序启动时会自动校验；一致时日志类似：

.. code-block:: text

   Loaded checkpoint: 120 completed, 3 failed subject(s).

不一致时：

.. code-block:: text

   Checkpoint config hash changed (abc -> def); discarding checkpoint and restarting all subjects.

手动校验（Python）：

.. code-block:: python

   import json
   from pathlib import Path
   import yaml
   from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
   from habit.core.habitat_analysis.checkpoint.manager import compute_config_hash

   cfg = HabitatAnalysisConfig(**yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")))
   manifest = json.loads(Path("out/.habitat_checkpoint/manifest.json").read_text(encoding="utf-8"))
   print(compute_config_hash(cfg) == manifest["config_hash"])

**常见日志与含义**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 日志
     - 含义
   * - ``Loaded checkpoint: N completed, M failed``
     - config_hash 一致，续训生效
   * - ``Resume: loaded K completed subject(s) from checkpoint``
     - 从 pkl 加载 K 个已完成被试，不派 worker
   * - ``Resume: skipping M previously failed subject(s)``
     - M 个失败被试被跳过（需 ``force_rerun_subjects`` 才能重跑）
   * - ``Processing X/Y pending subjects``
     - 共 Y 个被试，本次还需跑 X 个
   * - ``Timeout (>Ns) for item subXXX``
     - 单被试超时 → ``failed_subjects``
   * - ``Child exited without a queue result (exit code ...)``
     - 子进程崩溃（Windows 上 ``3221225477`` 常为 native 访问冲突）→ ``failed_subjects``
   * - ``Training succeeded; clearing checkpoint at ...``
     - ``clear_checkpoint_on_success: true`` 且训练成功，checkpoint 已删除

**与最终产物的关系**

- Checkpoint 是**训练中间态**；最终模型为 ``<out_dir>/habitat_pipeline.pkl``。
- Stage 2（group 聚类等）**没有** checkpoint：若在 Stage 2 失败，Stage 1 的 checkpoint 仍保留，修复问题后 ``resume: true`` 可跳过已完成被试并**重新执行 Stage 2**。

**FeatureConstruction**: 特征提取设置

- `voxel_level`: 体素级特征提取
- `supervoxel_level`: 超像素级特征提取
- `preprocessing_for_subject_level`: 个体级别预处理
- `preprocessing_for_group_level`: 群体级别预处理

**HabitatSegmentation**: 生境分割设置

- `clustering_mode`: 聚类策略（one_step、two_step、direct_pooling）
- `supervoxel`: 超像素聚类设置
- `habitat`: 生境聚类设置

聚类策略详解
----------------

**One-Step 策略**

**描述：**

对每个肿瘤独立进行体素到生境的聚类。

**适用场景：**

- 样本量较小
- 希望保留每个肿瘤的独特特征
- 不需要跨患者的泛化

**优点：**

- 简单直接
- 每个肿瘤独立分析
- 计算复杂度较低

**缺点：**

- 可能难以发现跨患者的共同模式
- 泛化能力较弱

**配置示例：**

.. code-block:: yaml

   HabitatSegmentation:
     clustering_mode: one_step
     supervoxel:
       algorithm: kmeans
       n_clusters: 10  # 最大聚类数
       random_state: 42
       one_step_settings:
         min_clusters: 2
         max_clusters: 10
         selection_method: silhouette
         plot_validation_curves: true

**Two-Step 策略（推荐）**

**描述：**

两步聚类，

- 第一步：体素到超像素（个体级别）
- 第二步：超像素到生境（群体级别）

**适用场景：**

- 样本量较大
- 希望发现跨患者的共同生境模式
- 需要更好的泛化能力

**优点：**

- 可以发现跨患者的共同生境模式
- 提高泛化能力
- 减少计算复杂度

**缺点：**

- 计算复杂度较高
- 需要更多的参数调优

**配置示例：**

.. code-block:: yaml

   HabitatSegmentation:
     clustering_mode: two_step
     supervoxel:
       algorithm: kmeans
       n_clusters: 50
       random_state: 42
     habitat:
       algorithm: kmeans
       max_clusters: 10
       habitat_cluster_selection_method:
         - inertia
         - silhouette
       fixed_n_clusters: null  # 自动选择
       random_state: 42

**3. Direct Pooling 策略**

**描述：**

将所有受试者的所有体素池化后一次性聚类。

**适用场景：**

- 数据量较小
- 希望快速获得结果
- 不需要个体差异

**优点：**

- 计算速度较快
- 实现简单
- 能够捕捉群体中最显著的组织模式

**缺点：**

- 可能忽略个体差异
- 对内存要求较高（因为一次性处理所有体素）

**关于数据泄露的说明：**
虽然该策略使用了所有受试者的数据进行聚类，但由于生境发现本质上是一个*无监督学义*过程（不涉及预测标签），在特征空间进行的 Pooling 操作通常不被视为传统意义上的“标签泄露”。它更像是一种群体水平的特征表示学习。

**配置示例：**

.. code-block:: yaml

   HabitatSegmentation:
     clustering_mode: direct_pooling
     habitat:
       algorithm: kmeans
       max_clusters: 10
       habitat_cluster_selection_method:
         - inertia
       fixed_n_clusters: null
       random_state: 42

支持的聚类算法
----------------

下表列出 ``habit/core/habitat_analysis/clustering/`` 内置、且与 YAML Schema 一致的聚类算法。
配置时把对应的 key 填到 ``HabitatSegmentation.supervoxel.algorithm`` 或
``HabitatSegmentation.habitat.algorithm`` 即可。

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - 算法
     - 配置 key
     - 说明
   * - K-Means
     - ``kmeans``
     - 默认；速度快，适合球形簇结构。可用于 supervoxel 与 habitat。
   * - Gaussian Mixture
     - ``gmm``
     - 概率模型；支持 AIC/BIC 选数。可用于 supervoxel 与 habitat。
   * - SLIC
     - ``slic``
     - 体素 → supervoxel 的空间感知超像素分割（仅用于 ``supervoxel.algorithm``）。

要新增自定义聚类算法，参见
``habit/core/habitat_analysis/clustering/custom_clustering_template.py``。

最优聚类数选择方法
--------------------

当 ``fixed_n_clusters`` 未设置时，HABIT 在 ``min_clusters`` 至
``max_clusters`` 区间内枚举 ``k``，每个 ``k`` 计算评分并按下表逻辑选最优 k。

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - 方法
     - 选择逻辑
     - 说明
   * - ``silhouette``
     - 取最大值
     - 轮廓系数（-1 ~ 1，越高越好）。
   * - ``calinski_harabasz``
     - 取最大值
     - 方差比指标（越高越好）。
   * - ``davies_bouldin``
     - 取最小
     - 簇间分离度（越低越好）。
   * - ``inertia``
     - Kneedle 拐点
     - 簇内平方误差曲线，选拐点k。
   * - ``kneedle``
     - Kneedle 拐点
     - 显式使用 Kneedle；当前实现与 ``inertia`` 共用同一 Inertia 曲线。
   * - ``bic`` / ``aic``
     - 取最小
     - 信息准则，主要用于GMM。
   * - ``gap``
     - 取最大值
     - Gap statistic（与随机参考分布的差距，越大越好）。

**多方法投票**：当 ``habitat_cluster_selection_method`` 写成 YAML 列表时，
每个方法独立选出最优 k，按得票投票，最终选得票最多的 k；票数相同时取较小的
k（更保守）。例如 ``[silhouette, calinski_harabasz, davies_bouldin]`` 中
若两种选 5、一种选 4，则结果是 5。

Pipeline 机制
------------

连通域后处理（减少碎块）
----------------------

为了减少生境图中“破碎小块”，HABIT 支持在ROI 内执行连通域后处理。当前实现采用**SimpleITK 快路径**，

1. 按标签识别并临时移除小连通域（体素数小于 ``min_component_size``）。
2. 使用最近的大连通域种子标签对被移除体素进行回填。

该流程的关键目标是：**减少碎块，同时保证 ROI 内体素不丢失**。

可配置入口：

- ``HabitatSegmentation.postprocess_supervoxel``：超体素层后处理（主要用于two_step 的voxel->supervoxel 结果，
- ``HabitatSegmentation.postprocess_habitat``：生境层后处理（适用于one_step/two_step/direct_pooling 的最终生境图，

参数说明：

- ``enabled``: 是否启用后处理
- ``min_component_size``: 连通域最小体素数阈值，小于阈值的组件会被临时移除并回填
- ``connectivity``: 连通性设置。当前实现中 ``1`` 为面邻接优先；``2``/``3`` 在快路径中均表现为全连接行为
- ``debug_postprocess``: 是否输出后处理详细日志（按标签等阶段）
- ``reassign_method``: 兼容字段，当前快路径中已忽略
- ``max_iterations``: 兼容字段，当前快路径中已忽略

配置示例：

.. code-block:: yaml

   HabitatSegmentation:
     postprocess_supervoxel:
       enabled: false
       min_component_size: 30
       connectivity: 1
       debug_postprocess: false
       reassign_method: neighbor_vote  # deprecated/ignored
       max_iterations: 3               # deprecated/ignored

     postprocess_habitat:
       enabled: true
       min_component_size: 30
       connectivity: 1
       debug_postprocess: false
       reassign_method: neighbor_vote  # deprecated/ignored
       max_iterations: 3               # deprecated/ignored

说明，

- 后处理仅作用于 ROI 内标签，ROI 外保持 0。
- 当前快路径会保证 ROI 内体素保持有标签（不会永久删除ROI 体素）。
- 建议先从 ``min_component_size=30``、``connectivity=1`` 开始调参。

HABIT 继承于scikit-learn 的Pipeline 机制，这是避免数据泄露的关键设计。

**什么是数据泄露：**

数据泄露是指在模型训练过程中，测试集的信息意外地泄露到训练集中，导致模型性能被高估。

**Pipeline 如何避免数据泄露：**

1. **训练阶段**: 在训练集上训练 Pipeline，包括特征提取、聚类等步骤
2. **预测阶段**: 加载训练好的 Pipeline，应用于测试集，确保使用相同的处理流程
3. **严格分离**: 训练集和测试集完全分离，避免测试集信息泄露

**生境分析中的 Pipeline:**

.. code-block:: python

   from habit.core.habitat_analysis.configurator import HabitatConfigurator
   from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig

   # Load and validate YAML into HabitatAnalysisConfig
   config = HabitatAnalysisConfig.from_file('./config/habitat/config_habitat_two_step.yaml')

   # Wires collaborator services + logging (mirror: habit.cli habitat command)
   configurator = HabitatConfigurator(config=config)

   habitat_analysis = configurator.create_habitat_analysis()

   # Train path; predict: habitat_analysis.predict(pipeline_path=config.pipeline_path, ...)
   habitat_analysis.fit()

**关键要点：**

- 训练和预测必须使用相同的 Pipeline
- 不要在整个数据集上进行聚类
- 使用交叉验证时，每个 fold 的训练和预测必须严格分离

底层 sklearn 风格 API
~~~~~~~~~~~~~~~~~~~~~~~

日常推荐 CLI（``habit get-habitat``）或 Python 中的 ``HabitatAnalysis.fit()`` /
``predict()``（``run()`` 仍可按配置分发，但已标注为遗留入口）。若要直接操作
序列化后的``HabitatPipeline``，可使用与 sklearn 一致的接口，

.. code-block:: python

   from habit.core.habitat_analysis import HabitatPipeline

   # 训练：HabitatAnalysis.fit() 内部会构建并调用 fit_transform()
   #       并把已 fit 的 pipeline 保存为habitat_pipeline.pkl

   # 推理：直接加载训练产物并 transform
   pipeline = HabitatPipeline.load('./results/habitat/train/habitat_pipeline.pkl')
   results_df = pipeline.transform(X_test)   # X_test: Dict[subject_id, {}]

   # 也可以保存：
   pipeline.save('./results/habitat/train/habitat_pipeline.pkl')

要点：

- ``HabitatPipeline.fit(X)``：在训练数据上学习参数，包括 group 级预处理统计量
  以及 habitat 聚类模型；调用后 ``fitted_=True``。

- ``HabitatPipeline.transform(X)``：仅在``fitted_=True`` 时可用；用已学到的
  参数处理新数据。

- ``HabitatPipeline.save(path) / .load(path)``：joblib 序列化整个 pipeline，
  反序列化时所有 step 的状态（``PreprocessingState``、聚类模型、最优 k 等）
  都会恢复。

- 在``HabitatAnalysis.predict()`` 加载的pipeline 上，运行时的 service
  会通过 ``_PIPELINE_SERVICE_ATTRS`` 白名单重新注入；用户层无需手动操作。

特征提取详解
----------------

**体素级特征提取**

**描述：**

对每个体素提取特征。

**适用场景：**

- 需要精细的空间分析
- 研究体素级别的异质性

**配置示例：**

.. code-block:: yaml

   FeatureConstruction:
     voxel_level:
       method: concat(raw(delay2), raw(delay3), raw(delay5))
       params: {}

**内置特征提取器：**

- `raw`: 原始图像强度
- `kinetic`: 动力学特征
- `local_entropy`: 局部熵
- `mean_voxel_features`: 平均体素特征
- `supervoxel_radiomics`: 超像素影像组学特征

**超像素级特征提取**

**描述：**

对超像素区域提取特征。

**适用场景：**

- 需要区域级别的分析
- 减少计算复杂度

**配置示例：**

.. code-block:: yaml

   FeatureConstruction:
     supervoxel_level:
       supervoxel_file_keyword: '*_supervoxel.nrrd'
       method: mean_voxel_features()
       params:
         params_file: {}

**预处理方法**

**个体级别预处理：**

.. code-block:: yaml

   preprocessing_for_subject_level:
     methods:
       - method: winsorize
         winsor_limits: [0.05, 0.05]
         global_normalize: true
       - method: minmax
         global_normalize: true
       - method: log
         global_normalize: true

**群体级别预处理：**

仅 ``two_step`` 与 ``direct_pooling`` 模式会使用此配置；``one_step`` 模式的 pipeline 不含群体级预处理步骤。

.. code-block:: yaml

   preprocessing_for_group_level:
     methods:
       - method: binning
         n_bins: 10
         bin_strategy: uniform
         global_normalize: false

无监督特征筛选放置建议
----------------------

在聚类任务中，``variance_filter`` 和 ``correlation_filter`` 这类“删列型”方法会改变特征维度。
为避免跨受试者特征列不一致，建议按聚类策略放置：

- **two_step**:

  - 个体级别（``preprocessing_for_subject_level``）不要使用删列型方法（配置校验会拒绝）
  - 推荐在群体级别（``preprocessing_for_group_level``）执行删列型方法

- **one_step**:

  - 仅在个体级别（``preprocessing_for_subject_level``）使用删列型方法即可
  - 本模式无群体级预处理步骤；每个受试者独立聚类，不存在跨受试者列对齐问题
  - 若需要跨受试者统一特征空间或群体级建模，请改用 ``two_step`` 或 ``direct_pooling``

- **direct_pooling**:

  - 个体级别不要使用删列型方法（拼接后会导致列不一致）
  - 推荐在群体级别（``preprocessing_for_group_level``）执行删列型方法，以保证列空间一致

推荐的无监督筛选方法：

- ``variance_filter``: 删除低方差特征
- ``correlation_filter``: 删除高相关冗余特征

**扩展自定义特征预处理方法**

生境特征预处理（非图像 ``habit preprocess``）使用 ``PreprocessingMethodFactory``
与 ``@register_preprocessing``。新增步骤：

1. 继承 ``BaseFeaturePreprocessing``，实现 ``fit`` / ``transform``（DataFrame 进/出）
2. 用 ``@register_preprocessing("your_method")`` 注册；删列型方法设 ``changes_columns=True``
3. 在 ``PreprocessingMethod.method`` Literal 中追加方法名
4. 参考 ``habit/core/habitat_analysis/feature_preprocessing/custom_preprocessing_template.py``

个体级与群体级共用同一 handler；群体级由 ``PreprocessingState`` 在训练时缓存参数。

自定义特征提取器
----------------

HABIT 支持自定义特征提取器，您可以添加自己的特征提取方法。

**步骤 1: 创建自定义特征提取器**

.. code-block:: python

   from habit.core.habitat_analysis.clustering_features.base_extractor import BaseClusteringExtractor
   from habit.core.habitat_analysis.clustering_features.base_extractor import register_feature_extractor

   @register_feature_extractor('my_feature_extractor')
   class MyFeatureExtractor(BaseClusteringExtractor):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           self.feature_names = ['feature1', 'feature2', 'feature3']

       def extract_features(self, image_data, **kwargs):
           # Implement feature extraction logic.
           n_samples = image_data.shape[0]
           features = np.random.random((n_samples, 3))
           return features

**步骤 2: 在配置文件中使用**

.. code-block:: yaml

   FeatureConstruction:
     voxel_level:
       # 注意：多图像输入时，每个图像名称必须包裹在raw() 个
       method: my_feature_extractor(raw(delay2), raw(delay3))
       params:
         param1: value1

**步骤 3: 运行生境分析**

.. code-block:: bash

   habit get-habitat --config config_with_custom_extractor.yaml

实际示例
--------

**示例 1: 训练模式**

.. code-block:: yaml

   run_mode: train
   data_dir: ./file_habitat.yaml
   out_dir: ./results/habitat/train

   FeatureConstruction:
     voxel_level:
       method: concat(raw(delay2), raw(delay3), raw(delay5))
       params: {}

     supervoxel_level:
       supervoxel_file_keyword: '*_supervoxel.nrrd'
       method: mean_voxel_features()
       params:
         params_file: {}

     preprocessing_for_group_level:
       methods:
         - method: binning
           n_bins: 10
           bin_strategy: uniform

   HabitatSegmentation:
     clustering_mode: two_step
     supervoxel:
       algorithm: kmeans
       n_clusters: 50
       random_state: 42
     habitat:
       algorithm: kmeans
       max_clusters: 10
       habitat_cluster_selection_method:
         - inertia
         - silhouette
       random_state: 42

   processes: 2
   individual_subject_timeout_sec: 900
   resume: true
   # checkpoint_dir: null
   # force_rerun_subjects: []
   clear_checkpoint_on_success: false
   plot_curves: true
   save_results_csv: true
   random_state: 42

**示例 2: 预测模式**

.. code-block:: yaml

   run_mode: predict
   pipeline_path: ./results/habitat/train/habitat_pipeline.pkl
   data_dir: ./file_habitat.yaml
   out_dir: ./results/habitat/predict

   processes: 2
   individual_subject_timeout_sec: 900
   random_state: 42

输出结构
--------

生境分析的输出结构（具体文件取决于策略与是否开启可视化）：

.. code-block:: text

   results/habitat/
   ├── train/                      # 训练模式输出
   ─  ├── habitat_pipeline.pkl    # 训练好的 Pipeline
   ─  ├── habitats.csv            # 生境结果表（如启用保存）
   ─  ├── subj001_habitats.nrrd   # 生境图（One-Step/Direct Pooling，
   ─  ├── subj001_supervoxel.nrrd # 超像素图（Two-Step，
   ─  ├── visualizations/         # 可视化图行
   ─  ─  ├── optimal_clusters/   # 自动选簇场景的聚类验证曲线（One-Step/Two-Step，
   ─  ─  ├── one_step_clustering/
   ─  ─  ├── supervoxel_clustering/
   ─  ─  └── habitat_clustering/
   ─  └── habitat_analysis.log    # 日志文件
   └── predict/                    # 预测模式输出
       ├── habitats.csv
       ├── subj001_habitats.nrrd
       ├── subj001_supervoxel.nrrd
       ├── visualizations/
       ─  └── ...
       └── habitat_analysis.log

常见问题
--------

**Q1: 如何选择聚类策略：**

A: 根据您的研究需求选择，

- **研究型项目**: 推荐使用 Two-Step 策略，平衡了群体一致性和个体差异。
- **快速原型**: 可以使用 Direct Pooling 策略，适合初步探索群体模式。
- **小样本研究**: 可以考虑 One-Step 策略，关注每个病例的独特性。

**Q2: 如何确定最佳聚类数：**

A: HABIT 提供了多种自动化评估方法，

- **inertia**: 计算不同 k 下的 Inertia 曲线，并使用 Kneedle 在曲线上选拐点。
- **kneedle**: 显式使用 Kneedle 拐点检测；当前实现与 ``inertia`` 共用同一 Inertia 曲线。
- **silhouette**: 使用轮廓系数。
- **calinski_harabasz**: 使用 Calinski-Harabasz 指数。
- **davies_bouldin**: 使用 Davies-Bouldin 指数。

多方法自动选择时，请使用YAML 列表，例如``[inertia, silhouette, calinski_harabasz]``。系统会让每个方法各自投票，得票最多的 k 被选中；如果平票，选择较小的k。

**Q3: 训练和预测模式有什么区别？**

A: 

- **训练模式**: 在训练集上训练新的Pipeline（包括特征预处理、聚类模型等），并保存为 `.pkl` 文件。
- **预测模式**: 加载已保存的 Pipeline，将其直接应用于新数据，确保新数据的处理逻辑与训练集完全一致。

**Q4: 如何避免数据泄露：**

A: 遵循以下原则，

1. 始终使用 Pipeline 机制进行预测。
2. 在机器学习建模前，确保生境发现过程不包含任何标签信息。
3. 如果使用 Direct Pooling，请明确其无监督属性。

**Q5: 生境分割失败怎么办？**

A: 检查以下几点：

1. 配置文件中的 `data_dir` 是否指向了正确的 YAML 或文件夹。
2. 图像和ROI Mask 的空间几何信息（Origin, Spacing, Direction）是否完全一致。
3. 查看输出目录下的 `habitat_analysis.log` 文件，寻找具体的 Python 报错堆栈。

下一步
-------

生境分割完成后，您可以：

- :doc:`habitat_feature_extraction_zh`: 进行生境特征提取
- :doc:`machine_learning_modeling_zh`: 进行机器学习建模
- :doc:`../customization/index_zh`: 了解如何自定义特征提取器和聚类算法
