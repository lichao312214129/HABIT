配置参考
========

本节详细说明 HABIT 的所有配置文件参数和选项。

.. seealso::

   第三方库侧的算法、度量与全量 API 见 :doc:`reference/upstream_libraries_zh`。

概述
----

HABIT 使用 YAML 格式的配置文件来控制所有功能。每个功能模块都有对应的配置文件，用户可以通过修改配置文件来调整功能。

**配置文件类型：**

- **预处理配置**: 控制图像预处理流程（``PreprocessingConfig``）
- **DICOM 整理配置**: 仅 dcm2niix 重命名/整理（``DicomSortConfig`` ，``habit sort-dicom``）
- **生境分析配置**: 控制生境分割和特征提取
- **特征提取配置**: 控制生境特征提取
- **机器学习配置**: 控制机器学习建模
- **数据配置**: 指定数据路径和结构

**配置文件特点：**

- **易于理解**: 使用 YAML 格式，易于阅读和编辑
- **灵活配置**: 支持多种参数组合
- **版本控制**: 可以纳入版本控制，便于追踪变更
- **可重复性**: 相同的配置文件产生相同的结果

参数说明约定
~~~~~~~~~~~~

除非写明 **无（必填）**，每条参数均列出 **默认值**（来自 ``habit/core/*/config_schemas.py`` 中的 Pydantic 定义）。

- YAML 中 **省略该键** → 加载后使用 Schema **默认值**（与仓库模板 YAML 里的示例值可能不同）。
- 默认值为 ``null`` → 表示「未设置」；部分字段会再 **继承** 顶层（如 ``random_state``）或按运行模式推导（如 ``checkpoint_dir``）。
- 模板文件（``config/**/*.yaml``）中的数值仅为演示；以本页 **默认值** 列为准。

.. note::

   下列「通用配置参数」中的默认值适用于多处文档叙述；**图像预处理**（``PreprocessingConfig``）的专用默认值与额外字段以本节 **「预处理配置参数」** 为准（例如 ``processes`` 默认为 1、顶层含 ``auto_select_first_file`` 等）。

快速导航（按功能）
------------------

.. contents::
   :local:
   :depth: 2

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 主题
     - 本节标题（本页内搜索）
   * - 命令与 Schema 对照、路径规则
     - 「CLI 命令与配置 Schema 对照」「路径解析规则」
   * - 多模块共用字段
     - 「通用配置参数」「随机种子传递规则」
   * - ``habit preprocess``
     - 「预处理配置参数」
   * - ``habit sort-dicom``
     - 「DICOM 整理配置参数」
   * - ``habit get-habitat``
     - 「生境分析配置参数」（含 Stage-1 并行总表、``config_hash``）
   * - ``habit extract``
     - 「特征提取配置参数」
   * - ``habit model`` / ``habit cv``
     - 「机器学习配置参数」
   * - ``habit compare``
     - 「模型对比配置」
   * - 被试清单 YAML
     - 「数据配置参数」
   * - ``habit icc`` / ``habit retest`` / ``habit radiomics``
     - 文末 ICC / Test-Retest / 传统组学小节
   * - 仓库模板路径
     - 「仓库配置模板索引」

操作流程（断点续训、配准后端选型等）见各 :doc:`user_guide/index_zh` 用户指南；**字段级说明以本页为准**。

CLI 命令与配置 Schema 对照
----------------------------

下表汇总 **CLI 命令 → 配置 Schema → 仓库内示例 YAML**。无 Schema 的命令仅接受命令行参数。

.. list-table::
   :header-rows: 1
   :widths: 16 28 36

   * - CLI 命令
     - Pydantic Schema（``habit/`` 源码）
     - 示例配置文件
   * - ``habit preprocess``
     - ``PreprocessingConfig``
     - ``config/preprocessing/config_preprocessing_demo_elastix.yaml``
   * - ``habit sort-dicom``
     - ``DicomSortConfig``
     - ``config/dicom_sort/config_sort_dicom.yaml``
   * - ``habit get-habitat``
     - ``HabitatAnalysisConfig``
     - ``config/habitat/config_habitat_two_step.yaml`` （另有 one_step / direct_pooling 及 ``*_predict.yaml``）
   * - ``habit extract``
     - ``FeatureExtractionConfig``
     - ``config/feature_extraction/config_extract_features.yaml``
   * - ``habit model`` / ``habit cv``
     - ``MLConfig``
     - ``config/machine_learning/config_machine_learning.yaml`` （``cv`` 常用 ``config_machine_learning_kfold.yaml``）
   * - ``habit compare``
     - ``ModelComparisonConfig``
     - ``config/model_comparison/config_model_comparison.yaml``
   * - ``habit icc``
     - ``ICCConfig``
     - ``config/auxiliary/config_icc_demo.yaml``
   * - ``habit retest``
     - ``TestRetestConfig``
     - ``config/auxiliary/config_test_retest.yaml``
   * - ``habit radiomics``
     - ``RadiomicsConfig``
     - ``config/radiomics/config_traditional_radiomics.yaml``
   * - ``habit merge-csv`` / ``habit dicom-info`` / ``habit dice``
     - （无 YAML Schema）
     - 见 :doc:`cli_zh`

**被试/文件列表 YAML**（非独立 Schema，由 ``data_dir`` 引用）：

- 预处理：``config/preprocessing/files_preprocessing.yaml`` 、``config/preprocessing/image_files.yaml``
- 生境：``config/habitat/file_habitat.yaml``

路径解析规则
------------

除特别说明外，所有通过 ``BaseConfig.from_file()`` / ``load_config()`` 加载的 YAML，其 **相对路径均相对于该 YAML 文件所在目录** 解析为绝对路径（``habit/core/common/configs/loader.py`` 中的 ``PathResolver``）。

**解析策略（摘要）**

- 字段名以 ``_path`` 、``_dir`` 、``_file`` 、``_folder`` 等结尾，或值以 ``./`` 、``../`` 、``.\`` 、``..\`` 开头，或值带常见扩展名（``.yaml`` 、``.csv`` 、``.nrrd`` 、``.nii.gz`` 等）时，会尝试解析。
- 解析后若目标存在则使用绝对路径；若不存在但形如相对路径，仍会转为 ``base_dir / value`` 的绝对形式。
- **不解析**：URL（``http://`` 等）、``sort-dicom`` 的 ``f`` / ``filename_format`` （dcm2niix ``-f`` 原样传递）、``extra_args`` 列表项。

**CLI 覆盖 YAML（优先级高于配置文件）**

- ``habit get-habitat --mode train|predict`` → 覆盖 ``run_mode``
- ``habit get-habitat --pipeline <path>`` → 覆盖 ``pipeline_path`` （predict 必需）
- ``habit get-habitat --resume`` → 等效 ``resume: true``
- ``habit model --mode train|predict`` → 覆盖 ``MLConfig.run_mode``

通用配置参数
------------

**data_dir**: 数据目录路径

- **类型**: 字符串
- **必需**: 是
- **默认值**: 无（必填）
- **说明**: 可以是文件夹或 YAML 配置文件
- **示例**: `./config/preprocessing/files_preprocessing.yaml`

**out_dir**: 输出目录路径

- **类型**: 字符串
- **必需**: 是
- **默认值**: 无（必填）
- **说明**: 输出文件将保存在此目录
- **示例**: `./preprocessed`

**processes**: 并行进程数

- **类型**: 整数
- **必需**: 否
- **默认值**: 因模块而异（预处理 ``PreprocessingConfig`` 默认为 **1**）
- **说明**: 用于并行处理的进程数；图像预处理中实际 worker 数为 ``min(配置值, CPU核心数-2)`` ，且至少为 1
- **示例**: `4`

**random_state**: 随机种子

- **类型**: 整数
- **必需**: 否
- **默认值**: ``42``
- **说明**: 配置文件**顶层**全局随机种子。各子模块未显式设置 ``random_state`` 时**继承**此值；子模块 YAML 中**显式写入**的值**覆盖**顶层。详见下文「随机种子传递规则」。
- **示例**: `42`

随机种子传递规则
~~~~~~~~~~~~~~~~

**优先级**（高 → 低）：子模块 YAML 显式 ``random_state`` → 配置文件顶层 ``random_state`` → 代码默认值 ``42``。

**生境分析**（``HabitatAnalysisConfig``）：

- 顶层 ``random_state`` ：全局默认；训练入口 ``numpy.random.seed``；未被子模块覆盖的可视化 fallback。
- ``HabitatSegmentation.supervoxel.random_state`` ：**two_step** 个体超像素聚类；**one_step** 时在 ``habitat.random_state`` 未设置时作为 fallback。
- ``HabitatSegmentation.habitat.random_state`` ：**direct_pooling / two_step** 群体生境聚类；**one_step** 个体体素→生境聚类（优先于 ``supervoxel``）。
- 个体聚类与对应散点/t-SNE 图使用同一 effective seed；群体聚类图使用 ``habitat`` effective seed。
- **predict 模式**：聚类模型以 pkl 内已训练参数为准，不因 YAML 顶层种子重建模型。

**机器学习**（``MLConfig``）：

- 顶层 ``random_state`` ：train/test 划分、K-fold、重采样 fallback、模型与特征选择（未在 ``params`` 中写种子时自动注入）。
- ``resampling.random_state`` ：省略时继承顶层。
- ``models.<name>.params.random_state`` ：显式写入则覆盖顶层（不会被自动改写）。
- KNN / NaiveBayes 等无 ``random_state`` 参数的模型不受影响。

**图像预处理**（``PreprocessingConfig``）：

- 顶层 ``random_state`` 在 ``BatchProcessor.run()`` 入口调用 ``numpy.random.seed`` ，供流水线内可能的 NumPy 随机操作使用。

**debug**: 调试模式

- **类型**: 布尔值
- **必需**: 否
- **默认值**: ``false`` （多数带 ``debug`` 字段的 Schema 一致）
- **说明**: 启用详细日志的调试模式
- **示例**: ``true``

预处理配置参数
------------

对应模式类：``habit.core.preprocessing.config_schemas.PreprocessingConfig``。顶层字段 ``Preprocessing`` 的 **键名** 必须与 ``PreprocessorFactory`` 注册名一致；YAML 中书写的子块顺序即执行顺序。

DICOM **仅整理**使用独立配置 ``habit.core.dicom_sort.DicomSortConfig`` 与 CLI ``habit sort-dicom`` ，**不在本小节**：字段说明与路径解析规则见下文 **「DICOM 整理配置参数（sort-dicom）」**，用户指南详见 :doc:`user_guide/image_preprocessing_zh`。

**配置文件示例：**

.. code-block:: yaml

   data_dir: ./config/preprocessing/files_preprocessing.yaml
   out_dir: ./preprocessed
   auto_select_first_file: true

   Preprocessing:
     dcm2nii:
       images: [delay2, delay3, delay5]
       dcm2niix_path: ./dcm2niix.exe
       compress: true
       anonymize: false

     n4_correction:
       images: [delay2, delay3, delay5]
       num_fitting_levels: 4

     resample:
       images: [delay2, delay3, delay5]
       target_spacing: [1.0, 1.0, 1.0]
       img_mode: bilinear

     registration:
       images: [delay2, delay3, delay5]
       fixed_image: delay2
       type_of_transform: SyNRA
       metric: MI
       use_mask: false

     histogram_standardization:
       images: [delay2, delay3, delay5]
       target_min: 0.0
       target_max: 100.0

     zscore_normalization:
       images: [delay2, delay3, delay5]
       only_inmask: false
       clip_values: [-3, 3]

     adaptive_histogram_equalization:
       images: [delay2, delay3, delay5]
       alpha: 0.3
       beta: 0.3
       radius: 5

   save_options:
     save_intermediate: true
     intermediate_steps: [dcm2nii, n4_correction, resample]

   processes: 2
   random_state: 42

**顶层（``PreprocessingConfig``）**

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - 字段
     - 默认值
     - 说明
   * - ``data_dir`` / ``out_dir``
     - 无（必填）
     - 相对路径相对于本 YAML 所在目录解析
   * - ``Preprocessing``
     - ``{}``
     - 步骤名字典；键名须为已注册预处理器名
   * - ``processes``
     - ``1``
     - 须 ``>= 1``；实际并行 ``min(配置, CPU核心数-2)`` ，至少 1
   * - ``random_state``
     - ``42``
     - ``BatchProcessor.run()`` 入口 ``numpy.random.seed``
   * - ``auto_select_first_file``
     - ``true``
     - 目录内多文件时是否自动选第一个
   * - ``preprocessing_input_layout``
     - ``habit_default``
     - 当前仅支持 ``habit_default`` 目录布局
   * - ``save_options``
     - 见下表
     - 中间结果落盘选项

**Preprocessing**：各步骤公共字段 ``images`` （必填，非空列表）。

**dcm2nii**：DICOM 转换

- ``images`` ：模态键列表（**必填**）。
- ``dcm2niix_path`` ：可执行文件或目录；**可省略**，省略时在 ``PATH`` 中查找 ``dcm2niix``。
- 其它常用项：``compress`` 、``anonymize`` 、``filename_format`` 、``adjacent_dicoms`` 、``ignore_derived`` 、``crop_images`` 、``generate_json`` 、``verbose`` 、``batch_mode`` 、``merge_slices`` 、``single_file_mode`` 等（参见源码）。

**n4_correction**

- ``images`` （必填）；``num_fitting_levels`` （默认 4）；``num_iterations``；``convergence_threshold``；``shrink_factor``；可选 ``mask_keys``。

**resample**

- ``images`` （必填）；``target_spacing`` [x,y,z] mm；``img_mode`` （图像插值，默认 ``bilinear``）；``padding_mode``；``align_corners``。掩码重采样使用最近邻。

**registration**

- ``images`` （必填，须包含 ``fixed_image`` ）； ``fixed_image`` （必填）；浮动序列为 ``images`` 去掉 ``fixed_image`` 后的全部键，**不要使用** YAML 字段 ``moving_images`` （实现不读取，且可能作为多余关键字传入 ANTs）。
- ``backend`` （可选）： ``ants`` （默认）、 ``simpleitk`` 、 ``elastix`` （调用官方 elastix / transformix 可执行文件；可选 ``elastix_path`` / ``transformix_path`` ）；详见 :doc:`user_guide/image_preprocessing_zh` 「registration」。
- ``type_of_transform`` 、 ``metric`` 、 ``optimizer`` 等对 **ants / simpleitk** 有意义； ``elastix`` 后端不使用这些键驱动配准。**全部可选值** （ANTS 路径）见同文档列表；常见如 ``Rigid`` 、 ``Affine`` 、 ``SyN`` 、 ``SyNRA`` 等。
- ``use_mask`` ；可选 ``mask_keys`` ； ``replace_by_fixed_image_mask``。
- **elastix 专属** （``elastix`` 后端）：

  - ``elastix_parameter_files`` ：参数模板 ``.txt`` ，宜从 `LKEB elastix Model Zoo <https://lkeb.ml/modelzoo/>`_ 等按数据类型与配准任务选取（说明见 :doc:`user_guide/image_preprocessing_zh` elastix「参数与数据类型」）。
  - ``elastix_parameter_overrides`` ：dict，覆盖参数值。
  - ``elastix_path`` 、``transformix_path`` 、``elastix_threads``。
- 余下 ANTs 允许的参数可通过额外键传入（谨慎使用）。

**histogram_standardization**

- ``images`` （必填）；``percentiles``；``target_min`` / ``target_max``；可选 ``mask_key`` （统计直方图用）。

**zscore_normalization**

- ``images`` （必填）；``only_inmask``；``mask_key`` （当 ``only_inmask`` 为真时必须存在于 ``data`` 中，如共享掩膜键或 ``mask_<modality>``）；``clip_values``。

**adaptive_histogram_equalization**

- ``images`` （必填）；``alpha`` 、``beta`` ∈ [0,1]；``radius`` 为 int 或 (x,y,z)。

**save_options**（``SaveOptionsConfig``）

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - 字段
     - 默认值
     - 说明
   * - ``save_intermediate``
     - ``false``
     - 是否写中间目录
   * - ``intermediate_steps``
     - ``[]``
     - 非空时仅列出的步骤写中间结果；**空列表** 且 ``save_intermediate: true`` 时每步都写

DICOM 整理配置参数（``habit sort-dicom``）
------------------------------------------

对应模式类 ``habit.core.dicom_sort.DicomSortConfig``； Sphinx API 见 :doc:`api/dicom_sort`。CLI：``habit sort-dicom -c <yaml>``。

**推荐模板**：仓库内 ``config/dicom_sort/config_sort_dicom.yaml``。遗留路径 ``config/preprocessing/config_image_preprocessing_sort_dicom.yaml`` 可为同一格式的副本，仅供旧文档链接兼容。

**扁平 YAML 顶层字段**

.. list-table::
   :header-rows: 1
   :widths: 24 18 58

   * - 字段
     - 默认值
     - 说明
   * - ``data_dir`` / ``out_dir``
     - 无（必填）
     - 输入树 / 默认输出根；相对路径相对于 YAML 目录解析
   * - ``f``
     - 无（必填，与 ``filename_format`` 二选一）
     - dcm2niix ``-f`` **原样**传递，不做路径解析
   * - ``filename_format``
     - ``null``
     - 弃用别名，语义同 ``f``
   * - ``dcm2niix_path``
     - ``null``
     - 省略则在 ``PATH`` 中查找 ``dcm2niix``；相对路径相对于 YAML 目录
   * - ``extra_args``
     - ``[]``
     - 逐项原样追加到 dcm2niix 命令行
   * - ``output_dir``
     - ``null``
     - 若设置则用作 ``-o``；否则用 ``out_dir``

详见 :doc:`user_guide/image_preprocessing_zh`。

生境分析配置参数
------------

对应 ``habit.core.habitat_analysis.config_schemas.HabitatAnalysisConfig``；CLI：``habit get-habitat -c <yaml>``。train 示例见 ``config/habitat/config_habitat_two_step.yaml`` ，predict 见 ``config/habitat/config_habitat_two_step_predict.yaml``。

**配置文件示例：**

.. code-block:: yaml

   run_mode: train
   pipeline_path: ./results/habitat_pipeline.pkl
   data_dir: ./file_habitat.yaml
   out_dir: ./results/habitat/train

   FeatureConstruction:
     voxel_level:
       method: concat(raw(delay2), raw(delay3), raw(delay5))
       params: {}

     supervoxel_level:
       supervoxel_file_keyword: '*_supervoxel.nrrd'
       method: mean_voxel_features()
       params: {}

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

   HabitatSegmentation:
     clustering_mode: two_step

     supervoxel:
       algorithm: kmeans
       n_clusters: 50
       random_state: 42
       max_iter: 300
       n_init: 10

     habitat:
       algorithm: kmeans
       max_clusters: 10
       habitat_cluster_selection_method:
         - inertia
         - silhouette
       fixed_n_clusters: null
       random_state: 42
       max_iter: 300
       n_init: 10

   processes: 2
   cap_processes_to_gpu_pool: true
   individual_subject_timeout_sec: 900
   individual_subject_spawn_timeout_sec: 120
   resume: true
   strict_checkpoint_hash: true
   checkpoint_dir: null
   force_rerun_subjects: []
   retry_failed_subjects: false
   individual_subject_auto_retry_rounds: 2
   individual_subject_parallel_mode: persistent
   persistent_worker_max_consecutive_failures: 1
   persistent_worker_recycle_after_tasks: 0
   clear_checkpoint_on_success: false
   plot_curves: true
   save_images: true
   save_results_csv: true
   random_state: 42
   verbose: true
   debug: false

**run_mode**: 运行模式

- **类型**: 字符串
- **必需**: 否
- **默认值**: ``train``
- **可选值**: ``train``, ``predict``
- **说明**: ``train`` 表示训练新模型，``predict`` 表示使用预训练模型进行预测。
- **示例**: ``train``

**pipeline_path**: Pipeline 文件路径

- **类型**: 字符串
- **必需**: 否（``predict`` 模式必填）
- **默认值**: ``null``
- **说明**: 指定训练好的 Pipeline 文件路径。
- **示例**: ``./results/habitat_pipeline.pkl``

**data_dir** / **out_dir**（生境分析顶层）

- **类型**: 字符串
- **必需**: 是
- **默认值**: 无（必填）
- **说明**: ``data_dir`` 可为目录或 ``file_habitat.yaml`` 等清单；``out_dir`` 为结果与 checkpoint 默认父目录。

**FeatureConstruction**: 特征提取设置

- **类型**: 对象
- **必需**: ``train`` 时必填；``predict`` 时可省略
- **默认值**: ``null``
- **说明**: 未配置时由校验拒绝或按运行模式报错。子块 ``voxel_level`` / ``supervoxel_level`` / ``preprocessing_*`` 如下。

**voxel_level**: 体素级特征提取

- ``method`` : 特征提取方法表达式

  - **类型**: 字符串
  - **必需**: 是
  - **默认值**: 无（必填）
  - **说明**: 支持函数式语法组合多个特征提取器。

- ``params`` : 体素级提取器参数字典

  - **类型**: 字典
  - **必需**: 否
  - **默认值**: ``{}``
  - **说明**: 传给 ``method`` 表达式中各提取器的键值（如 ``voxel_radiomics`` 的 ``params_file`` 也可写在 ``params`` 内，依实现解析）。
  - **可用方法及参数**:

    **raw(image_name)**:

      - **说明**: 提取原始图像体素值（最基础的特征）
      - **参数**: 无
      - **示例**: ``raw(delay2)``

    **concat(...)**:

      - **说明**: 拼接多个特征向量
      - **参数**: 接受多个特征提取表达式
      - **示例**: ``concat(raw(delay2), raw(delay3), raw(delay5))``

    **kinetic(...)**:

      - **说明**: 提取动力学特征（wash-in/wash-out 斜率等）
      - **参数**:

        - ``timestamps`` (str, 必需): 时间戳文件路径
        - 接受多个 ``raw(image_name)`` 表达式

      - **示例**: ``kinetic(raw(LAP), raw(PVP), raw(delay_3min), timestamps=...)``
      - **提取的特征**:

        - ``wash_in_slope`` : 洗入斜率
        - ``wash_out_slope_lap_pvp`` : LAP 到 PVP 的洗出斜率
        - ``wash_out_slope_pvp_dp`` : PVP 到延迟期的洗出斜率

    **local_entropy(...)**:

      - **说明**: 计算局部熵（衡量局部纹理复杂度）
      - **参数**:

        - ``kernel_size`` (int, 默认: ``3``): 局部邻域大小
        - ``bins`` (int, 默认: ``32``): 直方图分箱数

      - **示例**: ``local_entropy(raw(delay2), kernel_size=5, bins=32)``

    **voxel_radiomics(...)**:

      - **说明**: 提取体素级影像组学特征
      - **参数**:

        - ``params_file`` (str, 必需): PyRadiomics 参数文件路径
        - ``kernelRadius`` (int, 默认: ``1``): 局部邻域半径（1=3×3×3, 2=5×5×5）
        - ``voxelBatch`` (int, 默认: ``1000``): 体素批处理大小；``-1`` 表示一次处理 ROI 内全部体素（PyRadiomics 原生不分批）。设为正整数可限制内存占用（GPU 或大 ROI 时建议 ``512``–``1000``）
        - ``useTorchRadiomics`` (str, 默认: ``auto``): ``auto`` 在已安装 torch 且 CUDA 可用时使用 TorchRadiomics，否则回退 CPU PyRadiomics；``true`` 强制 torch；``false`` 始终 CPU
        - ``torchDevice`` (str, 默认: ``auto``): 单 GPU 设备；未设置 ``torchGpus`` 时生效
        - ``torchGpus`` (list/int/str): 允许使用的 GPU 编号，如 ``[0, 1, 2]`` 或 ``"0,1,2"``；设置后覆盖 ``torchDevice``
        - ``torchGpuCount`` (int, 可选): 从 ``torchGpus`` 中实际使用前 N 张卡
        - ``torchDtype`` (str, 默认: ``float32``): Torch 计算 dtype（``float32`` 或 ``float64``；``float64`` 更接近 CPU PyRadiomics）

      - **体素 GLCM 注意**: 请使用 ``config/radiomics/params_voxel_radiomics.yaml``（显式列出 21 个稳定 GLCM 特征）。
        若 ``params_file`` 中仅写 ``glcm:`` 而不列特征名，PyRadiomics 会计算全部 24 个 GLCM 特征；在
        ``kernelRadius=1``–``3`` 的小邻域内大量体素灰度均匀，GLCM 退化为 1×1 矩阵，**MCC / Imc1 / Imc2**
        的特征值或互信息计算会在 CUDA/MKL 上崩溃或产生 NaN。HABIT 在检测到未限制 GLCM 时会自动替换为
        上述 21 个稳定特征并记录 warning；若在 ``params_file`` 中已显式列出特征则尊重用户配置。

      - **示例**: ``voxel_radiomics(raw(delay2), params_file='./config/radiomics/params_voxel_radiomics.yaml', kernelRadius=1)``

  - **完整示例**:

    .. code-block:: yaml

       # 简单拼接原始图像
       voxel_level:
         method: concat(raw(delay2), raw(delay3), raw(delay5))
         params: {}
       
       # 提取动力学特征
       voxel_level:
         method: kinetic(raw(LAP), raw(PVP), raw(delay_3min))
         params:
           timestamps: ./timestamps.txt
       
       # 组合局部熵和原始值
       voxel_level:
         method: concat(raw(delay2), local_entropy(raw(delay2)))
         params:
           kernel_size: 5
           bins: 32

       # 体素级影像组学（纹理特征，计算较慢）
       voxel_level:
         method: voxel_radiomics(T2)
         params:
           params_file: ./config/radiomics/params_voxel_radiomics.yaml
           kernelRadius: 3
           voxelBatch: 1000
           useTorchRadiomics: auto
           # torchGpus: [0, 1]
           # torchGpuCount: 2

- ``params`` : 全局参数

  - **类型**: 字典
  - **必需**: 否
  - **默认值**: ``{}``
  - **说明**: 传递给所有特征提取器的公共参数。``voxel_radiomics`` 专用项（如 ``voxelBatch``、``useTorchRadiomics``）写在 ``params`` 中即可，**不必**出现在 ``method`` 表达式字符串里；未在表达式中列出的键会自动合并转发。
  - **常用参数**:

    - ``timestamps`` (str): 时间戳文件路径（用于 kinetic 方法）
    - ``kernel_size`` (int): 局部邻域大小（用于 local_entropy）
    - ``bins`` (int): 直方图分箱数（用于 local_entropy）
    - ``params_file`` (str): PyRadiomics 参数文件（用于 voxel_radiomics）
    - ``kernelRadius`` (int): 体素级组学邻域半径（用于 voxel_radiomics）
    - ``voxelBatch`` (int): 体素级组学批大小（用于 voxel_radiomics；默认 ``1000``；``-1`` 表示不分批）
    - ``useTorchRadiomics`` (str): 是否使用 TorchRadiomics 加速（``auto`` / ``true`` / ``false``）
    - ``torchDevice`` (str): 单 GPU 设备（未设置 ``torchGpus`` 时）
    - ``torchGpus`` (list/int/str): 允许使用的 GPU 列表
    - ``torchGpuCount`` (int): 实际使用的 GPU 数量上限
    - ``torchDtype`` (str): Torch dtype（用于 voxel_radiomics torch 后端）

**supervoxel_level**: 超像素级特征提取 (可选)

- **整块默认值**: ``null`` （省略表示不使用超像素级块；``two_step`` 训练通常需要配置）

- ``supervoxel_file_keyword`` : 超像素文件匹配模式

  - **类型**: 字符串
  - **必需**: 否（配置 ``supervoxel_level`` 时生效）
  - **默认值**: ``*_supervoxel.nrrd``
  - **说明**: 用于匹配已有的超像素分割文件（由 two_step 模式生成）。
  - **示例**: ``"*_supervoxel.nrrd"``

- ``method`` : 特征聚合/提取方法

  - **类型**: 字符串
  - **必需**: 否（配置 ``supervoxel_level`` 时建议填写）
  - **默认值**: ``mean_voxel_features()``
  - **说明**: 定义如何从体素特征聚合到超像素，或直接从超像素提取特征。
  - **可用方法及参数**:

    **mean_voxel_features()**:

      - **说明**: 计算每个超像素内体素特征的平均值（最常用）
      - **参数**: 无
      - **用途**: 将体素级特征（如 ``voxel_level`` 提取的特征）聚合到超像素级
      - **示例**: ``mean_voxel_features()``

    **supervoxel_radiomics(params_file=...)**:

      - **说明**: 对每个超体素 label 提取 **整 ROI** 影像组学纹理（非体素 kernel 邻域）
      - **离散化**: 在全部超体素并集 mask（``sv_map > 0``）上 **一次** PyRadiomics ``_applyBinning``，再逐 label 用 ``cMatrices`` 建矩阵
      - **矩阵后端**: ``useSupervoxelCext`` 默认 ``auto``：已编译 ``supervoxel_cext``（``pip install -e .``）时用 C 扩展批量建矩阵；否则回退到原有 Torch/PyRadiomics 堆叠矩阵路径。设为 ``false`` 时 **强制** 使用 Torch/PyRadiomics 堆叠矩阵（``matrix_backend=torch_cmatrices``），即使 C 扩展已编译
      - **特征后端**: ``useTorchRadiomics`` 解析为 torch 时用 TorchRadiomics（GPU/CPU torch）；否则 CPU PyRadiomics（语义相同）
      - **参数**（写在 ``FeatureConstruction.supervoxel_level.params``，可继承 ``voxel_level.params`` 中的 torch 项）:

        - ``params_file`` (str, 必需): PyRadiomics 参数 YAML（仅 featureClass / setting）
        - ``supervoxelBatch`` (int): 批分组大小，默认 ``64``（非 kernel 半径）
        - ``supervoxelUnionBboxCrop`` (bool): 是否裁切到并集 bbox，默认 ``true``
        - ``useSupervoxelCext`` (str | bool): ``auto`` / ``true`` / ``false``，默认 ``auto``；须写在 ``supervoxel_level.params``（不要写在 ``params_file``）
        - ``useTorchRadiomics`` (str): ``auto`` / ``true`` / ``false``
        - ``torchGpus`` / ``torchGpuCount`` / ``torchDevice`` / ``torchDtype``: 同体素级

      - **注意**: ``kernelRadius`` 仅用于 ``voxel_radiomics``，``supervoxel_radiomics`` 不使用
      - **用途**: 不依赖 ``voxel_level`` 特征，直接从超体素区域提取纹理等组学特征
      - **示例**: ``supervoxel_radiomics(T2)`` 且 ``params_file: ./config/radiomics/params_supervoxel_radiomics.yaml``

  - **方法对比**:

    - ``mean_voxel_features()`` : 依赖 ``voxel_level`` 特征，速度快，适合大多数场景
    - ``supervoxel_radiomics()`` : 独立 ROI 组学；union-mask 一次 bin + 逐 label 提取；特征数值与旧版逐 label ``execute``（per-label bin）**不一致**

  - **完整示例**:

    .. code-block:: yaml

       # 场景1：聚合体素特征（推荐）
       supervoxel_level:
         supervoxel_file_keyword: '*_supervoxel.nrrd'
         method: mean_voxel_features()
         params: {}
       
       # 场景2：直接提取影像组学特征
       supervoxel_level:
         supervoxel_file_keyword: '*_supervoxel.nrrd'
         method: supervoxel_radiomics(T2)
         params:
           params_file: ./config/radiomics/params_supervoxel_radiomics.yaml
           supervoxelBatch: 64
           useSupervoxelCext: auto
           useTorchRadiomics: auto
           # torchGpus: [0, 1]

- ``params`` : 参数

  - **类型**: 字典
  - **必需**: 否
  - **默认值**: ``{}``
  - **说明**: 传递给特征提取器的参数。``supervoxel_radiomics`` 常用键：
    ``params_file``、``supervoxelBatch``、``supervoxelUnionBboxCrop``、``useSupervoxelCext``、
    ``useTorchRadiomics``、``torchGpus``、``torchGpuCount``、``torchDtype``（torch 项可继承
    ``voxel_level.params``）。

**preprocessing_for_subject_level**: 个体级别预处理 (可选)

- ``methods`` : 预处理方法列表

  - **类型**: 列表
  - **必需**: 否
  - **默认值**: ``[]``
  - **说明**: 在个体水平对特征进行预处理，消除个体内异常值和尺度差异。底层由
    ``PreprocessingMethodFactory`` 统一调度，DataFrame 进/出（见下文「生境特征预处理实现与扩展」）。
  - **注意**: ``two_step`` 与 ``direct_pooling`` 模式下，个体级别不应使用会删列的方法（``variance_filter`` 、``correlation_filter``），否则跨受试者拼接后会出现列不一致；``two_step`` 会在配置校验阶段直接拒绝。``one_step`` 模式可在个体级别使用删列型方法（每例独立聚类）。
  - **支持方法及参数**:

    **winsorize (缩尾处理)**:

      - ``winsor_limits`` (list, 默认: ``[0.05, 0.05]``): 下限和上限的截断比例
      - ``global_normalize`` (bool, 默认: ``false``): 是否全局归一化（跨所有特征）

    **minmax (最小-最大归一化)**:

      - ``global_normalize`` (bool, 默认: ``false``): 是否全局归一化

    **zscore (Z-Score 标准化)**:

      - ``global_normalize`` (bool, 默认: ``false``): 是否全局标准化

    **robust (鲁棒标准化)**:

      - ``global_normalize`` (bool, 默认: ``false``): 是否全局归一化
      - 使用分位距（IQR）进行缩放，对异常值鲁棒

    **log (对数变换)**:

      - ``global_normalize`` (bool, 默认: ``false``): 是否全局变换
      - 自动处理负值（平移后再取对数）

    **variance_filter (低方差筛选)**:

      - ``variance_threshold`` (float, 默认: ``0.0``): 保留方差大于该阈值的特征
      - 说明: 该方法会删除特征列

    **correlation_filter (高相关筛选)**:

      - ``corr_threshold`` (float, 默认: ``0.95``): 相关系数绝对值大于该阈值时删除冗余特征
      - ``corr_method`` (str, 默认: ``spearman``): 相关系数方法，可选 ``pearson``/``spearman``/``kendall``
      - 说明: 该方法会删除特征列

  - **示例**:

    .. code-block:: yaml

       # 去除异常值后归一化
       - method: winsorize
         winsor_limits: [0.05, 0.05]
         global_normalize: true
       - method: minmax
         global_normalize: true
       
       # Z-Score 标准化
       - method: zscore
         global_normalize: false

**preprocessing_for_group_level**: 群体级别预处理 (可选)

- ``methods`` : 预处理方法列表

  - **类型**: 列表
  - **必需**: 否
  - **默认值**: ``[]``
  - **说明**: 在群体水平对特征进行预处理，通常用于离散化以提高聚类的稳定性。
  - **适用模式**: 仅 ``two_step`` 与 ``direct_pooling``；``one_step`` 模式的 pipeline 不含群体级预处理步骤，配置了也不会生效。
  - **支持方法及参数**:

    **binning (特征离散化/分箱)**:

      - ``n_bins`` (int, 默认: ``10``): 分箱数量
      - ``bin_strategy`` (str, 默认: ``uniform``): 分箱策略，可选:

        - ``uniform`` : 均匀分箱（等宽）
        - ``quantile`` : 分位数分箱（等频）
        - ``kmeans`` : K-means 聚类分箱

      - ``global_normalize`` (bool, 默认: ``false``): 是否全局分箱（跨所有特征）

    **winsorize (缩尾处理)**:

      - ``winsor_limits`` (list, 默认: ``[0.05, 0.05]``): 下限和上限的截断比例
      - ``global_normalize`` (bool, 默认: ``false``): 是否全局归一化

    **minmax / zscore / robust / log**:

      - 同 ``preprocessing_for_subject_level`` ，但作用于群体汇总后的数据

    **variance_filter / correlation_filter (推荐放在群体级执行)**:

      - 用于无监督场景下的特征删列，降低噪声与冗余
      - ``variance_filter`` 参数: ``variance_threshold``
      - ``correlation_filter`` 参数: ``corr_threshold`` 、``corr_method``
      - 建议: 在训练阶段确定保留列，预测阶段复用同一列集合

  - **示例**:

    .. code-block:: yaml

       # 均匀分箱（推荐用于生境分析）
       - method: binning
         n_bins: 10
         bin_strategy: uniform
         global_normalize: false
       
       # 分位数分箱（等频分箱）
       - method: binning
         n_bins: 20
         bin_strategy: quantile
         global_normalize: false

**YAML 结构要点**

``preprocessing_for_*_level`` 下必须是 **单个** ``methods:`` 键，其值为 **列表**；不要把 ``methods:`` 写两次，也不要把列表项写在 ``methods:`` 块外：

.. code-block:: yaml

   # 正确
   preprocessing_for_group_level:
     methods:
       - method: winsorize
         winsor_limits: [0.05, 0.05]
       - method: variance_filter
         variance_threshold: 0.0

   # 错误：重复 methods 键，variance_filter 未在列表内
   preprocessing_for_group_level:
     methods:
       - method: winsorize
     methods:
       - method: variance_filter

**三种 clustering_mode 下的预处理生效矩阵**

.. list-table::
   :header-rows: 1
   :widths: 22 18 18 18

   * - 预处理块
     - one_step
     - two_step
     - direct_pooling
   * - ``preprocessing_for_subject_level``
     - 生效
     - 生效（禁止 ``variance_filter`` / ``correlation_filter``）
     - 生效
   * - ``preprocessing_for_group_level``
     - **不生效**
     - 生效（Stage 2 群体级）
     - 生效（池化后群体级）

**推荐 pipeline（训练 → 预测复用）**

- **two_step / direct_pooling（训练）**：subject 级 ``winsorize`` + ``minmax`` （或 ``zscore``）→ group 级 ``binning`` （离散化利于聚类）→ 可选 ``variance_filter`` / ``correlation_filter`` （删列型，训练期 ``fit`` 缓存列集合）。
- **two_step / direct_pooling（预测）**：与训练相同的 ``methods`` 顺序；``PreprocessingState`` 从 ``habitat_pipeline.pkl`` 加载，**勿改删列型方法的阈值** 以免列不一致。
- **one_step**：仅配置 ``preprocessing_for_subject_level``；群体级块会被忽略。

**生境特征预处理实现与扩展**

- **统一接口**: 所有内置与自定义方法均实现 ``BaseFeaturePreprocessing`` ，
  通过 ``@register_preprocessing`` 注册到 ``PreprocessingMethodFactory``。
- **执行路径**: ``preprocessing_for_subject_level`` → 个体级无状态
  ``apply_stateless_preprocessing``；``preprocessing_for_group_level`` →
  ``PreprocessingState.fit/transform`` （训练期缓存 ``baseline`` 与各步
  ``step_states`` ，预测期复用）。
- **删列型方法**: ``variance_filter`` 、``correlation_filter`` 设
  ``changes_columns=True``；``two_step`` 禁止在 subject 级使用（见上文注意事项）。
- **新增方法**:

  1. 参考 ``habit/core/habitat_analysis/feature_preprocessing/custom_preprocessing_template.py``
  2. 在 ``config_schemas.PreprocessingMethod.method`` Literal 中追加方法名
  3. 若删列，同步更新 ``DROPPING_PREPROCESSING_METHODS``
  4. 确保模块被 import，使注册装饰器执行

- **兼容性**: YAML 配置格式未变；旧版 ``habitat_pipeline.pkl`` 若含重构前的
  ``PreprocessingState`` 结构，需重新 train。

**HabitatSegmentation**: 生境分割设置

- **类型**: 对象
- **必需**: ``train`` 时必填；``predict`` 时建议保留（至少 ``clustering_mode``）
- **默认值**: ``null`` （完全省略时 Pydantic 使用 ``HabitatSegmentationConfig`` 默认子块，见下表）

- ``clustering_mode`` : 聚类策略

  - **类型**: 字符串
  - **必需**: 否
  - **默认值**: ``two_step``
  - **可选值**:

    - ``one_step`` : 直接对体素进行聚类。
    - ``two_step`` : 先生成超像素，再对超像素进行聚类生成生境。
    - ``direct_pooling`` : 直接汇总所有受试者的体素进行聚类（计算量大）。

  - **示例**: ``two_step``

**supervoxel**: 超像素聚类设置 (仅用于 ``two_step`` 模式)

- ``algorithm`` : 聚类或分割算法

  - **类型**: 字符串
  - **默认值**: ``kmeans``
  - **可选值**（与 ``SupervoxelClusteringConfig`` 一致）:

    - ``kmeans`` : K-means
    - ``gmm`` : 高斯混合；底层实现默认 ``covariance_type='full'`` ，当前 ``SupervoxelClusteringConfig`` **未** 声明该字段，YAML 中填写可能被忽略
    - ``slic`` : SLIC 超像素；使用下列 ``compactness`` / ``sigma`` / ``enforce_connectivity``

- ``n_clusters`` : 超像素（或 SLIC）数量

  - **类型**: 整数
  - **默认值**: ``50``
  - **说明**: ``two_step`` 下每个被试的超像素个数，常用 30–100。

- ``random_state`` : 随机种子

  - **类型**: 整数或 ``null``
  - **默认值**: ``null`` （继承 ``HabitatAnalysisConfig.random_state``）

- ``max_iter`` : 最大迭代次数

  - **类型**: 整数
  - **默认值**: ``300``

- ``n_init`` : 初始化次数

  - **类型**: 整数
  - **默认值**: ``10``

- ``compactness`` (float, 默认 ``0.1``): **仅 ``slic``**，特征与空间紧致度权衡。

- ``sigma`` (float, 默认 ``0.0``): **仅 ``slic``**，SLIC 前高斯平滑宽度。

- ``enforce_connectivity`` (bool, 默认 ``true``): **仅 ``slic``**，是否约束连通性。

- ``one_step_settings`` : 嵌套的 one-step 自动簇数选择（见下文 **one_step_settings**）。

- **完整示例**:

  .. code-block:: yaml

     supervoxel:
       algorithm: kmeans
       n_clusters: 50
       random_state: 42
       max_iter: 300
       n_init: 10

     # SLIC 示例
     supervoxel:
       algorithm: slic
       n_clusters: 80
       compactness: 0.1
       sigma: 0.0
       enforce_connectivity: true
       random_state: 42

**one_step_settings**: One-Step 模式设置 (仅用于 ``one_step`` 模式)

- ``min_clusters`` : 最小聚类数

  - **类型**: 整数
  - **默认值**: ``2``
  - **说明**: 自动选择时的下限

- ``max_clusters`` : 最大聚类数

  - **类型**: 整数
  - **默认值**: ``10``
  - **说明**: 自动选择时的上限

- ``fixed_n_clusters`` : 固定聚类数

  - **类型**: 整数或 null
  - **默认值**: ``null``
  - **说明**: 若设置，则跳过自动选择，直接使用该值。

- ``selection_method`` : 自动选择指标

  - **类型**: 字符串
  - **默认值**: ``silhouette``
  - **可选值及说明**:

    - ``silhouette`` : 轮廓系数（-1 到 1，越接近 1 表示聚类越紧密）
    - ``calinski_harabasz`` : Calinski-Harabasz 指数（越大表示聚类越好）
    - ``davies_bouldin`` : Davies-Bouldin 指数（越小表示聚类越好）
    - ``inertia`` : 簇内平方和（越小表示聚类越紧密，内部用 Kneedle 选拐点）
    - ``kneedle`` : Kneedle 方法（对 inertia 曲线归一化后选最大偏离点）

  - **推荐**: ``silhouette`` （综合性能最佳）

- ``plot_validation_curves`` : 是否绘制验证曲线

  - **类型**: 布尔值
  - **默认值**: ``true``
  - **说明**: 生成不同聚类数下的指标曲线图，帮助理解自动选择结果

**habitat**: 生境聚类设置

- ``algorithm`` : 聚类算法

  - **类型**: 字符串
  - **默认值**: ``kmeans``
  - **可选值**:

    - ``kmeans`` : K-means 聚类
    - ``gmm`` : 高斯混合模型

- ``max_clusters`` : 最大生境数

  - **类型**: 整数
  - **必需**: 否
  - **默认值**: ``10``
  - **说明**: 自动选择生境数时的上限。推荐范围: 5-10。
  - **示例**: ``10``

- ``min_clusters`` : 最小生境数

  - **类型**: 整数
  - **默认值**: ``2``
  - **说明**: 自动选择生境数时的下限。

- ``habitat_cluster_selection_method`` : 自动选择指标

  - **类型**: 列表或字符串
  - **默认值**: ``inertia`` （YAML 中可写为字符串或单元素列表）
  - **可选值及说明**:

    - ``inertia`` : 簇内平方和（越小越好，适用于 kmeans，内部用 Kneedle 选拐点）
    - ``kneedle`` : Kneedle 方法（对 inertia 曲线归一化后选最大偏离点）
    - ``silhouette`` : 轮廓系数（-1 到 1，越接近 1 越好）
    - ``calinski_harabasz`` : Calinski-Harabasz 指数（越大越好）
    - ``davies_bouldin`` : Davies-Bouldin 指数（越小越好）
    - ``aic`` : 赤池信息准则（越小越好，仅用于 gmm）
    - ``bic`` : 贝叶斯信息准则（越小越好，仅用于 gmm）

  - **说明**: 可指定多个指标，系统会综合评估选择最佳生境数。
  - **示例**: ``[inertia, silhouette]``

- ``fixed_n_clusters`` : 固定生境数

  - **类型**: 整数或 null
  - **默认值**: ``null``
  - **说明**: 若设置为具体数值，则跳过自动选择，直接使用该生境数。

- ``random_state`` : 随机种子

  - **类型**: 整数或 ``null``
  - **默认值**: ``null`` （继承 ``HabitatAnalysisConfig.random_state``）
  - **说明**: **direct_pooling / two_step** 群体生境聚类；**one_step** 个体体素→生境聚类（优先于 ``supervoxel.random_state``）。显式写入时覆盖顶层。

- ``max_iter`` : 最大迭代次数

  - **类型**: 整数
  - **默认值**: ``300`` (kmeans) 或 ``100`` (gmm)

- ``n_init`` : 初始化次数

  - **类型**: 整数
  - **默认值**: ``10`` (kmeans) 或 ``1`` (gmm)

- **完整示例**:

  .. code-block:: yaml

     # 自动选择生境数（推荐）
     habitat:
       algorithm: kmeans
       max_clusters: 10
       min_clusters: 2
       habitat_cluster_selection_method:
         - inertia
         - silhouette
       fixed_n_clusters: null
       random_state: 42
     
     # 固定生境数
     habitat:
       algorithm: kmeans
       fixed_n_clusters: 5
       random_state: 42

**postprocess_supervoxel / postprocess_habitat**: 连通域后处理设置

- **类型**: 字典
- **必需**: 否
- **默认值**: ``enabled: false``
- **说明**:

  - ``postprocess_supervoxel`` 作用于超体素标签图（主要 two_step 阶段）。
  - ``postprocess_habitat`` 作用于最终生境标签图（one_step/two_step/direct_pooling）。
  - 当前实现采用 SimpleITK 快路径：先按标签移除小连通域，再按最近种子标签回填。
  - 该流程旨在减少碎片并保持 ROI 内体素不丢失。

- **子参数**（与 ``ConnectedComponentPostprocessConfig`` 一致）:

  - ``enabled`` (bool, 默认: ``false``)
  - ``min_component_size`` (int, 默认: ``30``, ≥1)
  - ``connectivity`` (1 / 2 / 3): 6/18/26 邻接
  - ``reassign_method`` : 当前仅为 ``neighbor_vote`` （占位）
  - ``max_iterations`` (int, 默认 ``3``, ≥1): 清理迭代上限

- **示例**:

  .. code-block:: yaml

     HabitatSegmentation:
       postprocess_supervoxel:
         enabled: false
         min_component_size: 30
         connectivity: 1
         reassign_method: neighbor_vote
         max_iterations: 3

       postprocess_habitat:
         enabled: true
         min_component_size: 30
         connectivity: 1
         reassign_method: neighbor_vote
         max_iterations: 3

**plot_curves**: 是否生成和保存图表

- **类型**: 布尔值
- **默认值**: ``true``
- **说明**: 群体聚类自动寻优 k 时，若传入 logger，日志会输出 ``Trying N cluster(s) [i/total]`` 与 ``Cluster search finished: selected k=...`` ，便于判断寻优进度（``predict`` 模式强制关闭绘图）。

生境 Stage-1 并行与断点续训（顶层字段总表）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下列字段位于生境 YAML **顶层**（与 ``data_dir`` 同级）。``predict`` 模式忽略并行与 checkpoint 相关项。续训操作步骤见 :doc:`user_guide/habitat_segmentation_zh` 「断点续训详解」。

.. list-table::
   :header-rows: 1
   :widths: 26 12 12 14 36

   * - 字段
     - 类型
     - 默认
     - config_hash
     - 简述
   * - ``processes``
     - int
     - ``2``
     - 否
     - Stage 1 最大并行 worker 数；峰值内存约 ``processes × 单被试内存``
   * - ``cap_processes_to_gpu_pool``
     - bool
     - ``true``
     - 否
     - Torch CUDA 组学时：``true`` 将 worker 限制为 ``len(torchGpus)`` ；``false`` 保留 ``processes`` 并由多 worker 共享 GPU
   * - ``individual_subject_timeout_sec``
     - float / ``null``
     - ``900``
     - 否
     - 单被试墙钟上限（秒）；``null`` 禁用
   * - ``individual_subject_graceful_shutdown_sec``
     - float
     - ``15``
     - 否
     - 超时后 ``terminate()`` 再等待的秒数，之后 ``kill()`` 子进程
   * - ``individual_subject_spawn_timeout_sec``
     - float / ``null``
     - ``120``
     - 否
     - 子进程启动阶段上限；``null`` 禁用（防 import 卡死阻塞父进程轮询）
   * - ``on_subject_failure``
     - str
     - ``continue``
     - 否
     - ``continue`` 记录失败并尽量继续；``fail_fast`` 任一失败即中止
   * - ``oom_backoff``
     - bool
     - ``true``
     - 否
     - ``MemoryError`` 后按 ``oom_reduce_workers_by`` 降 worker（不处理 native 崩溃）
   * - ``oom_reduce_workers_by``
     - int
     - ``1``
     - 否
     - 每次 OOM 减少的 worker 数
   * - ``resume``
     - bool
     - ``true``
     - —
     - 从 checkpoint 跳过已完成被试；仅 ``train``
   * - ``checkpoint_dir``
     - str / ``null``
     - ``null``
     - 否
     - 默认 ``<out_dir>/.habitat_checkpoint``
   * - ``force_rerun_subjects``
     - list[str]
     - ``[]``
     - 否
     - 续训时强制重跑列表中的被试 ID
   * - ``retry_failed_subjects``
     - bool
     - ``false``
     - 否
     - **下次** ``resume`` 启动时重跑 manifest 中全部 ``failed_subjects``
   * - ``individual_subject_auto_retry_rounds``
     - int
     - ``2``
     - 否
     - **同一次** ``train`` 内 Stage 1 失败自动重试轮数；``0`` 关闭
   * - ``individual_subject_parallel_mode``
     - str
     - ``persistent``
     - 否
     - ``persistent`` 长生命周期 worker；``isolated`` 每被试 spawn
   * - ``persistent_worker_max_consecutive_failures``
     - int
     - ``1``
     - 否
     - ``persistent`` 下连续失败 N 次后重启该槽位
   * - ``persistent_worker_recycle_after_tasks``
     - int
     - ``0``
     - 否
     - ``persistent`` 下成功 N 次后回收 worker；``0`` 关闭
   * - ``clear_checkpoint_on_success``
     - bool
     - ``false``
     - 否
     - 训练全部成功后删除 checkpoint 目录

**processes**（生境分析顶层）: 个体级步骤并行进程数

- **类型**: 整数
- **默认值**: ``2`` （须 ``> 0``）
- **说明**: 见上表；与 ``cap_processes_to_gpu_pool`` 、``FeatureConstruction.*.params`` 中的 ``torchGpus`` 共同决定实际并发与 GPU 绑定。

**cap_processes_to_gpu_pool**（生境分析顶层）: 是否将 Stage 1 worker 数限制为 GPU 池大小

- **类型**: 布尔值
- **默认值**: ``true``
- **说明**: 当 ``useTorchRadiomics`` 启用 CUDA（``true`` 或 ``auto`` 且检测到 CUDA）时：

  - ``true`` （默认）：有效 worker 数 ``min(processes, len(torchGpus))`` ，每槽位绑定一张 GPU（``gpuSlotIndex``），显存争抢较少；
  - ``false`` ：保留完整 ``processes`` ；多 worker 通过 ``gpuSlotIndex % len(torchGpus)`` **共享** GPU，利于「单 GPU、多 CPU」上并行 CPU 步骤，但 GPU radiomics 可能同卡 OOM。

- **不纳入 config_hash** ；续训时可改。
- **CPU-only** （``useTorchRadiomics: false`` 或无 CUDA）时无效果。

**individual_subject_timeout_sec**（生境分析顶层）: 个体级并行阶段单被试墙钟时间上限

- **类型**: 浮点数 / 整数（秒）或 ``null``
- **默认值**: ``900`` （15 分钟）；YAML 可省略以使用默认。
- **说明**: 超时则跳过该被试（记入失败）并继续；``null`` 表示不启用单被试超时。多进程下子进程可能仍在后台运行直至自行结束。

**individual_subject_graceful_shutdown_sec**（生境分析顶层）: 超时后优雅终止的等待秒数

- **类型**: 浮点数（秒）
- **默认值**: ``15``
- **说明**: 单被试超过 ``individual_subject_timeout_sec`` 后，父进程先 ``terminate()`` ，等待本字段指定秒数后再 ``kill()`` 强杀隔离子进程。

**individual_subject_spawn_timeout_sec**（生境分析顶层）: 子进程启动阶段墙钟上限

- **类型**: 浮点数 / 整数（秒）或 ``null``
- **默认值**: ``120``
- **说明**: 从派发子进程到其开始处理被试之间的上限；超时则将该被试记为失败并继续，避免父进程在 spawn/import 卡死时无限阻塞。``null`` 表示不限制启动时间。

**on_subject_failure**（生境分析顶层）: 个体级并行失败策略

- **类型**: 字符串
- **默认值**: ``continue``
- **可选值**:

  - ``continue`` ：记录失败被试，在仍有成功被试时继续 Stage 2
  - ``fail_fast`` ：任一被试失败或超时即中止整个 run

**oom_backoff**（生境分析顶层）: 内存错误后降低并行度

- **类型**: 布尔值
- **默认值**: ``true`` （schema）；仓库 ``config/habitat/*.yaml`` 示例多为 ``false`` ，可按机器内存自行选择
- **说明**: 为 ``true`` 时，若隔离子进程抛出 Python ``MemoryError`` ，后续 pending 被试的 worker 数按 ``oom_reduce_workers_by`` 递减（最少 1）。**不处理** native 崩溃（如 Windows exit code ``3221225477`` / ``0xC0000005``）。

**oom_reduce_workers_by**（生境分析顶层）: 每次 OOM 减少的 worker 数

- **类型**: 整数
- **默认值**: ``1``
- **说明**: 仅 ``oom_backoff: true`` 时生效。

**resume**（生境分析顶层）: 个体级断点续训（Stage 1）

- **类型**: 布尔值
- **默认值**: ``true``
- **说明**: 为 ``true`` 时从 ``checkpoint_dir`` （默认 ``<out_dir>/.habitat_checkpoint``）读取 ``manifest.json`` ，跳过 ``completed_subjects`` 并从 ``subjects/{id}.pkl`` 加载结果；``failed_subjects`` 中的被试在**下次** ``resume`` 启动时**不会自动重试**（除非 ``retry_failed_subjects: true`` 或 ``force_rerun_subjects``）。**同一次** ``train`` 运行内，默认由 ``individual_subject_auto_retry_rounds`` 自动重试 Stage 1 失败被试。仅 ``run_mode: train`` 生效。
- **CLI**: ``habit get-habitat --resume`` 等效于 ``resume: true``。
- **详见**: :doc:`user_guide/habitat_segmentation_zh` 中「断点续训详解」。
- **并行可靠性计划**: 仓库根下 ``docs/HABITAT_PARALLEL_RELIABILITY_PLAN.md`` （GPU worker 槽位、processes 上限、Phase 2/3 路线图）。

**strict_checkpoint_hash**（生境分析顶层）: checkpoint hash 不兼容时是否报错

- **类型**: 布尔值
- **默认值**: ``true``
- **说明**: 与 ``resume: true`` 联用。为 ``true`` （默认）时，若 ``manifest.json`` 的 ``config_hash`` 或 ``run_mode`` 与当前 YAML 不兼容，抛出 ``CheckpointConfigHashError`` 并保留 checkpoint 目录；为 ``false`` 时记录警告并删除 checkpoint 后 fresh 重跑。仅 Stage-2 配置变更导致的 legacy hash 迁移仍允许续训。
- **不纳入 config_hash** ；续训时可改。

**checkpoint_dir**（生境分析顶层）: checkpoint 根目录

- **类型**: 字符串或 ``null``
- **默认值**: ``null`` （``train`` → ``<out_dir>/.habitat_checkpoint`` ；``predict`` → ``<out_dir>/.habitat_predict_checkpoint``）
- **说明**: 续训时必须与上次使用同一目录；可与 ``out_dir`` 分离（显式指定路径）。

**force_rerun_subjects**（生境分析顶层）: 强制重跑的被试 ID

- **类型**: 字符串列表
- **默认值**: ``[]``
- **说明**: ``resume: true`` 时仍重新处理列表中的被试（从 completed/failed 中移除并重跑）。

**retry_failed_subjects**（生境分析顶层）: 自动重跑 checkpoint 中全部失败被试

- **类型**: 布尔值
- **默认值**: ``false``
- **说明**: ``resume: true`` 时，将 ``manifest.json`` 里 ``failed_subjects`` 中的被试自动加入待处理队列并重新跑个体级 Stage 1。已成功被试仍跳过（除非同时出现在 ``force_rerun_subjects`` 中）。

**individual_subject_auto_retry_rounds**（生境分析顶层）: 同一次 train 运行内自动重试失败被试

- **类型**: 整数
- **默认值**: ``2``
- **说明**: 个体级 Stage 1 首轮并行结束后，若 checkpoint 中仍有 ``failed_subjects`` ，在同一进程内自动再跑最多该轮数次（仅重试仍失败的被试）。``0`` 表示关闭（保持旧行为）。与 ``retry_failed_subjects`` 不同：后者只在**下次** ``resume`` 启动时生效；本项在**当前** ``get-habitat`` / ``fit()`` 内生效。``on_subject_failure: fail_fast`` 时，会在全部重试轮次用尽后仍失败才报错。

**individual_subject_parallel_mode**（生境分析顶层）: 个体级 Stage 1 并行执行策略

- **类型**: 字符串
- **默认值**: ``persistent``
- **可选值**: ``isolated`` 、``persistent``
- **说明**: ``persistent`` （默认）：每个 worker 槽位一个长生命周期子进程，在同一 ``train`` 运行内复用（含 auto-retry 各轮），减少重复 import/spawn。``isolated`` ：每个被试单独 ``spawn`` 子进程（更强隔离，适合 pipeline 无法 pickle 或排查 spawn 问题时）。单 GPU 时 persistent 仍为串行，主要摊销启动成本。``processes=1`` 且 ``individual_subject_timeout_sec: null`` 时，两种模式均走主进程顺序执行、不 spawn。``predict`` 模式忽略。

**persistent_worker_max_consecutive_failures**（生境分析顶层）: 持久 worker 连续失败重启阈值

- **类型**: 整数
- **默认值**: ``1``
- **说明**: 仅 ``individual_subject_parallel_mode: persistent`` 时生效。某 worker 槽位连续失败达到该次数后，父进程终止并重启该槽位 worker，再处理后续被试。

**persistent_worker_recycle_after_tasks**（生境分析顶层）: 持久 worker 定期回收

- **类型**: 整数
- **默认值**: ``0``
- **说明**: 仅 ``persistent`` 模式生效。worker 连续成功处理该次数任务后主动退出并由父进程重启，用于缓解 GPU 显存缓慢泄漏。``0`` 表示关闭定期回收。

**clear_checkpoint_on_success**（生境分析顶层）: 训练成功后删除 checkpoint

- **类型**: 布尔值
- **默认值**: ``false``
- **说明**: 为 ``true`` 时 Stage 1 + Stage 2 全部成功后删除整个 checkpoint 目录。

**config_hash 与续训兼容性**

- **参与 hash**（Stage 1 个体级；变更则清空 checkpoint）：``data_dir`` 、``FeatureConstruction.voxel_level`` / ``preprocessing_for_subject_level`` / ``supervoxel_level`` 、``HabitatSegmentation.clustering_mode`` 、个体聚类块（``two_step`` → ``supervoxel`` ；``one_step`` → ``supervoxel`` + ``habitat`` ）。
- **不参与 hash**（可 ``resume: true`` 继续）：``preprocessing_for_group_level`` 、``two_step``/``direct_pooling`` 的群体 ``habitat.*`` 、``processes`` 、``cap_processes_to_gpu_pool`` 、``strict_checkpoint_hash`` 、``individual_subject_timeout_sec`` 、``individual_subject_graceful_shutdown_sec`` 、``individual_subject_spawn_timeout_sec`` 、``plot_curves`` 、``save_results_csv`` 、``save_images`` 、``verbose`` 、``debug`` 、``on_subject_failure`` 、``oom_backoff`` 、``oom_reduce_workers_by`` 、``retry_failed_subjects`` 、``individual_subject_auto_retry_rounds`` 、``individual_subject_parallel_mode`` 、``persistent_worker_max_consecutive_failures`` 、``persistent_worker_recycle_after_tasks`` 、``force_rerun_subjects`` 、``out_dir`` 等。
- ``manifest.json`` 另存 ``individual_config_hash`` （与 ``config_hash`` 相同）；旧版仅全量 hash 的 manifest 在仅改 Stage 2 配置时会迁移 hash 并保留 pkl。
- 程序在 ``resume: true`` 启动时自动比较 hash。个体级 hash 不一致且无法判定为 Stage 2 漂移时，默认（``strict_checkpoint_hash: true``）抛出 ``CheckpointConfigHashError`` ；设为 ``false`` 时记录警告并删除 checkpoint。

**checkpoint 目录结构**

.. code-block:: text

   <checkpoint_dir>/
   ├── manifest.json      # completed_subjects, failed_subjects, config_hash, individual_config_hash, stage
   └── subjects/
       └── {subject_id}.pkl

**三种 clustering_mode 的 checkpoint 边界**

- ``two_step`` / ``one_step`` ：在 ``merge_supervoxel_features`` 之后保存（``supervoxel_df``）
- ``direct_pooling`` ：在 ``individual_preprocessing`` 之后保存（体素级 ``features`` ，pkl 较大）
- Stage 2（combine / concat / group 聚类）均**无** checkpoint

**save_results_csv**: 是否将结果保存为 CSV 文件

- **类型**: 布尔值
- **默认值**: ``true``

**random_state**（生境分析顶层）

- **类型**: 整数
- **默认值**: ``42``

**debug**（生境分析顶层）

- **类型**: 布尔值
- **默认值**: ``false``

**habitat_pipeline.pkl**（训练产物，位于 ``<out_dir>/habitat_pipeline.pkl``）

- **内容**: joblib 序列化的已拟合 ``HabitatPipeline``（群体聚类模型、``PreprocessingState``、配置等）。
- **保存时自动瘦身**（``HabitatPipeline.save()`` 调用 ``prepare_pipeline_for_save``）:

  - 移除训练集 ``labels_`` （预测只需聚类中心/模型参数）
  - **不写入** ``mask_info_cache`` （写 NRRD 时从 ``data_dir`` 按需读取 mask，见 ``FeatureService.load_mask_info``）
  - 不写入 ``_train_checkpoint`` （断点仍在 ``checkpoint_dir`` 目录）

- **体积**:

  - 不再因 ``save_images: true`` 而膨胀：mask 体积与 pkl 解耦，``direct_pooling`` 大规模队列下 pkl 通常为 **几十～几百 MB**（主要取决于聚类特征维数与被试数相关的模型参数）
  - 旧版在 pkl 内嵌 ``mask_array`` 的 pkl 需 **重新 train 并 save** 后才会变小

**save_images**: 是否保存运行中生成的图像类输出（``*_habitats.nrrd`` 等）

- **类型**: 布尔值
- **默认值**: ``true``
- **说明**: 对应 ``HabitatAnalysisConfig.save_images`` 。为 ``true`` 时 train/predict 会写 habitat 标签图；mask 在写图时从 ``config.data_dir`` 加载，**不**写入 ``habitat_pipeline.pkl`` 。为 ``false`` 时仍可通过 ``habitats.csv`` 做下游分析，且不写 NRRD。

**verbose**: 是否输出较详细的运行日志

- **类型**: 布尔值
- **默认值**: ``true``

特征提取配置参数
------------

对应 ``habit.core.habitat_analysis.config_schemas.FeatureExtractionConfig``；CLI：``habit extract -c <yaml>``。示例：``config/feature_extraction/config_extract_features.yaml``。

**配置文件示例：**

.. code-block:: yaml

   params_file_of_non_habitat: ./parameter.yaml
   params_file_of_habitat: ./parameter_habitat.yaml

   raw_img_folder: ./preprocessed/processed_images
   habitats_map_folder: ./results/habitat
   out_dir: ./results/features

   n_processes: 3
   habitat_pattern: '*_habitats.nrrd'

   feature_types:
     - traditional
     - non_radiomics
     - whole_habitat
     - each_habitat
     - msi
     - ith_score

   n_habitats:

   debug: false

**params_file_of_non_habitat**: 从原始图像提取特征的参数文件

- **类型**: 字符串
- **必需**: 是
- **默认值**: 无（必填）
- **说明**: 使用 pyradiomics 提取传统影像组学特征的参数文件
- **示例**: ``./parameter.yaml``

**params_file_of_habitat**: 从生境图提取特征的参数文件

- **类型**: 字符串
- **必需**: 是
- **默认值**: 无（必填）
- **说明**: 使用 pyradiomics 从生境图中提取特征的参数文件
- **示例**: ``./parameter_habitat.yaml``

**raw_img_folder**: 原始图像根目录

- **类型**: 字符串
- **必需**: 是
- **默认值**: 无（必填）
- **说明**: 包含预处理后的图像
- **示例**: ``./preprocessed/processed_images``

**habitats_map_folder**: 生境图根目录

- **类型**: 字符串
- **必需**: 是
- **默认值**: 无（必填）
- **说明**: 包含生成的生境图
- **示例**: ``./results/habitat``

**out_dir**: 输出目录

- **类型**: 字符串
- **必需**: 是
- **默认值**: 无（必填）
- **说明**: 特征文件将保存在此目录
- **示例**: ``./results/features``

**debug**（``FeatureExtractionConfig``）

- **类型**: 布尔值
- **默认值**: ``false``

**n_processes**: 并行进程数

- **类型**: 整数
- **必需**: 否
- **默认值**: ``4`` （``FeatureExtractionConfig`` schema）
- **说明**: 用于并行处理的进程数
- **示例**: ``3``

**habitat_pattern**: 生境文件匹配模式

- **类型**: 字符串
- **必需**: 否
- **默认值**: ``'*_habitats.nrrd'``
- **说明**: 用于匹配生境图文件，支持通配符（``*``）
- **示例**: ``*_habitats.nrrd``

**feature_types**: 特征类型列表

- **类型**: 列表
- **必需**: 是
- **默认值**: 无（必填，至少一项）
- **说明**: 未在列表中的类型不会提取
- **可选值**: ``traditional``, ``non_radiomics``, ``whole_habitat``, ``each_habitat``, ``msi``, ``ith_score``
- **示例**: ``[traditional, non_radiomics, whole_habitat]``

**n_habitats**: 生境数量

- **类型**: 整数或 null
- **必需**: 否
- **默认值**: ``null`` （表示自动检测）
- **说明**: 可以手动指定生境数量
- **示例**: ``null``

机器学习配置参数
------------

对应 ``habit.core.machine_learning.config_schemas.MLConfig``；CLI：``habit model -c <yaml>`` （K-fold：``habit cv``）。train 示例 ``config/machine_learning/config_machine_learning.yaml``；predict 见 ``config/machine_learning/config_machine_learning_predict.yaml``。

**配置文件示例：**

.. code-block:: yaml

   run_mode: train
   input:
     - path: ./results/features/combined_features.csv
       name: training_data
       subject_id_col: Subject
       label_col: label
   output: ./results/ml/train
   random_state: 42

   split_method: stratified
   test_size: 0.3

   resampling:
     enabled: false
     method: random_over
     position: before_model
     ratio: 1.0

   normalization:
     method: z_score
     params: {}
   
   feature_selection_methods:
     - method: variance
       params:
         threshold: 0.0
     - method: correlation
       params:
         threshold: 0.9
   
   models:
     RandomForest:
       params:
         n_estimators: 100
         random_state: 42
     LogisticRegression:
       params:
         max_iter: 1000
   
   is_visualize: true
   is_save_model: true
   
   visualization:
     enabled: true
     plot_types: [roc, dca, calibration, pr, confusion, shap]
     dpi: 600
     format: pdf

**预测模式 YAML 示例**（``run_mode: predict``）：

.. code-block:: yaml

   run_mode: predict
   pipeline_path: ./ml_data/models/LogisticRegression_final_pipeline.pkl

   input:
     - path: ./ml_data/new_subjects.csv
       subject_id_col: subjID
       label_col: label   # evaluate: true 时需要

   output: ./ml_data/predictions

   evaluate: true
   output_label_col: predicted_label
   output_prob_col: predicted_probability

**run_mode**（YAML）

- **类型**: 字符串
- **默认值**: ``train``
- **可选值**: ``train`` 、``predict``
- **说明**: 训练与预测共用 ``MLConfig`` （``habit.core.machine_learning.config_schemas``）。``predict`` 时必须提供 ``pipeline_path``；``models`` 在预测模式下被忽略。

**mode / run_mode（CLI）**

- **命令**: ``habit model --mode <train|predict>``
- **说明**: ``--mode`` **覆盖** YAML 中的 ``run_mode`` （见 ``cmd_ml.run_ml``）。优先以命令行为准。

**pipeline_path**

- **类型**: 字符串
- **必需**: 当 ``run_mode`` 为 ``predict`` 时 **必填**
- **默认值**: ``null``
- **说明**: 指向已保存的 ``*_final_pipeline.pkl``。

**random_state**（``MLConfig`` 顶层）

- **类型**: 整数
- **必需**: 否
- **默认值**: ``42``
- **说明**: 划分、K-fold、重采样 fallback、未在 ``models.*.params`` 中写种子的模型等。

**input**: 输入数据配置

- **类型**: 列表
- **必需**: 是
- **说明**: 每个元素为 ``InputFileConfig``。预测模式仅使用 ``input[0].path`` 作为数据表。
- **子参数**:

  - ``path`` : 特征 CSV/Excel 路径（**必需**，无默认）。
  - ``name`` : 数据集名称；**默认** ``""``。
  - ``subject_id_col`` : 受试者 ID 列（**必需**，无默认）。
  - ``label_col`` : 标签列（**必需**，无默认）。
  - ``features`` : 仅使用这些列作为特征；**默认** ``null`` （自动推断数值特征列）。
  - ``features_from_log`` : 从日志/辅助文件解析特征列名；**默认** ``null``。
  - ``split_col`` : 自定义划分分组列；**默认** ``null``。
  - ``pred_col`` : 既有预测列名；**默认** ``null``。

**output**: 输出目录

- **类型**: 字符串
- **必需**: 是
- **默认值**: 无（必填）
- **说明**: 结果、模型与图表目录。CLI 会将日志写入该目录：训练默认 ``processing.log`` ，预测默认 ``prediction.log`` （见 ``habit.cli_commands.commands.cmd_ml``）。

**split_method**: 数据划分方法

- **类型**: 字符串
- **默认值**: ``stratified``
- **可选值**: ``random``, ``stratified``, ``custom``

**test_size**: 测试集比例

- **类型**: 浮点数
- **默认值**: ``0.3``
- **范围**: (0, 1)

**train_ids_file** / **test_ids_file**

- **类型**: 字符串路径（可选）
- **默认值**: ``null``
- **说明**: 当 ``split_method: custom`` 时使用，指向仅含受试者 ID 的文本文件（每行一个），用于固定训练/测试划分。

**n_splits** / **stratified**

- **类型**: 整数 / 布尔
- **默认值**: ``n_splits=5`` ，``stratified=true``
- **说明**: 供 ``habit cv`` （K 折）使用；字段定义见 ``MLConfig``。

**normalization**: 特征归一化设置

- ``method`` : 归一化方法

  - **类型**: 字符串
  - **默认值**: ``z_score``
  - **可选值**:

    - ``z_score`` : Z-Score 标准化 (StandardScaler)
    - ``min_max`` : 最小-最大归一化 (MinMaxScaler)
    - ``robust`` : 鲁棒缩放 (RobustScaler)
    - ``max_abs`` : 最大绝对值缩放 (MaxAbsScaler)
    - ``normalizer`` : L1/L2 归一化 (Normalizer)
    - ``quantile`` : 分位数转换 (QuantileTransformer)
    - ``power`` : 幂变换 (PowerTransformer)

- ``params`` : 方法特定参数

  - **类型**: 字典
  - **默认值**: ``{}``
  - **说明**: 根据选择的归一化方法传递不同的参数
  - **各方法支持的参数**:

    **z_score (StandardScaler)**:

      - ``with_mean`` (bool, 默认: ``true``): 是否在缩放前中心化数据
      - ``with_std`` (bool, 默认: ``true``): 是否缩放到单位方差

    **min_max (MinMaxScaler)**:

      - ``feature_range`` (list, 默认: ``[0, 1]``): 目标范围，如 ``[0, 1]`` 或 ``[-1, 1]``

    **robust (RobustScaler)**:

      - ``with_centering`` (bool, 默认: ``true``): 是否在缩放前中心化数据
      - ``with_scaling`` (bool, 默认: ``true``): 是否缩放到分位距
      - ``quantile_range`` (list, 默认: ``[25.0, 75.0]``): 用于计算缩放的分位数范围（IQR）

    **max_abs (MaxAbsScaler)**:

      - 无特殊参数（使用默认值即可）

    **quantile (QuantileTransformer)**:

      - ``n_quantiles`` (int, 默认: ``1000``): 分位数数量
      - ``output_distribution`` (str, 默认: ``uniform``): 输出分布，可选 ``uniform`` 或 ``normal``
      - ``subsample`` (int, 默认: ``10000``): 用于估计分位数的最大样本数

    **power (PowerTransformer)**:

      - ``method`` (str, 默认: ``yeo-johnson``): 变换方法，可选 ``yeo-johnson`` 或 ``box-cox``

  - **示例**:

    .. code-block:: yaml

       # Z-Score 标准化
       normalization:
         method: z_score
         params: {}
       
       # 最小-最大归一化到 [-1, 1]
       normalization:
         method: min_max
         params:
           feature_range: [-1, 1]
       
       # 鲁棒缩放（对异常值鲁棒）
       normalization:
         method: robust
         params:
           quantile_range: [25.0, 75.0]

**resampling**: 训练集重采样（``ResamplingConfig``）

- **YAML 键名**: ``resampling`` （推荐）。若仍使用旧键名 ``sampling`` ，加载时会自动映射为 ``resampling`` （见 ``MLConfig._migrate_legacy_sampling_key``）。
- **必需**: 否
- **默认值**: ``enabled: false`` （不对训练集做类别重采样）
- **说明**: 仅对 **训练** 数据重采样；验证/测试集不重采样。

- ``enabled`` : 是否启用

  - **类型**: 布尔值
  - **默认值**: ``false``

- ``method`` : 算法

  - **类型**: 字符串
  - **默认值**: ``random_over``
  - **可选值**: ``random_over`` | ``random_under`` | ``smote`` （SMOTE 需 ``imbalanced-learn``）

- ``position`` : 在流水线中的执行位置

  - **类型**: 字符串
  - **默认值**: ``before_model``
  - **可选值**: ``before_feature_selection`` 、``before_normalization`` 、``after_normalization`` 、``before_model``
  - **说明**: 决定重采样相对于特征选择/归一化/建模的顺序（见 ``habit.core.machine_learning`` 工作流）。

- ``ratio`` : 重采样比例，须 **> 0**

  - **默认值**: ``1.0``

- ``random_state`` : 随机种子（默认 ``null`` ，继承 ``MLConfig.random_state``；显式写入则覆盖顶层）

- **执行时机**: 训练流程在拟合模型前调用内部 ``_resample_training_data``；Holdout 与 K-Fold 共用该逻辑。

- **日志关键字**（便于确认已执行）: ``Sampling enabled`` 、``Sampling completed`` 等。

- **示例**:

  .. code-block:: yaml

     resampling:
       enabled: true
       method: random_over
       position: before_model
       ratio: 1.0
       random_state: 42

**feature_selection_methods**: 特征选择方法列表

- **类型**: 列表
- **默认值**: ``[]`` （空列表表示不做特征选择步骤）
- **说明**: 按顺序执行的特征选择步骤。每个方法都有特定的参数。
- **可选方法及其参数**:

  **variance (方差阈值)**:

    - ``threshold`` (float, 默认: ``0.0``): 方差阈值，低于此值的特征被移除
    - ``top_k`` (int, 可选): 选择方差最大的前 k 个特征（若指定则覆盖 threshold）
    - ``top_percent`` (float, 可选): 选择方差最大的前 x% 特征（0-100）
    - ``plot_variances`` (bool, 默认: ``true``): 是否绘制方差分布图

  **correlation (相关性过滤)**:

    - ``threshold`` (float, 默认: ``0.8``): 相关系数阈值，高于此值的特征对会移除其一
    - ``method`` (str, 默认: ``spearman``): 相关系数计算方法，可选 ``pearson``, ``spearman``, ``kendall``
    - ``visualize`` (bool, 默认: ``false``): 是否生成相关性热图

  **anova (方差分析)**:

    - ``p_threshold`` (float, 默认: ``0.05``): p 值阈值
    - ``n_features_to_select`` (int, 可选): 选择前 n 个特征（若指定则覆盖 p_threshold）
    - ``plot_importance`` (bool, 默认: ``true``): 是否绘制特征重要性图

  **chi2 (卡方检验)**:

    - ``p_threshold`` (float, 默认: ``0.05``): p 值阈值
    - ``n_features_to_select`` (int, 可选): 选择前 n 个特征
    - ``plot_importance`` (bool, 默认: ``true``): 是否绘制特征重要性图
    - **注意**: 仅适用于非负特征

  **lasso (Lasso 正则化)**:

    - ``cv`` (int, 默认: ``10``): 交叉验证折数
    - ``n_alphas`` (int, 默认: ``100``): alpha 参数的数量
    - ``alphas`` (list, 可选): 自定义 alpha 参数列表
    - ``random_state`` (int, 默认: ``42``): 随机种子
    - ``visualize`` (bool, 默认: ``false``): 是否生成系数路径图

  **rfecv (递归特征消除 + 交叉验证)**:

    - ``estimator`` (str, 默认: ``RandomForestClassifier``): 使用的估计器，可选:

      - 分类器: ``LogisticRegression``, ``RandomForestClassifier``, ``SVC``, ``GradientBoostingClassifier``, ``XGBClassifier``
      - 回归器: ``LinearRegression``, ``RandomForestRegressor``, ``SVR``, ``GradientBoostingRegressor``, ``XGBRegressor``

    - ``step`` (int, 默认: ``1``): 每次迭代移除的特征数
    - ``cv`` (int, 默认: ``5``): 交叉验证折数
    - ``scoring`` (str, 默认: ``roc_auc``): 评分指标
    - ``min_features_to_select`` (int, 默认: ``1``): 最少保留的特征数
    - ``n_jobs`` (int, 默认: ``-1``): 并行作业数（``-1`` 表示使用所有 CPU）
    - ``random_state`` (int, 可选): 随机种子

  **statistical_test (t 检验 / Mann-Whitney U，自动或强制)**:

    - ``p_threshold`` (float, 默认: ``0.05``)
    - ``n_features_to_select`` (int, 可选): 指定则覆盖 p 阈值
    - ``normality_test_threshold`` (float, 默认: ``0.05``): Shapiro-Wilk 正态性检验阈值
    - ``force_test`` (str, 可选): ``ttest`` 或 ``mannwhitney``；未设则按正态性自动选择
    - ``plot_importance`` (bool, 默认: ``true``)

  **icc (基于 ICC 结果 JSON 的稳定性筛选)**:

    - ``icc_results`` / ``icc_results_path`` (str): ``habit icc`` 输出的 JSON 路径
    - ``keys`` / ``groups`` (list): ICC 结果中要检查的组名列表
    - ``threshold`` (float, 默认: ``0.75``)
    - ``metric`` (str, 可选): 如 ``ICC3`` 、``ICC2`` 等

  **mrmr (最小冗余最大相关)**:

    - ``n_features`` (int, 默认: ``10``)
    - ``task_type`` (str, 默认: ``classification``): ``classification`` 或 ``regression``

  **vif (方差膨胀因子，去除共线性)**:

    - ``max_vif`` (float, 默认: ``10.0``)
    - ``visualize`` (bool, 默认: ``false``)

  **stepwise (Python 逐步 logistic 回归)**:

    - ``direction`` (str, 默认: ``backward``): ``forward`` 、``backward`` 、``both``
    - ``threshold_in`` / ``threshold_out`` (float, 默认: ``0.05``): ``criterion='pvalue'`` 时使用
    - ``criterion`` (str, 默认: ``aic``): ``aic`` 、``bic`` 或 ``pvalue``
    - ``verbose`` (bool, 默认: ``false``)

  **stepwise_r (R 语言逐步回归，需 R 环境)**:

    - 参数同 ``stepwise``；``method`` 键名为 ``stepwise_r``

  **univariate_logistic (单变量 logistic 回归)**:

    - ``alpha`` (float, 默认: ``0.05``): 显著性水平

- **示例**:

  .. code-block:: yaml

     # 方差阈值筛选
     feature_selection_methods:
       - method: variance
         params:
           threshold: 0.0
           plot_variances: true
     
     # 相关性过滤 + ANOVA
     feature_selection_methods:
       - method: correlation
         params:
           threshold: 0.9
           method: spearman
       - method: anova
         params:
           p_threshold: 0.05

**models**: 模型训练设置

- **类型**: 字典（模型名 → ``ModelConfig``）
- **默认值**: ``null`` （``run_mode: train`` 时必填且非空；``predict`` 时忽略）
- **说明**: 定义要训练的一个或多个模型。

- **支持的模型类型及常用参数**:

  **LogisticRegression (逻辑回归)**:

    - ``max_iter`` (int, 默认: ``100``): 最大迭代次数
    - ``C`` (float, 默认: ``1.0``): 正则化强度的倒数
    - ``penalty`` (str, 默认: ``l2``): 正则化类型，可选 ``l1``, ``l2``, ``elasticnet``
    - ``solver`` (str, 默认: ``lbfgs``): 优化算法
    - ``random_state`` (int): 随机种子

  **RandomForest (随机森林)**:

    - ``n_estimators`` (int, 默认: ``100``): 决策树数量
    - ``max_depth`` (int, 可选): 树的最大深度
    - ``min_samples_split`` (int, 默认: ``2``): 分裂节点所需的最小样本数
    - ``min_samples_leaf`` (int, 默认: ``1``): 叶子节点的最小样本数
    - ``max_features`` (str/int, 默认: ``sqrt``): 分裂时考虑的最大特征数
    - ``random_state`` (int): 随机种子

  **XGBoost (极端梯度提升)**:

    - ``n_estimators`` (int, 默认: ``100``): 提升轮数
    - ``max_depth`` (int, 默认: ``3``): 树的最大深度
    - ``learning_rate`` (float, 默认: ``0.1``): 学习率
    - ``subsample`` (float, 默认: ``1.0``): 样本采样比例
    - ``colsample_bytree`` (float, 默认: ``1.0``): 特征采样比例
    - ``random_state`` (int): 随机种子

  **SVM (支持向量机)**:

    - ``C`` (float, 默认: ``1.0``): 正则化参数
    - ``kernel`` (str, 默认: ``rbf``): 核函数，可选 ``linear``, ``poly``, ``rbf``, ``sigmoid``
    - ``gamma`` (str/float, 默认: ``scale``): 核系数
    - ``probability`` (bool, 默认: ``false``): 是否启用概率估计
    - ``random_state`` (int): 随机种子

  **KNN (K 近邻)**:

    - ``n_neighbors`` (int, 默认: ``5``): 邻居数量
    - ``weights`` (str, 默认: ``uniform``): 权重函数，可选 ``uniform``, ``distance``
    - ``metric`` (str, 默认: ``minkowski``): 距离度量

  **DecisionTree (决策树)**:

    - ``max_depth`` (int, 可选)
    - ``min_samples_split`` (int, 默认: ``2``)
    - ``min_samples_leaf`` (int, 默认: ``1``)
    - ``random_state`` (int)

  **MLP (多层感知机)**:

    - ``hidden_layer_sizes`` (tuple/list, 默认: `` (100,)``)
    - ``activation`` (str, 默认: ``relu``)
    - ``max_iter`` (int, 默认: ``200``)
    - ``random_state`` (int)

  **AdaBoost / GradientBoosting**:

    - ``n_estimators`` (int, 默认: ``100``)
    - ``learning_rate`` (float, 默认: ``1.0`` for AdaBoost, ``0.1`` for GradientBoosting)
    - ``random_state`` (int)

  **GaussianNB / MultinomialNB / BernoulliNB (朴素贝叶斯)**:

    - 多数情况下使用 sklearn 默认参数；``MultinomialNB`` 要求非负特征

  **AutoGluon (自动机器学习)**:

    - ``time_limit`` (int): 训练时间限制（秒）
    - ``presets`` (str, 默认: ``medium_quality``): 预设质量，可选 ``best_quality``, ``high_quality``, ``medium_quality``

- **示例**:

  .. code-block:: yaml

     # 训练多个模型
     models:
       LogisticRegression:
         params:
           max_iter: 1000
           C: 1.0
           random_state: 42
       
       RandomForest:
         params:
           n_estimators: 200
           max_depth: 10
           random_state: 42
       
       XGBoost:
         params:
           n_estimators: 100
           max_depth: 5
           learning_rate: 0.1
           random_state: 42

**is_visualize**: 是否启用可视化

- **类型**: 布尔值
- **默认值**: ``true``

**visualization**: 可视化详细设置（``VisualizationConfig``）

- ``enabled`` : **默认** ``true``
- ``plot_types`` : **默认** ``[roc, dca, calibration, pr, confusion, shap]`` ；可选值同列出的类型名
- ``dpi`` : **默认** ``600``
- ``format`` : **默认** ``pdf``

**is_save_model**: 是否保存训练好的流水线到 ``output`` （默认 ``true``）。

**预测模式专用字段**（``run_mode: predict`` 时）

- ``evaluate`` (bool, 默认 ``false``): 若数据含标签，是否在预测后计算指标。
- ``output_label_col`` (默认 ``predicted_label``): 输出表中的预测类别列名。
- ``output_prob_col`` (默认 ``predicted_probability``): 输出表中的概率列名。
- ``probability_class_index`` : 多分类时选取写入概率的一类索引（``None`` 表示保留全部或按实现约定）。
- ``binary_positive_class_index`` (默认 ``1``): 二分类正类在概率向量中的索引。

**模型对比配置（``habit compare`` ，``ModelComparisonConfig``）**

- ``output_dir`` (**必填**): 汇总输出目录。
- ``files_config`` (**列表**): 每个模型一个元素。

  - ``path`` (**必填**): 预测结果 CSV/Excel。
  - ``subject_id_col`` / ``label_col`` / ``prob_col`` (**必填**)。
  - ``pred_col`` / ``split_col`` (可选)。
  - ``model_name`` 或 ``name`` 或从 ``path`` 推断文件名 stem。

- ``merged_data`` : ``enabled`` 、``save_name`` （默认 ``combined_predictions.csv``）。
- ``split`` (内部 ``SplitConfig``): 是否在合并后做划分（默认 ``enabled: false``）。
- ``visualization`` : ``roc`` / ``dca`` / ``calibration`` / ``pr_curve`` 子块，各有 ``enabled`` 、``save_name`` 、``title`` 、``n_bins`` （校准用）等。
- ``delong_test`` : ``enabled`` 、``save_name`` （默认 ``delong_results.json``）。
- ``metrics`` : ``basic_metrics`` 、``youden_metrics`` 、``target_metrics`` （含 ``targets`` 字典，值为 (0,1) 内目标阈值）。

数据配置参数
------------

**配置文件示例：**

.. code-block:: yaml

   # 控制是否自动读取目录中的第一个文件
   auto_select_first_file: true

   images:
     subject1:
       T1: /path/to/subject1/T1/T1.nii.gz
       T2: /path/to/subject1/T2/T2.nii.gz
     subject2:
       T1: /path/to/subject2/T1/T1.nii.gz
       T2: /path/to/subject2/T2/T2.nii.gz

   masks:
     subject1:
       T1: /path/to/subject1/T1/mask_T1.nii.gz
     subject2:
       T1: /path/to/subject2/T1/mask_T1.nii.gz

**auto_select_first_file**: 是否自动读取目录中的第一个文件

- **类型**: 布尔值
- **默认值**: ``true``
- **说明**: 

  - ``true`` : 自动读取目录中的第一个文件（适用于已转换的 nii 文件等场景）。
  - ``false`` : 保持目录路径不变（适用于 dcm2nii 等需要整个文件夹的任务）。

**images**: 图像数据路径

- **类型**: 字典
- **必需**: 是
- **默认值**: 无（必填）
- **说明**: 嵌套字典，第一层是受试者 ID，第二层是图像类型（Key）。

**masks**: 掩码数据路径

- **类型**: 字典
- **必需**: 否
- **默认值**: 省略表示无掩码块
- **说明**: 结构同 ``images``。通常用于指定 ROI。

ICC 分析配置（``habit icc``）
------------------------------

对应 ``habit.core.machine_learning.feature_selectors.icc.config.ICCConfig``。示例：``config/auxiliary/config_icc_demo.yaml``。

**input**（必填）

- ``type`` : ``files`` 或 ``directories``
- ``file_groups`` (``type: files``): 二维列表，每组为一次 ICC 计算的重复测量文件路径；也接受扁平列表（每项视为单文件组）
- ``dir_list`` (``type: directories``): 目录列表，从各目录收集特征文件

**output**（必填）

- ``path`` : 结果 JSON 输出路径

**顶层可选字段**

- ``metrics`` : ICC 指标列表，如 ``icc1`` 、``icc2`` 、``icc3`` 、``icc1k`` 、``icc2k`` 、``icc3k`` 、``multi_icc`` 、``cohen_kappa`` 、``fleiss_kappa`` 、``krippendorff`` 等；默认示例为 ``[icc3]``
- ``selected_features`` : 限定参与 ICC 的特征列；``null`` 表示全部
- ``full_results`` (bool, 默认 ``false``): 是否输出完整明细
- ``processes`` (int, 可选): 并行进程数
- ``debug`` (bool, 默认 ``false``)

Test-Retest 配置（``habit retest``）
------------------------------------

对应 ``habit.core.machine_learning.config_schemas.TestRetestConfig``。示例：``config/auxiliary/config_test_retest.yaml``。用户指南：:doc:`app_habitat_test_retest_zh`。

**必填字段**

- ``test_habitat_table`` : 测试扫描生境特征表（CSV/Excel）
- ``retest_habitat_table`` : 重测扫描生境特征表
- ``input_dir`` : 重测组 NRRD 生境图目录（用于映射/重对齐）
- ``out_dir`` : 分析结果输出目录

**可选字段**

- ``features`` : 参与相似度计算的特征列；``null`` 表示全部
- ``similarity_method`` (默认 ``pearson``): ``pearson`` 、``spearman`` 、``kendall`` 、``euclidean`` 、``cosine`` 、``manhattan`` 、``chebyshev``
- ``output_dir`` : 重映射 NRRD 等中间结果目录（示例 YAML 中与 ``out_dir`` 并列使用）
- ``processes`` (默认 ``4``)
- ``debug`` (默认 ``false``)

传统影像组学 CLI 配置（``habit radiomics``）
------------------------------------------

对应 ``habit.core.habitat_analysis.config_schemas.RadiomicsConfig``。示例：``config/radiomics/config_traditional_radiomics.yaml``。

**paths**（必填）

- ``params_file`` : PyRadiomics 参数 YAML（`PyRadiomics 文档  <https://pyradiomics.readthedocs.io/>`_）
- ``images_folder`` : 根目录下含 ``images/`` 与 ``masks/`` 子文件夹
- ``out_dir`` : 特征输出目录

**processing**

- ``n_processes`` (默认 2)
- ``save_every_n_files`` (默认 5)：每处理 N 个文件保存中间结果
- ``process_image_types`` ：限制处理的序列/类型名列表，``null`` 表示全部
- ``target_labels`` ：从掩膜中抽取的标签列表（默认 ``[1]``），用于二值前景

**export**

- ``export_by_image_type`` 、``export_combined`` 、``export_format`` (``csv`` \| ``json`` \| ``pickle``)、``add_timestamp``

**logging**

- ``level`` (DEBUG/INFO/…)、``console_output`` 、``file_output``

**向后兼容顶层字段**（deprecated，等价于嵌套）：``params_file`` 、``images_folder`` 、``out_dir`` 、``n_processes``。

仓库配置模板索引
----------------

``config/`` 目录按功能分子文件夹，可直接复制修改：

.. list-table::
   :header-rows: 1
   :widths: 28 52

   * - 路径
     - 用途
   * - ``config/preprocessing/``
     - 图像预处理与 ``files_preprocessing.yaml`` 被试列表
   * - ``config/dicom_sort/``
     - DICOM 仅整理（``sort-dicom``）
   * - ``config/habitat/``
     - 生境 train/predict（two_step / one_step / direct_pooling）及 ``file_habitat.yaml``
   * - ``config/feature_extraction/``
     - ``habit extract`` 生境特征提取
   * - ``config/radiomics/``
     - PyRadiomics 参数与 ``habit radiomics`` 顶层配置
   * - ``config/machine_learning/``
     - 标准训练/预测、K-fold、临床/组学示例
   * - ``config/model_comparison/``
     - 多模型 ROC/DCA/DeLong 对比
   * - ``config/auxiliary/``
     - ICC、Test-Retest 等辅助分析

配置文件验证
------------

HABIT 提供了配置文件验证机制，确保参数的正确性。

**验证规则：**

1. **必需参数检查**: 检查所有必需参数是否提供
2. **类型检查**: 检查参数类型是否正确
3. **范围检查**: 检查参数值是否在有效范围内
4. **依赖检查**: 检查参数依赖是否满足

**验证示例：**

.. code-block:: python

   from habit.core.common.configs.loader import load_config

   # 加载配置并验证
   config = load_config('./config.yaml')

   # 如果配置有误，会抛出异常
   # ValueError: Missing required parameter: data_dir

常见问题
--------

**Q1: 如何创建配置文件？**

A: 可以通过以下方式创建：

1. 复制示例配置文件并修改
2. 参考本文档创建新的配置文件
3. 使用配置文件生成工具（如果有）

**Q2: 如何调试配置文件？**

A: 可以使用以下方法：

1. 使用 `debug` 模式启用详细日志
2. 检查配置文件语法
3. 逐步添加参数，定位问题
4. 查看错误信息
