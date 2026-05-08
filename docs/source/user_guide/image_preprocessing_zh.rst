图像预处理
==========

本节介绍如何使用 HABIT 进行医学图像预处理。

文档范围（必读）
----------------

- **目标**：说明 HABIT 预处理流水线的 **配置键、数据流、调用链**\ （``BatchProcessor`` → ``PreprocessorFactory``），并给出 **ANTsPy / SimpleITK / dcm2niix** 的权威外链。配准、重采样等算法的 **完整理论与参数全集**\ 以上游文档为准；HABIT 不复制一整本 ANTs 手册。
- **其它模块**\ （生境聚类、机器学习、``habit radiomics`` 等）各有独立章节（见根目录 ``index.rst`` 与 :doc:`../configuration_zh`）。若某功能仅在 API 或 ``config_schemas`` 中出现而用户文档未写全，以源码与 Schema 为准，并欢迎提 issue 补页。
- **版本**：ANTsPy、SimpleITK 的小版本差异会导致 ``ants.registration`` 支持的参数名略不同；部署后请在本地用 ``help(ants.registration)`` 与 `ANTsPy 文档`_ 核对。

.. _`ANTsPy 文档`: https://antspy.readthedocs.io/en/stable/registration.html

概述
----

图像预处理是生境分析的第一步，目的是提高图像质量，统一图像格式和空间分辨率，为后续的生境分割和特征提取做好准备。

**依赖（按步骤）**

- **DICOM 整理（``sort_dicom``）**：需安装 ``dcm2niix``。该步骤使用 ``dcm2niix -r y`` 只整理/重命名 DICOM 文件，不转换为 NIfTI。
- **DICOM 转 NIfTI（``dcm2nii``）**：需安装 ``dcm2niix``，并在配置中指定可执行文件路径或将可执行文件加入 ``PATH``（亦可省略 ``dcm2niix_path``，此时使用命令名 ``dcm2niix``）。
- **配准（``registration``）**：默认 ``backend: ants`` 时需 **ANTsPy**（及底层 ANTs）；若设置 ``backend: simpleitk`` 则仅用 **SimpleITK**，可不装 ANTsPy。仅使用重采样 / 标准化等步骤、不配准时也可不装 ANTsPy。
- **其余步骤**：基于 SimpleITK 等库（随 HABIT 依赖安装）。

HABIT 提供的预处理方法（配置中 ``Preprocessing`` 下的 **键名** 须与下列注册名一致）包括：

- **sort_dicom**：DICOM 文件整理/重命名（``dcm2niix -r y``，不转换）
- **dcm2nii**：DICOM → NIfTI（``dcm2niix``）
- **n4_correction**：N4 偏置场校正（MRI）
- **resample**：重采样至目标体素间距
- **registration**：多时相配准到参考图像（``backend: ants`` 用 ANTsPy，``backend: simpleitk`` 用 SimpleITK）
- **zscore_normalization**：逐序列 Z-score 强度标准化
- **histogram_standardization**：Nyúl 直方图标准化（百分位 landmark）
- **adaptive_histogram_equalization**：对比度受限自适应直方图均衡（CLAHE）

流水线行为
----------

1. **隐式加载**：每个被试在执行大多数 ``Preprocessing`` 步骤前，会先用内部步骤 ``load_image``（无需写入 YAML）将路径替换为内存中的 SimpleITK 图像。例外：如果第一步是 ``sort_dicom``，流水线会先保留 DICOM 目录路径，让 ``sort_dicom`` 完成文件整理。
2. **步骤顺序**：``Preprocessing`` 下各子块在 YAML 中的 **书写顺序** 即为执行顺序；调整顺序会改变结果（例如应先重采样再配准还是先配准，需按数据与课题决定）。
3. **并行**：顶层 ``processes`` 为并行被试进程数；实现中会取 ``min(processes, CPU 逻辑核心数 - 2)``，且至少为 **1**\ （避免占满整机）。
4. **重复性字段**：``random_state`` 在 ``PreprocessingConfig`` 中保留默认值，但**当前预处理流水线不会读取**；若需完全可复现，请关注各步骤内部算法与多进程顺序。
5. **掩码**：文件清单 YAML 中可为每个模态提供掩码；无掩码时仍会处理该被试，但会记录警告（详见下文「无掩码」说明）。

CLI 使用方法
------------

**基本语法：**

.. code-block:: bash

   habit preprocess --config <config_file>

**参数说明：**

- ``--config``, ``-c``: 配置文件路径（必需）

**使用示例：**

.. code-block:: bash

   habit preprocess --config ./demo_data/config_preprocessing.yaml

**输出与日志：**

- 最终结果写入 ``out_dir`` 下的 ``processed_images`` 目录（见下文「输出结构」）。
- 使用 CLI 时，日志文件 ``processing.log`` 写在 ``out_dir`` 下。若直接调用 ``BatchProcessor`` 且未单独配置全局日志，默认亦如此。

Python API 使用方法
-------------------

**推荐路径（与 CLI 一致）**：通过 ``PreprocessingConfigurator`` 创建 ``BatchProcessor``。配置对象 **必须** 由 ``PreprocessingConfig.from_file(path)`` 加载，以便内部保存 ``config_file`` 供处理器读取。

.. code-block:: python

   from habit.core.preprocessing.configurator import PreprocessingConfigurator
   from habit.core.preprocessing.config_schemas import PreprocessingConfig

   config = PreprocessingConfig.from_file("./demo_data/config_preprocessing.yaml")
   configurator = PreprocessingConfigurator(config=config)
   processor = configurator.create_batch_processor()
   processor.run()

**直接使用 BatchProcessor**\ （自行传入 YAML 路径）：

.. code-block:: python

   from habit.core.preprocessing.image_processor_pipeline import BatchProcessor

   processor = BatchProcessor(config_path="./demo_data/config_preprocessing.yaml")
   processor.run()

**自定义日志**：若在外部调用 ``setup_logger``，请将日志文件名与 ``out_dir`` 约定清楚；CLI 使用 ``cli.preprocessing`` 与 ``processing.log``。

YAML 配置详解
--------------

**顶层字段** （``PreprocessingConfig``）

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - 字段
     - 说明
   * - ``data_dir``
     - 指向 **目录** 或 **文件清单 YAML**\ （参见 :doc:`../data_structure_zh`）。目录时可通过 ``keyword_of_raw_folder`` 等在 ``get_image_and_mask_paths`` 中解析（默认行为见 ``habit.utils.io_utils``）。
   * - ``out_dir``
     - 预处理输出根目录；其下将创建 ``processed_images`` 等。
   * - ``Preprocessing``
     - 各预处理步骤的字典；**键名** 必须为已注册步骤名（见上表）。
   * - ``save_options``
     - ``save_intermediate``：是否落盘中途结果；``intermediate_steps``：仅保存列出的步骤名；**空列表** 表示 **每一步** 都保存中间结果（若 ``save_intermediate`` 为 true）。
   * - ``processes``
     - 并行被试进程数；``>= 1``；实际上限为 ``min(配置值, CPU核心数-2)``，至少 1。
   * - ``auto_select_first_file``
     - 默认 ``true``：当路径指向 **文件夹** 且内含多个文件时，自动选取第一个文件；设为 ``false`` 则保留目录行为（详见 ``get_image_and_mask_paths``）。
   * - ``random_state``
     - 整数，默认 ``42``；当前预处理流水线 **未读取**，保留用于配置扩展或与其它工具对齐。

``Preprocessing`` **公共约定**

- 每个步骤块 **必须** 包含非空列表 ``images``：模态键名，须与 ``files_preprocessing.yaml`` 等清单中一致。
- 掩码在内存中的键名为 ``mask_<模态>``（例如模态 ``delay2`` 对应 ``mask_delay2``）。配准在 ``use_mask: true`` 时使用固定图与浮动图对应的掩码键。

**完整示例（节选，注册名即 YAML 键名）：**

.. code-block:: yaml

   data_dir: ./files_preprocessing.yaml
   out_dir: ./preprocessed
   auto_select_first_file: true

   Preprocessing:
     sort_dicom:
       images: [dicom]
       dcm2niix_path: ./dcm2niix.exe
       filename_format: "%n_%g_%x/%s_%d/%r_%o.dcm"
       output_dir: ./sorted_dicom

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
       # backend: simpleitk   # optional: ants | simpleitk; see registration section for simpleitk-only keys
       # number_of_iterations: 200
       # shrink_factors_per_level: [4, 2, 1]

     histogram_standardization:
       images: [delay2, delay3, delay5]
       percentiles: [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
       target_min: 0.0
       target_max: 100.0

     zscore_normalization:
       images: [delay2, delay3, delay5]
       only_inmask: false
       clip_values: [-3, 3]

   save_options:
     save_intermediate: true
     intermediate_steps: [dcm2nii, n4_correction, resample]

   processes: 2

**中间结果与 ``sort_dicom`` / ``dcm2nii``**

- ``sort_dicom`` 使用 ``dcm2niix -r y`` 直接把 DICOM 文件整理到目标目录；它不产生 SimpleITK 图像，也不转换 NIfTI。
- 对除 ``sort_dicom`` / ``dcm2nii`` 外的步骤，中间结果常通过写入 ``<step>_<序号>/images|masks/...`` 保存。
- ``dcm2nii`` 自行将 NIfTI 写到当步配置的 ``output_dirs``；开启中间保存时，流水线 **不会** 再对 ``dcm2nii`` 调用通用落盘逻辑（避免重复），详见 ``BatchProcessor._process_single_subject``。

预处理方法详解
----------------

**sort_dicom（DicomSorter）**

``sort_dicom`` 用于整理原始 DICOM 文件，等价于调用 ``dcm2niix -r y``。默认文件名模式为 ``%n_%g_%x/%s_%d/%r_%o.dcm``，即按「患者名/Accession/Study ID → 序列号/序列描述 → instance/media object」组织。常用参数：``dcm2niix_path``、``filename_format``、``output_dir``、``directory_depth``、``adjacent_dicoms``、``ignore_derived``、``write_behavior``、``progress``、``verbose``、``extra_args``。如果 ``filename_format`` 中包含 ``/``，dcm2niix 会创建对应子目录。

可自定义的 ``-f`` token 包括：``%n`` 患者名、``%g`` accession number、``%x`` study ID、``%s`` series number、``%d`` description、``%r`` instance number、``%o`` mediaObjectInstanceUID 等；完整列表见 ``config_templates/config_image_preprocessing_sort_dicom_annotated.yaml`` 或 dcm2niix ``-h``。

示例：

.. code-block:: yaml

   data_dir: ./demo_data/dicom
   out_dir: ./demo_data/sorteddcm
   auto_select_first_file: false

   Preprocessing:
     sort_dicom:
       images: [dicom]
       dcm2niix_path: ./dcm2niix.exe
       filename_format: "%n_%g_%x/%s_%d/%r_%o.dcm"
       output_dir: ./demo_data/sorteddcm

**dcm2nii（Dcm2niixConverter）**

除 ``images`` 外常用参数：``dcm2niix_path``（可选）、``compress``、``anonymize``、``filename_format``、``adjacent_dicoms``、``ignore_derived``、``crop_images``、``generate_json``、``verbose``、``batch_mode``、``merge_slices``、``single_file_mode`` 等。入口路径可为 DICOM 目录；须保证 ``dcm2niix`` 可用。上游工具说明与命令行标志见 `dcm2niix 项目 <https://github.com/rordenlab/dcm2niix>`__。

**n4_correction**

常用字段包括 num_fitting_levels（默认 4）、num_iterations（默认每层 50 次，长度与层数一致）、convergence_threshold、shrink_factor（下采样加速），以及可选的 mask_keys（掩膜参与校正）。

算法细节（SimpleITK Doxygen）：见 `N4BiasFieldCorrectionImageFilter <https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1N4BiasFieldCorrectionImageFilter.html>`__。

**resample（ResamplePreprocessor，SimpleITK）**

HABIT 的重采样**不**依赖 ANTs；实现位于 ``habit/core/preprocessing/resample.py``，核心使用 SimpleITK 的 ``ResampleImageFilter``。在保持图像原点与方向矩阵不变的前提下，按 ``target_spacing`` 所给体素间距（物理单位，一般为 mm）重建网格。

**官方参考（重采样 / 物理空间）**

- `SimpleITK 基本概念（间距、原点、方向） <https://simpleitk.readthedocs.io/en/master/FundamentalConcepts.html>`__
- `ResampleImageFilter（Doxygen） <https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ResampleImageFilter.html>`__

**HABIT 中的计算要点**

- 体素维度的尺寸换算遵循 ``_resample_image`` 中的实现：由原始与目标 spacing 推导新网格大小。
- 参考图 ``reference_image``：尺寸为 ``new_size``，间距为 ``target_spacing``，原点与方向取自输入图像；``ResampleImageFilter.SetReferenceImage`` 后 ``Execute``。

**配置参数**

- ``target_spacing``：长度为 3 的浮点序列，依次对应 x、y、z 方向体素间距（单位与输入 NIfTI 一致；记法可与 sx、sy、sz 三轴 spacing 对照）。
- ``img_mode``：图像通道的插值器名称，映射到 SimpleITK 枚举（节选，完整见源码 ``interp_map``）：

  .. list-table::
     :header-rows: 1
     :widths: 28 72

     * - ``img_mode`` 字符串
       - SimpleITK 插值器
     * - ``nearest``
       - ``sitkNearestNeighbor``
     * - ``linear`` / ``bilinear``
       - ``sitkLinear``
     * - ``bspline`` / ``bicubic``
       - ``sitkBSpline``
     * - ``gaussian``
       - ``sitkGaussian``
     * - ``lanczos`` / ``hamming`` / ``cosine`` / ``welch`` / ``blackman``
       - 各类 ``WindowedSinc``

- **掩码**：凡键名 ``mask_<modality>`` 使用 **最近邻**\ （``sitkNearestNeighbor``），避免标签插值产生非整数类号。
- **未接入滤波的键**：``padding_mode``、``align_corners`` 在构造函数中可解析，但当前 ``_resample_image`` 未将其传入 ``ResampleImageFilter``；若在 YAML 中填写，不会产生效果（保留作未来扩展）。需要边界填充或特殊对齐时，请改源码或自定义预处理器。

**registration（RegistrationPreprocessor）**

``RegistrationPreprocessor`` 支持 ``backend: ants``（默认）与 ``backend: simpleitk``。二者在 YAML 中共享 ``images``、``fixed_image``、``type_of_transform``、``metric``、``optimizer``、``use_mask``、``replace_by_fixed_image_mask`` 等入口键；**高级调参** 因后端而异（见下文）。

**公共约定**

- **浮动序列**：``images`` 中除 ``fixed_image`` 以外的每个模态都会依次配准。请勿使用 ``moving_images``，实现不会读取该键。
- **掩膜**：``use_mask`` 为真且存在 ``mask_<modality>`` 时参与度量；``replace_by_fixed_image_mask`` 为真时，配准后用固定侧掩膜替换浮动侧掩膜键。否则掩膜经变换对齐到固定网格：ANTS 路径用 ``ants.apply_transforms``，SimpleITK 路径用 ``ResampleImageFilter``。

**backend: ants（默认）**

HABIT 调用 **ANTsPy** ``ants.registration``（底层对应 ANTs ``antsRegistration``）。**与** SimpleITK ``ImageRegistrationMethod`` **不是**同一套 API；度量、优化器、多分辨率等请以 **ANTs/ANTsPy** 文档为准。

**官方参考（ANTS 路径）**

- `ANTsPy — Registration（教程型说明） <https://antspy.readthedocs.io/en/stable/registration.html>`__
- `ants.registration API（参数列表） <https://antspy.readthedocs.io/en/stable/api/ants.registration.html>`__
- `ANTs 主仓库（构建与引用） <https://github.com/ANTsX/ANTs>`__

**HABIT 调用关系（ANTS）**

- 构造 ``reg_params``（含 metric、optimizer 以及 YAML 其余关键字）。若 ``use_mask`` 为真且存在掩膜，再附加 ``mask`` 与 ``moving_mask``。
- 最终调用 ``ants.registration``，并传入 fixed、moving、type_of_transform 与 reg_params；源码见 ``habit/core/preprocessing/registration.py``。
- 下列 **仅 SimpleITK 使用的调优键**\ （见后文「backend: simpleitk」）在 ``backend: ants`` 时会被预先 **剥除**，不会传入 ANTs。

**type_of_transform（ANTS；写入 YAML 的字符串与 ANTs 一致）**

下列与 ``registration.py`` docstring 一致，并随 ANTsPy 版本可能增删；完整说明与默认值以 ``ants.registration`` 的文档为准。

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 取值
     - 含义
   * - ``Rigid``
     - 刚性变换
   * - ``Affine``
     - 仿射变换
   * - ``SyN``
     - 对称归一化（可变形）
   * - ``SyNRA``
     - SyN + 刚性 + 仿射
   * - ``SyNOnly``
     - 不含初始刚性/仿射的 SyN
   * - ``TRSAA``
     - 平移 + 旋转 + 缩放 + 仿射
   * - ``Elastic``
     - 弹性变换
   * - ``SyNCC``
     - 带互相关度量的 SyN
   * - ``SyNabp``
     - 带互信息度量的 SyN
   * - ``SyNBold``
     - 针对 BOLD 的 SyN
   * - ``SyNBoldAff``
     - BOLD 的 SyN + 仿射
   * - ``SyNAggro``
     - 激进优化策略的 SyN
   * - ``TVMSQ``
     - 时变差分 + 均方度量

**HABIT 显式包装的参数（ANTS）**

- ``metric`` 默认为 MI，写入 ``reg_params``；常见取值还有 CC、MeanSquares、Demons 等（见 RegistrationPreprocessor 文档）。
- ``optimizer`` 写入 ``reg_params``；例如 gradient_descent、lbfgsb、amoeba（以 ANTsPy 版本为准）。

**通过 YAML 透传到 ANTs 的高级参数**

在 registration 配置段中，除 images、fixed_image、backend 以及 **仅 SimpleITK 的调优键** 外，凡 **未** 被 HABIT 单独解析的参数名（若与 ANTs 一致）都会并入 ``ants.registration``。以下为常用示例（是否可用取决于 ANTsPy 版本）：

- 迭代与多分辨率：如 ``reg_iterations``、``aff_iterations``、``aff_shrink_factors``、``aff_smoothing_sigmas``、``grad_step``、``flow_sigma``、``total_sigma`` 等。
- 度量细分：如 ``aff_metric``、``syn_metric`` 等与阶段相关的度量选项。
- 初始变换：如 ``initial_transform``。

**backend: simpleitk**

- **依赖**：仅需 **SimpleITK**（与不配准的步骤一致），**不**\ 导入 ANTsPy。
- **实现概要**：使用 ``sitk.ImageRegistrationMethod`` 与多级分辨率参数 shrink_factors_per_level、smoothing_sigmas_per_level；输出 ``sitk.Transform`` 写入临时 tfm 文件；``data`` 中的变换文件列表字段命名方式与 ANTs 路径一致（键名以 ``_transform_files`` 结尾）；掩膜重采样用 ``sitk.ResampleImageFilter``。
- **type_of_transform 映射**\ （兼容 ANTs **名称**，但数值行为不同：**可变形路径为 BSpline 近似，不等价于 ANTs SyN**）：

  .. list-table::
     :header-rows: 1
     :widths: 28 72

     * - YAML 取值（示例）
       - SimpleITK 内部族
     * - ``Rigid``
       - 3D：``VersorRigid3DTransform``；2D：``Euler2DTransform``
     * - ``Affine``、``TRSAA``
       - ``AffineTransform``
     * - ``BSpline``
       - ``BSplineTransform``（由 ``BSplineTransformInitializer`` 初始化）
     * - ``SyN``、``SyNRA``、``SyNOnly`` 及 docstring 中列出的其它 **可变形型** ANTs 名、或以 ``Syn`` 开头的常见写法
       - **近似** 为 BSpline 可变形（非 ANTs 矢量场 SyN）

- 不支持的字符串会抛出 ``ValueError``；需 **刚性/仿射** 时优先使用 ``Rigid`` 或 ``Affine``。
- **metric（SimpleITK）**：``MI`` → Mattes 互信息；``CC`` → ``SetMetricAsCorrelation``；``MeanSquares`` / ``Mean_Squares`` → 均方；其它值 **回退** 为 Mattes MI 并记录警告。
- **optimizer（SimpleITK）**：字符串 **包含** ``lbfgs``（不区分大小写）时使用 ``LBFGSB``；否则使用梯度下降。迭代上限主要由下表 ``number_of_iterations`` 控制。

**仅 simpleitk 后端读取的 YAML 键** （与同名的 ANTs 迭代键 **无关**；这些键在 ``backend: ants`` 时会被 **剥离**，不传入 ``ants.registration``）：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 键
     - 含义与默认
   * - ``number_of_histogram_bins``
     - Mattes MI 直方图箱数；默认 ``50``
   * - ``metric_sampling_percentage``
     - 度量随机采样体素占比；默认 ``0.01``
   * - ``shrink_factors_per_level``
     - 多分辨率各级 **shrink** 因子；默认 ``[4, 2, 1]``
   * - ``smoothing_sigmas_per_level``
     - 各级高斯平滑 σ（**物理单位**，且 ``SetSmoothingSigmasAreSpecifiedInPhysicalUnits(True)``）；默认约 ``[2.1, 1.0, 0.0]``
   * - ``learning_rate``
     - 梯度下降学习率；默认 ``1.0``
   * - ``number_of_iterations``
     - 优化迭代次数；默认 ``100``
   * - ``bspline_mesh_size``
     - 仅 **BSpline** 族：控制点网格尺寸；可为单个 ``int``（各维相同）或长度为维度的 ``list``；默认每维 ``8``
   * - ``bspline_order``
     - 仅 BSpline：B 样条阶数；默认 ``3``

实现中键名常量见源码 ``habit/core/preprocessing/registration.py`` 中的 ``_SITK_OPTION_KEYS``。

**其它**：``replace_by_fixed_image_mask``、元数据字段 ``backend`` 等详见 ``registration.py``。

**zscore_normalization**

``only_inmask`` 与单个 ``mask_key``：启用仅在掩膜内统计时，``mask_key`` 须指向 ``data`` 字典中已存在的键（例如与某模态对齐的掩膜键名）；多模态时常用共享掩膜键。``clip_values``：可选，归一化后裁剪到 ``(low, high)``。

**histogram_standardization**

Nyúl 方法：``percentiles`` 分位点、``target_min`` / ``target_max`` 映射目标范围、可选 ``mask_key`` 限定直方图统计体素。

**adaptive_histogram_equalization**

``alpha``、``beta``（[0,1]）与 ``radius`` 传入 SimpleITK ``AdaptiveHistogramEqualizationImageFilter``；行为与参数解释以上游为准：`AdaptiveHistogramEqualizationImageFilter <https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1AdaptiveHistogramEqualizationImageFilter.html>`__。

自定义预处理器
--------------

1. 使用 ``@PreprocessorFactory.register("步骤名")`` 注册类，且须在包导入时执行注册（例如从 ``habit.core.preprocessing`` 子模块导入该模块）。
2. 在 YAML 的 ``Preprocessing`` 下使用 **与注册名相同的键**，并设置 ``images`` 及自定义参数。
3. 模板可参考 ``habit/core/preprocessing/custom_preprocessor_template.py``（内置示例注册名 ``custom_preprocessor``）。

输出结构
--------

最终输出目录（默认）：

.. code-block:: text

   <out_dir>/
   ├── processing.log              # CLI 默认日志
   ├── dcm2nii_01/                 # 可选中间：步骤名 + 两位序号
   ├── ...
   └── processed_images/            # 每被试最终 NIfTI
       ├── images/<subject>/<modality>/<modality>.nii.gz
       └── masks/<subject>/<modality>/mask_<modality>.nii.gz

**无掩膜被试**：若清单中无任何 ``mask_*``，流水线仍会保留该被试并打印警告；下游若依赖掩膜，请提前在数据中提供掩膜或从流程中排除该被试。

常见问题
--------

**Q：配准很慢或报错？**

- **ANTS 路径**：安装 ANTsPy 与 ANTs；尝试 ``Rigid`` / ``Affine``；减少 ``processes`` 以免内存占满。
- **若 ANTsPy 频繁报错**：在 ``registration`` 下设置 ``backend: simpleitk``，仅需 SimpleITK；可变形名在 SimpleITK 下为 **BSpline 近似**，刚性/仿射需求优先写 ``Rigid`` 或 ``Affine``。调参键见本节「backend: simpleitk」表格。

**Q：Z-score 在 only_inmask 下结果异常？**

- 检查 ``mask_key`` 是否与装入 ``data`` 的键一致（常为 ``mask_<modality>`` 或课题定义的共用掩膜键）。

**Q：中间结果占磁盘？**

- 设 ``save_intermediate: false``；或缩小 ``intermediate_steps`` 列表。

下一步
-------

图像预处理完成后，您可以：

- :doc:`habitat_segmentation_zh`: 进行生境分割
- :doc:`habitat_feature_extraction_zh`: 提取生境特征
- :doc:`machine_learning_modeling_zh`: 进行机器学习建模
