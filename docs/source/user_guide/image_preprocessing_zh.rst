图像预处理
========

本节介绍如何使用 HABIT 进行医学图像预处理。

文档范围（必读）
~~~~~~~~~~~~

- **目标**：说明 HABIT 预处理流水线的 **配置键、数据流、调用链**（``BatchProcessor`` → ``PreprocessorFactory``），并给出 **ANTsPy / SimpleITK / dcm2niix** 的权威外链。配准、重采样等算法的**完整理论与参数全集**以上游文档为准；HABIT 不复制一整本 ANTs 手册。
- **其它模块**（生境聚类、机器学习、`habit radiomics` 等）各有独立章节（见根目录 ``index.rst`` 与 :doc:`../configuration_zh`）。若某功能仅在 API 或 ``config_schemas`` 中出现而用户文档未写全，以源码与 Schema 为准，并欢迎提 issue 补页。
- **版本**：ANTsPy、SimpleITK 的小版本差异会导致 ``ants.registration`` 支持的参数名略不同；部署后请在本地用 ``help(ants.registration)`` 与 `ANTsPy 文档`_ 核对。

.. _`ANTsPy 文档`: https://antspy.readthedocs.io/en/stable/registration.html

概述
----

图像预处理是生境分析的第一步，目的是提高图像质量，统一图像格式和空间分辨率，为后续的生境分割和特征提取做好准备。

**依赖（按步骤）**

- **DICOM 转 NIfTI（``dcm2nii``）**：需安装 ``dcm2niix``，并在配置中指定可执行文件路径或将可执行文件加入 ``PATH``（亦可省略 ``dcm2niix_path``，此时使用命令名 ``dcm2niix``）。
- **配准（``registration``）**：需安装 **ANTsPy**（及其底层 ANTs）；仅使用重采样 / 标准化等步骤时可不装。
- **其余步骤**：基于 SimpleITK 等库（随 HABIT 依赖安装）。

HABIT 提供的预处理方法（配置中 ``Preprocessing`` 下的 **键名** 须与下列注册名一致）包括：

- **dcm2nii**：DICOM → NIfTI（``dcm2niix``）
- **n4_correction**：N4 偏置场校正（MRI）
- **resample**：重采样至目标体素间距
- **registration**：多时相配准到参考图像（ANTs）
- **zscore_normalization**：逐序列 Z-score 强度标准化
- **histogram_standardization**：Nyúl 直方图标准化（百分位 landmark）
- **adaptive_histogram_equalization**：对比度受限自适应直方图均衡（CLAHE）

流水线行为
----------

1. **隐式加载**：每个被试在执行 ``Preprocessing`` 中各步骤前，会先用内部步骤 **``load_image``**（无需写入 YAML）将路径替换为内存中的 SimpleITK 图像。
2. **步骤顺序**：``Preprocessing`` 下各子块在 YAML 中的 **书写顺序** 即为执行顺序；调整顺序会改变结果（例如应先重采样再配准还是先配准，需按数据与课题决定）。
3. **并行**：顶层 ``processes`` 为并行被试进程数；实现中会取 ``min(processes, CPU 逻辑核心数 - 2)``，且至少为 **1**（避免占满整机）。
4. **重复性字段**：``random_state`` 在 ``PreprocessingConfig`` 中保留默认值，**当前图像预处理实现未使用该字段**；若需完全可复现，请关注各步骤内部算法与多进程顺序。
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

- 最终结果写入 ``out_dir/processed_images/``（见下文「输出结构」）。
- 使用 CLI 时，日志文件为 ``<out_dir>/processing.log``（若直接使用 ``BatchProcessor`` 且未先配置全局日志，则默认也为 ``processing.log`` 写在 ``out_dir`` 下）。

Python API 使用方法
------------------

**推荐路径（与 CLI 一致）**：通过 ``PreprocessingConfigurator`` 创建 ``BatchProcessor``。配置对象 **必须** 由 ``PreprocessingConfig.from_file(path)`` 加载，以便内部保存 ``config_file`` 供处理器读取。

.. code-block:: python

   from habit.core.preprocessing.configurator import PreprocessingConfigurator
   from habit.core.preprocessing.config_schemas import PreprocessingConfig

   config = PreprocessingConfig.from_file("./demo_data/config_preprocessing.yaml")
   configurator = PreprocessingConfigurator(config=config)
   processor = configurator.create_batch_processor()
   processor.run()

**直接使用 BatchProcessor**（自行传入 YAML 路径）：

.. code-block:: python

   from habit.core.preprocessing.image_processor_pipeline import BatchProcessor

   processor = BatchProcessor(config_path="./demo_data/config_preprocessing.yaml")
   processor.run()

**自定义日志**：若在外部调用 ``setup_logger``，请将日志文件名与 ``out_dir`` 约定清楚；CLI 使用 ``cli.preprocessing`` 与 ``processing.log``。

YAML 配置详解
--------------

**顶层字段（``PreprocessingConfig``）**

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - 字段
     - 说明
   * - ``data_dir``
     - 指向 **目录** 或 **文件清单 YAML**（参见 :doc:`../data_structure_zh`）。目录时可通过 ``keyword_of_raw_folder`` 等在 ``get_image_and_mask_paths`` 中解析（默认行为见 ``habit.utils.io_utils``）。
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

**``Preprocessing`` 公共约定**

- 每个步骤块 **必须** 包含非空列表 ``images``：模态键名，须与 ``files_preprocessing.yaml`` 等清单中一致。
- 掩码在内存中的键名为 ``mask_<模态>``（例如模态 ``delay2`` 对应 ``mask_delay2``）。配准在 ``use_mask: true`` 时使用固定图与浮动图对应的掩码键。

**完整示例（节选，注册名即 YAML 键名）：**

.. code-block:: yaml

   data_dir: ./files_preprocessing.yaml
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

**中间结果与 ``dcm2nii``**

- 对除 ``dcm2nii`` 外的步骤，中间结果常通过写入 ``<step>_<序号>/images|masks/...`` 保存。
- ``dcm2nii`` 自行将 NIfTI 写到当步配置的 ``output_dirs``；开启中间保存时，流水线 **不会** 再对 ``dcm2nii`` 调用通用落盘逻辑（避免重复），详见 ``BatchProcessor._process_single_subject``。

预处理方法详解
----------------

**dcm2nii（Dcm2niixConverter）**

除 ``images`` 外常用参数：``dcm2niix_path``（可选）、``compress``、``anonymize``、``filename_format``、``adjacent_dicoms``、``ignore_derived``、``crop_images``、``generate_json``、``verbose``、``batch_mode``、``merge_slices``、``single_file_mode`` 等。入口路径可为 DICOM 目录；须保证 ``dcm2niix`` 可用。上游工具说明与命令行标志见 `dcm2niix 项目 <https://github.com/rordenlab/dcm2niix>`__。

**n4_correction**

``num_fitting_levels``（默认 4）、``num_iterations``（默认每层 50 次，长度为层数）、``convergence_threshold``、``shrink_factor``（下采样加速）、可选 ``mask_keys``（掩膜参与校正）。算法细节见 SimpleITK `N4BiasFieldCorrectionImageFilter <https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1N4BiasFieldCorrectionImageFilter.html>`__。

**resample（``ResamplePreprocessor``，SimpleITK）**

HABIT 的**重采样不是** ANTs；实现位于 ``habit/core/preprocessing/resample.py``，核心为 SimpleITK ``ResampleImageFilter``：在保持 **原点 ``Origin`` 与方向矩阵 ``Direction`` 不变** 的前提下，按目标体素间距 ``target_spacing``（物理单位，与 SimpleITK 一致，一般为 mm）重建网格尺寸。

**官方参考（重采样 / 物理空间）**

- `SimpleITK 基本概念（间距、原点、方向） <https://simpleitk.readthedocs.io/en/master/FundamentalConcepts.html>`__
- `ResampleImageFilter（Doxygen） <https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ResampleImageFilter.html>`__

**HABIT 中的计算要点**

- 对每个体素维度 ``i``：\ ``factor_i = original_spacing[i] / target_spacing[i]``，\ ``new_size_i = round(size[i] * factor_i)``（见 ``_resample_image``）。
- 参考图 ``reference_image``：尺寸为 ``new_size``，间距为 ``target_spacing``，原点与方向取自输入图像；``ResampleImageFilter.SetReferenceImage`` 后 ``Execute``。

**配置参数**

- ``target_spacing``：长度为 3 的序列 ``[sx, sy, sz]``（单位与输入 NIfTI 一致）。
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

- **掩码**：凡键名 ``mask_<modality>`` 使用 **最近邻**（``sitkNearestNeighbor``），避免标签插值产生非整数类号。
- **未接入滤波的键**：``padding_mode``、``align_corners`` 在构造函数中可解析，但 **当前 ``_resample_image`` 未将其传入 ``ResampleImageFilter``**；若在 YAML 中填写，不会产生效果（保留作未来扩展）。需要边界填充或特殊对齐时，请改源码或自定义预处理器。

**registration（``RegistrationPreprocessor``，ANTsPy）**

HABIT 将多期向 ``fixed_image`` 配准，底层调用 **ANTsPy** ``ants.registration``（底层对应 ANTs 的 ``antsRegistration``）。**与** SimpleITK ``ImageRegistrationMethod`` **不是**同一套 API；若要对照理论（度量、优化器、多分辨率），请以 **ANTs/ANTsPy** 文档为准。

**官方参考（配准）**

- `ANTsPy — Registration（教程型说明） <https://antspy.readthedocs.io/en/stable/registration.html>`__
- `ants.registration API（参数列表） <https://antspy.readthedocs.io/en/stable/api/ants.registration.html>`__
- `ANTs 主仓库（构建与引用） <https://github.com/ANTsX/ANTs>`__

**HABIT 调用关系**

- 构造 ``reg_params = { "metric": ..., "optimizer": ..., **YAML中其余关键字}``；若 ``use_mask`` 为真且存在掩膜，则附加 ``mask``（固定图像侧）、``moving_mask``（浮动图像侧），对应 ANTs 的掩膜参数。
- 最终调用等价于：\ ``ants.registration(fixed=..., moving=..., type_of_transform=..., **reg_params)``（见 ``registration.py``）。
- **浮动序列**：``images`` 中除 ``fixed_image`` 外的每个键依次作为 ``moving``；不要添加 ``moving_images``，以免多余键进入 ``**reg_params``。

**``type_of_transform``（写入 YAML 的字符串与 ANTs 一致）**

下列与 ``habit/core/preprocessing/registration.py`` docstring 一致，并随 ANTsPy 版本可能增删；**完整与默认值以** ``ants.registration`` **文档为准**。

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

**HABIT 显式包装的参数**

- ``metric``（默认 ``MI``）：相似性度量简写，将写入 ``reg_params``；可选名参见 RegistrationPreprocessor docstring（例如 ``CC``、``MI``、``MeanSquares``、``Demons``）。
- ``optimizer``：写入 ``reg_params``；可选 ``gradient_descent``、``lbfgsb``、``amoeba`` 等（以 ANTsPy 接受为准）。

**通过 YAML 透传到 ANTs 的高级参数**

在 ``registration:`` 下除 ``images`` / ``fixed_image`` 外，凡 **未** 被 HABIT 单独解析的键（若与 ANTs 参数名一致）会进入 ``**kwargs`` 再合并进 ``ants.registration``。常用例子（是否可用取决于你的 ANTsPy 版本，请务必查阅上列 **API 文档**）：

- 迭代与多分辨率：如 ``reg_iterations``、``aff_iterations``、``aff_shrink_factors``、``aff_smoothing_sigmas``、``grad_step``、``flow_sigma``、``total_sigma`` 等。
- 度量细分：如 ``aff_metric``、``syn_metric`` 等与阶段相关的度量选项。
- 初始变换：如 ``initial_transform``。

勿把 HABIT 未定义且无意义的键（例如历史示例里的 ``moving_images``）放进 YAML。

**其它**：``replace_by_fixed_image_mask``、掩膜配准后的 ``ants.apply_transforms`` 对掩膜使用最近邻等，见源码 ``registration.py``。

**zscore_normalization**

``only_inmask`` 与单个 ``mask_key``：启用仅在掩膜内统计时，``mask_key`` 须是 **``data`` 字典中存在的键**（例如与某模态对齐的掩膜键名）；多模态时常用共享掩膜键。``clip_values``：可选，归一化后裁剪到 ``(low, high)``。

**histogram_standardization**

Nyúl 方法：``percentiles`` 分位点、``target_min`` / ``target_max`` 映射目标范围、可选 ``mask_key`` 限定直方图统计体素。

**adaptive_histogram_equalization**

``alpha``、``beta``（[0,1]）与 ``radius`` 传入 SimpleITK ``AdaptiveHistogramEqualizationImageFilter``；行为与参数解释以上游为准：`AdaptiveHistogramEqualizationImageFilter <https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1AdaptiveHistogramEqualizationImageFilter.html>`__。

自定义预处理器
------------

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

- 安装 ANTsPy 与 ANTs；尝试 ``Rigid`` / ``Affine`` 降低耗时；减少并行 ``processes`` 以免内存占满。

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
