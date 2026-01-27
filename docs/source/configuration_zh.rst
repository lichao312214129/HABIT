配置参考
========

本节详细说明 HABIT 的所有配置文件参数和选项。

概述
----

HABIT 使用 YAML 格式的配置文件来控制所有功能。每个功能模块都有对应的配置文件，用户可以通过修改配置文件来调整功能。

**配置文件类型：**

- **预处理配置**: 控制图像预处理流程
- **生境分析配置**: 控制生境分割和特征提取
- **特征提取配置**: 控制生境特征提取
- **机器学习配置**: 控制机器学习建模
- **数据配置**: 指定数据路径和结构

**配置文件特点：**

- **易于理解**: 使用 YAML 格式，易于阅读和编辑
- **灵活配置**: 支持多种参数组合
- **版本控制**: 可以纳入版本控制，便于追踪变更
- **可重复性**: 相同的配置文件产生相同的结果

通用配置参数
------------

**data_dir**: 数据目录路径

- **类型**: 字符串
- **必需**: 是
- **说明**: 可以是文件夹或 YAML 配置文件
- **示例**: `./files_preprocessing.yaml`

**out_dir**: 输出目录路径

- **类型**: 字符串
- **必需**: 是
- **说明**: 输出文件将保存在此目录
- **示例**: `./preprocessed`

**processes**: 并行进程数

- **类型**: 整数
- **必需**: 否
- **默认值**: 2
- **说明**: 用于并行处理的进程数
- **示例**: `4`

**random_state**: 随机种子

- **类型**: 整数
- **必需**: 否
- **默认值**: None
- **说明**: 用于可重复性的随机种子
- **示例**: `42`

**debug**: 调试模式

- **类型**: 布尔值
- **必需**: 否
- **默认值**: false
- **说明**: 启用详细日志的调试模式
- **示例**: `true`

预处理配置参数
------------

**配置文件示例：**

.. code-block:: yaml

   data_dir: ./files_preprocessing.yaml
   out_dir: ./preprocessed

   Preprocessing:
     dcm2nii:
       images: [delay2, delay3, delay5]
       dcm2niix_path: ./dcm2niix.exe
       compress: true
       anonymize: true

     n4_correction:
       images: [delay2, delay3, delay5]
       num_fitting_levels: 4

     resample:
       images: [delay2, delay3, delay5]
       target_spacing: [1.0, 1.0, 1.0]

     registration:
       images: [delay2, delay3, delay5]
       fixed_image: delay2
       moving_images: [delay3, delay5]
       type_of_transform: SyNRA
       use_mask: false

     zscore_normalization:
       images: [delay2, delay3, delay5]
       only_inmask: false
       mask_key: mask

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

**Preprocessing**: 预处理设置

**dcm2niix**: DICOM 转换设置

- ``images``: 要转换的图像列表

  - **类型**: 列表
  - **必需**: 是
  - **示例**: ``[delay2, delay3, delay5]``

- ``dcm2niix_path``: dcm2niix 可执行文件路径

  - **类型**: 字符串
  - **必需**: 是
  - **示例**: ``./dcm2niix.exe``

- ``compress``: 是否压缩输出文件

  - **类型**: 布尔值
  - **必需**: 否
  - **默认值**: ``true``
  - **示例**: ``true``

- ``anonymize``: 是否匿名化

  - **类型**: 布尔值
  - **必需**: 否
  - **默认值**: ``true``
  - **示例**: ``true``

**n4_correction**: N4 偏置场校正设置

- ``images``: 要校正的图像列表

  - **类型**: 列表
  - **必需**: 是
  - **示例**: ``[delay2, delay3, delay5]``

- ``num_fitting_levels``: 拟合级别数

  - **类型**: 整数
  - **必需**: 否
  - **默认值**: 4
  - **范围**: 2-4
  - **示例**: ``4``

**resample**: 重采样设置

- ``images``: 要重采样的图像列表

  - **类型**: 列表
  - **必需**: 是
  - **示例**: ``[delay2, delay3, delay5]``

- ``target_spacing``: 目标间距

  - **类型**: 列表
  - **必需**: 是
  - **格式**: [x, y, z]（单位：mm）
  - **示例**: ``[1.0, 1.0, 1.0]``

**registration**: 配准设置

- ``images``: 所有涉及的图像列表

  - **类型**: 列表
  - **必需**: 是
  - **示例**: ``[delay2, delay3, delay5]``

- ``fixed_image``: 固定图像

  - **类型**: 字符串
  - **必需**: 是
  - **说明**: 参考图像
  - **示例**: ``delay2``

- ``moving_images``: 要配准的图像列表

  - **类型**: 列表
  - **必需**: 是
  - **示例**: ``[delay3, delay5]``

- ``type_of_transform``: 变换类型

  - **类型**: 字符串
  - **必需**: 否
  - **默认值**: ``SyNRA``
  - **可选值**: ``SyNRA``, ``SyN``, ``Affine``
  - **示例**: ``SyNRA``

- ``use_mask``: 是否使用掩码引导配准

  - **类型**: 布尔值
  - **必需**: 否
  - **默认值**: ``false``
  - **示例**: ``false``

- ``mask_key``: 掩码键名

  - **类型**: 字符串
  - **必需**: 否（当 ``use_mask`` 为 ``true`` 时必需）
  - **示例**: ``mask``

**zscore_normalization**: Z-Score 标准化设置

- ``images``: 要标准化的图像列表

  - **类型**: 列表
  - **必需**: 是
  - **示例**: ``[delay2, delay3, delay5]``

- ``only_inmask``: 是否仅在掩码内计算统计量

  - **类型**: 布尔值
  - **必需**: 否
  - **默认值**: ``false``
  - **示例**: ``false``

- ``mask_key``: 掩码键名

  - **类型**: 字符串
  - **必需**: 否（当 ``only_inmask`` 为 ``true`` 时必需）
  - **示例**: ``mask``

**adaptive_histogram_equalization**: 自适应直方图均衡化设置

- ``images``: 要均衡化的图像列表

  - **类型**: 列表
  - **必需**: 是
  - **示例**: ``[delay2, delay3, delay5]``

- ``alpha``: 全局对比度增强因子

  - **类型**: 浮点数
  - **必需**: 否
  - **默认值**: 0.3
  - **范围**: [0, 1]
  - **示例**: ``0.3``

- ``beta``: 局部对比度增强因子

  - **类型**: 浮点数
  - **必需**: 否
  - **默认值**: 0.3
  - **范围**: [0, 1]
  - **示例**: ``0.3``

- ``radius``: 局部窗口半径

  - **类型**: 整数
  - **必需**: 否
  - **默认值**: 5
  - **单位**: 像素
  - **示例**: ``5``

**save_options**: 保存选项

- ``save_intermediate``: 是否保存中间结果

  - **类型**: 布尔值
  - **必需**: 否
  - **默认值**: ``false``
  - **示例**: ``true``

- ``intermediate_steps``: 要保存的中间步骤列表

  - **类型**: 列表
  - **必需**: 否
  - **默认值**: 空列表（表示保存所有步骤）
  - **示例**: ``[dcm2nii, n4_correction, resample]``

生境分析配置参数
------------

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

   HabitatsSegmention:
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
   plot_curves: true
   save_results_csv: true
   random_state: 42
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
- **必需**: 否 (``predict`` 模式必需)
- **说明**: 指定训练好的 Pipeline 文件路径。
- **示例**: ``./results/habitat_pipeline.pkl``

**FeatureConstruction**: 特征提取设置

**voxel_level**: 体素级特征提取

- ``method``: 特征提取方法表达式

  - **类型**: 字符串
  - **必需**: 是
  - **说明**: 支持函数式语法组合多个特征提取器。
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

        - ``wash_in_slope``: 洗入斜率
        - ``wash_out_slope_lap_pvp``: LAP 到 PVP 的洗出斜率
        - ``wash_out_slope_pvp_dp``: PVP 到延迟期的洗出斜率

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

      - **示例**: ``voxel_radiomics(raw(delay2), params_file='./parameter.yaml', kernelRadius=1)``

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

- ``params``: 全局参数

  - **类型**: 字典
  - **必需**: 否
  - **默认值**: ``{}``
  - **说明**: 传递给所有特征提取器的公共参数。
  - **常用参数**:

    - ``timestamps`` (str): 时间戳文件路径（用于 kinetic 方法）
    - ``kernel_size`` (int): 局部邻域大小（用于 local_entropy）
    - ``bins`` (int): 直方图分箱数（用于 local_entropy）
    - ``params_file`` (str): PyRadiomics 参数文件（用于 voxel_radiomics）
    - ``kernelRadius`` (int): 体素级组学邻域半径（用于 voxel_radiomics）

**supervoxel_level**: 超像素级特征提取 (可选)

- ``supervoxel_file_keyword``: 超像素文件匹配模式

  - **类型**: 字符串
  - **必需**: 是
  - **默认值**: ``"*_supervoxel.nrrd"``
  - **说明**: 用于匹配已有的超像素分割文件（由 two_step 模式生成）。
  - **示例**: ``"*_supervoxel.nrrd"``

- ``method``: 特征聚合/提取方法

  - **类型**: 字符串
  - **必需**: 是
  - **默认值**: ``"mean_voxel_features()"``
  - **说明**: 定义如何从体素特征聚合到超像素，或直接从超像素提取特征。
  - **可用方法及参数**:

    **mean_voxel_features()**:

      - **说明**: 计算每个超像素内体素特征的平均值（最常用）
      - **参数**: 无
      - **用途**: 将体素级特征（如 ``voxel_level`` 提取的特征）聚合到超像素级
      - **示例**: ``mean_voxel_features()``

    **supervoxel_radiomics(params_file=...)**:

      - **说明**: 直接从原始图像的超像素块提取影像组学特征
      - **参数**:

        - ``params_file`` (str, 必需): PyRadiomics 参数文件路径

      - **用途**: 不依赖 ``voxel_level`` 特征，直接从超像素区域提取纹理、形状等组学特征
      - **示例**: ``supervoxel_radiomics(params_file='./parameter.yaml')``

  - **方法对比**:

    - ``mean_voxel_features()``: 依赖 ``voxel_level`` 特征，速度快，适合大多数场景
    - ``supervoxel_radiomics()``: 独立提取，特征更丰富但计算量大

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
         method: supervoxel_radiomics()
         params:
           params_file: ./parameter_supervoxel.yaml

- ``params``: 参数

  - **类型**: 字典
  - **必需**: 否
  - **默认值**: ``{}``
  - **说明**: 传递给特征提取器的参数（如 ``params_file``）。

**preprocessing_for_subject_level**: 个体级别预处理 (可选)

- ``methods``: 预处理方法列表

  - **类型**: 列表
  - **必需**: 否
  - **默认值**: ``[]``
  - **说明**: 在个体水平对特征进行预处理，消除个体内异常值和尺度差异。
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

- ``methods``: 预处理方法列表

  - **类型**: 列表
  - **必需**: 否
  - **默认值**: ``[]``
  - **说明**: 在群体水平对特征进行预处理，通常用于离散化以提高聚类的稳定性。
  - **支持方法及参数**:

    **binning (特征离散化/分箱)**:

      - ``n_bins`` (int, 默认: ``10``): 分箱数量
      - ``bin_strategy`` (str, 默认: ``uniform``): 分箱策略，可选:

        - ``uniform``: 均匀分箱（等宽）
        - ``quantile``: 分位数分箱（等频）
        - ``kmeans``: K-means 聚类分箱

      - ``global_normalize`` (bool, 默认: ``false``): 是否全局分箱（跨所有特征）

    **winsorize (缩尾处理)**:

      - ``winsor_limits`` (list, 默认: ``[0.05, 0.05]``): 下限和上限的截断比例
      - ``global_normalize`` (bool, 默认: ``false``): 是否全局归一化

    **minmax / zscore / robust / log**:

      - 同 ``preprocessing_for_subject_level``，但作用于群体汇总后的数据

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

**HabitatsSegmention**: 生境分割设置

- ``clustering_mode``: 聚类策略

  - **类型**: 字符串
  - **必需**: 否
  - **默认值**: ``two_step``
  - **可选值**:

    - ``one_step``: 直接对体素进行聚类。
    - ``two_step``: 先生成超像素，再对超像素进行聚类生成生境。
    - ``direct_pooling``: 直接汇总所有受试者的体素进行聚类（计算量大）。

  - **示例**: ``two_step``

**supervoxel**: 超像素聚类设置 (仅用于 ``two_step`` 模式)

- ``algorithm``: 聚类算法

  - **类型**: 字符串
  - **默认值**: ``kmeans``
  - **可选值**:

    - ``kmeans``: K-means 聚类（速度快，适合大多数场景）
    - ``gmm``: 高斯混合模型（考虑数据分布，更灵活但速度较慢）

  - **示例**: ``kmeans``

- ``n_clusters``: 超像素数量

  - **类型**: 整数
  - **必需**: 是
  - **说明**: 每个受试者生成的超像素个数。推荐范围: 30-100。
  - **示例**: ``50``

- ``random_state``: 随机种子

  - **类型**: 整数
  - **默认值**: ``42``
  - **说明**: 用于结果可重复性

- ``max_iter``: 最大迭代次数

  - **类型**: 整数
  - **默认值**: ``300``
  - **说明**: 聚类算法的最大迭代次数

- ``n_init``: 初始化次数

  - **类型**: 整数
  - **默认值**: ``10``
  - **说明**: 使用不同初始化运行算法的次数，选择最佳结果

- ``covariance_type``: 协方差类型（仅用于 ``gmm``）

  - **类型**: 字符串
  - **默认值**: ``full``
  - **可选值**: ``full``, ``tied``, ``diag``, ``spherical``
  - **说明**: 

    - ``full``: 每个组件有独立的完整协方差矩阵
    - ``tied``: 所有组件共享相同的协方差矩阵
    - ``diag``: 对角协方差矩阵（假设特征独立）
    - ``spherical``: 球形协方差（各向同性）

- **完整示例**:

  .. code-block:: yaml

     # K-means 聚类（推荐）
     supervoxel:
       algorithm: kmeans
       n_clusters: 50
       random_state: 42
       max_iter: 300
       n_init: 10
     
     # GMM 聚类
     supervoxel:
       algorithm: gmm
       n_clusters: 50
       covariance_type: full
       random_state: 42
       max_iter: 100
       n_init: 5

**one_step_settings**: One-Step 模式设置 (仅用于 ``one_step`` 模式)

- ``min_clusters``: 最小聚类数

  - **类型**: 整数
  - **默认值**: ``2``
  - **说明**: 自动选择时的下限

- ``max_clusters``: 最大聚类数

  - **类型**: 整数
  - **默认值**: ``10``
  - **说明**: 自动选择时的上限

- ``fixed_n_clusters``: 固定聚类数

  - **类型**: 整数或 null
  - **默认值**: ``null``
  - **说明**: 若设置，则跳过自动选择，直接使用该值。

- ``selection_method``: 自动选择指标

  - **类型**: 字符串
  - **默认值**: ``silhouette``
  - **可选值及说明**:

    - ``silhouette``: 轮廓系数（-1 到 1，越接近 1 表示聚类越紧密）
    - ``calinski_harabasz``: Calinski-Harabasz 指数（越大表示聚类越好）
    - ``davies_bouldin``: Davies-Bouldin 指数（越小表示聚类越好）
    - ``inertia``: 簇内平方和（越小表示聚类越紧密，但可能过拟合）

  - **推荐**: ``silhouette``（综合性能最佳）

- ``plot_validation_curves``: 是否绘制验证曲线

  - **类型**: 布尔值
  - **默认值**: ``true``
  - **说明**: 生成不同聚类数下的指标曲线图，帮助理解自动选择结果

**habitat**: 生境聚类设置

- ``algorithm``: 聚类算法

  - **类型**: 字符串
  - **默认值**: ``kmeans``
  - **可选值**:

    - ``kmeans``: K-means 聚类
    - ``gmm``: 高斯混合模型

- ``max_clusters``: 最大生境数

  - **类型**: 整数
  - **必需**: 是
  - **说明**: 自动选择生境数时的上限。推荐范围: 5-10。
  - **示例**: ``10``

- ``min_clusters``: 最小生境数

  - **类型**: 整数
  - **默认值**: ``2``
  - **说明**: 自动选择生境数时的下限。

- ``habitat_cluster_selection_method``: 自动选择指标

  - **类型**: 列表或字符串
  - **默认值**: ``[inertia]``
  - **可选值及说明**:

    - ``inertia``: 簇内平方和（越小越好，适用于 kmeans）
    - ``silhouette``: 轮廓系数（-1 到 1，越接近 1 越好）
    - ``calinski_harabasz``: Calinski-Harabasz 指数（越大越好）
    - ``davies_bouldin``: Davies-Bouldin 指数（越小越好）
    - ``aic``: 赤池信息准则（越小越好，仅用于 gmm）
    - ``bic``: 贝叶斯信息准则（越小越好，仅用于 gmm）

  - **说明**: 可指定多个指标，系统会综合评估选择最佳生境数。
  - **示例**: ``[inertia, silhouette]``

- ``fixed_n_clusters``: 固定生境数

  - **类型**: 整数或 null
  - **默认值**: ``null``
  - **说明**: 若设置为具体数值，则跳过自动选择，直接使用该生境数。

- ``random_state``: 随机种子

  - **类型**: 整数
  - **默认值**: ``42``

- ``max_iter``: 最大迭代次数

  - **类型**: 整数
  - **默认值**: ``300`` (kmeans) 或 ``100`` (gmm)

- ``n_init``: 初始化次数

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

**plot_curves**: 是否生成和保存图表

- **类型**: 布尔值
- **默认值**: ``true``

**save_results_csv**: 是否将结果保存为 CSV 文件

- **类型**: 布尔值
- **默认值**: ``true``

特征提取配置参数
------------

**配置文件示例：**

.. code-block:: yaml

   params_file_of_non_habitat: ./parameter.yaml
   params_file_of_habitat: ./parameter_habitat.yaml

   raw_img_folder: ./preprocessed/processed_images
   habitats_map_folder: ./results/habitat
   out_dir: ./results/features

   n_processes:3
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
- **说明**: 使用 pyradiomics 提取传统影像组学特征的参数文件
- **示例**: ``./parameter.yaml``

**params_file_of_habitat**: 从生境图提取特征的参数文件

- **类型**: 字符串
- **必需**: 是
- **说明**: 使用 pyradiomics 从生境图中提取特征的参数文件
- **示例**: ``./parameter_habitat.yaml``

**raw_img_folder**: 原始图像根目录

- **类型**: 字符串
- **必需**: 是
- **说明**: 包含预处理后的图像
- **示例**: ``./preprocessed/processed_images``

**habitats_map_folder**: 生境图根目录

- **类型**: 字符串
- **必需**: 是
- **说明**: 包含生成的生境图
- **示例**: ``./results/habitat``

**out_dir**: 输出目录

- **类型**: 字符串
- **必需**: 是
- **说明**: 特征文件将保存在此目录
- **示例**: ``./results/features``

**n_processes**: 并行进程数

- **类型**: 整数
- **必需**: 否
- **默认值**: 2
- **说明**: 用于并行处理的进程数
- **示例**: ``3``

**habitat_pattern**: 生境文件匹配模式

- **类型**: 字符串
- **必需**: 否
- **默认值**: ``'*_habitats.nrrd'``
- **说明**: 用于匹配生境图文件，支持通配符（`*`）
- **示例**: ``*_habitats.nrrd``

**feature_types**: 特征类型列表

- **类型**: 列表
- **必需**: 否
- **默认值**: ``[traditional]``
- **可选值**: ``traditional``, ``non_radiomics``, ``whole_habitat``, ``each_habitat``, ``msi``, ``ith_score``
- **示例**: ``[traditional, non_radiomics, whole_habitat]``

**n_habitats**: 生境数量

- **类型**: 整数或 null
- **必需**: 否
- **默认值**: ``null``（表示自动检测）
- **说明**: 可以手动指定生境数量
- **示例**: ``null``

机器学习配置参数
------------

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

**run_mode**: 运行模式

- **类型**: 字符串
- **默认值**: ``train``
- **可选值**: ``train``, ``predict``
- **说明**: ``train`` 表示训练新模型，``predict`` 表示使用预训练模型进行预测。

**input**: 输入数据配置

- **类型**: 列表
- **必需**: 是
- **说明**: 包含一个或多个输入文件的配置字典。
- **子参数**:

  - ``path``: 特征文件路径 (CSV/Excel)。
  - ``name``: 数据集名称。
  - ``subject_id_col``: 受试者 ID 列名。
  - ``label_col``: 标签列名。

**output**: 输出目录

- **类型**: 字符串
- **必需**: 是
- **说明**: 结果、模型和图表保存的路径。

**split_method**: 数据划分方法

- **类型**: 字符串
- **默认值**: ``stratified``
- **可选值**: ``random``, ``stratified``, ``custom``

**test_size**: 测试集比例

- **类型**: 浮点数
- **默认值**: ``0.3``
- **范围**: (0, 1)

**normalization**: 特征归一化设置

- ``method``: 归一化方法

  - **类型**: 字符串
  - **默认值**: ``z_score``
  - **可选值**:

    - ``z_score``: Z-Score 标准化 (StandardScaler)
    - ``min_max``: 最小-最大归一化 (MinMaxScaler)
    - ``robust``: 鲁棒缩放 (RobustScaler)
    - ``max_abs``: 最大绝对值缩放 (MaxAbsScaler)
    - ``normalizer``: L1/L2 归一化 (Normalizer)
    - ``quantile``: 分位数转换 (QuantileTransformer)
    - ``power``: 幂变换 (PowerTransformer)

- ``params``: 方法特定参数

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

**feature_selection_methods**: 特征选择方法列表

- **类型**: 列表
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

定义要训练的一个或多个模型。

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

**visualization**: 可视化详细设置

- ``plot_types``: 要生成的图表类型。

  - **可选值**: ``roc``, ``dca``, ``calibration``, ``pr``, ``confusion``, ``shap``

- ``dpi``: 分辨率 (默认 600)。
- ``format``: 文件格式 (如 ``pdf``, ``png``)。

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

  - ``true``: 自动读取目录中的第一个文件（适用于已转换的 nii 文件等场景）。
  - ``false``: 保持目录路径不变（适用于 dcm2nii 等需要整个文件夹的任务）。

**images**: 图像数据路径

- **类型**: 字典
- **必需**: 是
- **说明**: 嵌套字典，第一层是受试者 ID，第二层是图像类型（Key）。

**masks**: 掩码数据路径

- **类型**: 字典
- **必需**: 否
- **说明**: 结构同 ``images``。通常用于指定 ROI。

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

   from habit.core.common.config_loader import load_config

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
