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

**dcm2nii**: DICOM 转换设置

- `images`: 要转换的图像列表
  - 类型: 列表
  - 必需: 是
  - 示例: `[delay2, delay3, delay5]`

- `dcm2niix_path`: dcm2niix 可执行文件路径
  - 类型: 字符串
  - 必需: 是
  - 示例: `./dcm2niix.exe`

- `compress`: 是否压缩输出文件
  - 类型: 布尔值
  - 必需: 否
  - 默认值: true
  - 示例: `true`

- `anonymize`: 是否匿名化
  - 类型: 布尔值
  - 必需: 否
  - 默认值: true
  - 示例: `true`

**n4_correction**: N4 偏置场校正设置

- `images`: 要校正的图像列表
  - 类型: 列表
  - 必需: 是
  - 示例: `[delay2, delay3, delay5]`

- `num_fitting_levels`: 拟合级别数
  - 类型: 整数
  - 必需: 否
  - 默认值: 4
  - 范围: 2-4
  - 示例: `4`

**resample**: 重采样设置

- `images`: 要重采样的图像列表
  - 类型: 列表
  - 必需: 是
  - 示例: `[delay2, delay3, delay5]`

- `target_spacing`: 目标间距
  - 类型: 列表
  - 必需: 是
  - 格式: [x, y, z]（单位：mm）
  - 示例: `[1.0, 1.0, 1.0]`

**registration**: 配准设置

- `images`: 所有涉及的图像列表
  - 类型: 列表
  - 必需: 是
  - 示例: `[delay2, delay3, delay5]`

- `fixed_image`: 固定图像
  - 类型: 字符串
  - 必需: 是
  - 说明: 参考图像
  - 示例: `delay2`

- `moving_images`: 要配准的图像列表
  - 类型: 列表
  - 必需: 是
  - 示例: `[delay3, delay5]`

- `type_of_transform`: 变换类型
  - 类型: 字符串
  - 必需: 否
  - 默认值: SyNRA
  - 可选值: SyNRA、SyN、Affine
  - 示例: `SyNRA`

- `use_mask`: 是否使用掩码引导配准
  - 类型: 布尔值
  - 必需: 否
  - 默认值: false
  - 示例: `false`

- `mask_key`: 掩码键名
  - 类型: 字符串
  - 必需: 否（当 use_mask 为 true 时必需）
  - 示例: `mask`

**zscore_normalization**: Z-Score 标准化设置

- `images`: 要标准化的图像列表
  - 类型: 列表
  - 必需: 是
  - 示例: `[delay2, delay3, delay5]`

- `only_inmask`: 是否仅在掩码内计算统计量
  - 类型: 布尔值
  - 必需: 否
  - 默认值: false
  - 示例: `false`

- `mask_key`: 掩码键名
  - 类型: 字符串
  - 必需: 否（当 only_inmask 为 true 时必需）
  - 示例: `mask`

**adaptive_histogram_equalization**: 自适应直方图均衡化设置

- `images`: 要均衡化的图像列表
  - 类型: 列表
  - 必需: 是
  - 示例: `[delay2, delay3, delay5]`

- `alpha`: 全局对比度增强因子
  - 类型: 浮点数
  - 必需: 否
  - 默认值: 0.3
  - 范围: [0, 1]
  - 示例: `0.3`

- `beta`: 局部对比度增强因子
  - 类型: 浮点数
  - 必需: 否
  - 默认值: 0.3
  - 范围: [0, 1]
  - 示例: `0.3`

- `radius`: 局部窗口半径
  - 类型: 整数
  - 必需: 否
  - 默认值: 5
  - 单位: 像素
  - 示例: `5`

**save_options**: 保存选项

- `save_intermediate`: 是否保存中间结果
  - 类型: 布尔值
  - 必需: 否
  - 默认值: false
  - 示例: `true`

- `intermediate_steps`: 要保存的中间步骤列表
  - 类型: 列表
  - 必需: 否
  - 默认值: 空列表（表示保存所有步骤）
  - 示例: `[dcm2nii, n4_correction, resample]`

生境分析配置参数
------------

**配置文件示例：**

.. code-block:: yaml

   run_mode: predict
   pipeline_path: ./results/habitat_pipeline.pkl
   data_dir: ./file_habitat.yaml
   out_dir: ./results/habitat/predict

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

   HabitatsSegmention:
     clustering_mode: two_step

     supervoxel:
       algorithm: kmeans
       n_clusters: 50
       random_state: 42
       max_iter: 300
       n_init: 10

       one_step_settings:
         min_clusters: 2
         max_clusters: 10
         selection_method: inertia
         plot_validation_curves: true

     habitat:
       algorithm: kmeans
       max_clusters: 10
       habitat_cluster_selection_method:
         - inertia
         - silhouette
       best_n_clusters:
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
- **默认值**: predict
- **可选值**: train、predict
- **说明**: train 表示训练新模型，predict 表示使用预训练模型进行预测
- **示例**: `train`

**pipeline_path**: Pipeline 文件路径

- **类型**: 字符串
- **必需**: 否（predict 模式必需）
- **说明**: 指定训练好的 Pipeline 文件路径
- **示例**: `./results/habitat_pipeline.pkl`

**FeatureConstruction**: 特征提取设置

**voxel_level**: 体素级特征提取

- `method`: 特征提取方法
  - 类型: 字符串
  - 必需: 是
  - 示例: `concat(raw(delay2), raw(delay3), raw(delay5))`

- `params`: 参数
  - 类型: 字典
  - 必需: 否
  - 默认值: {}
  - 示例: `{}`

**supervoxel_level**: 超像素级特征提取

- `supervoxel_file_keyword`: 超像素文件匹配模式
  - 类型: 字符串
  - 必需: 是
  - 示例: `*_supervoxel.nrrd`

- `method`: 特征提取方法
  - 类型: 字符串
  - 必需: 是
  - 示例: `mean_voxel_features()`

- `params`: 参数
  - 类型: 字典
  - 必需: 否
  - 默认值: {}
  - 示例: `{params_file: {}}`

**preprocessing_for_subject_level**: 个体级别预处理

- `methods`: 预处理方法列表
  - 类型: 列表
  - 必需: 否
  - 默认值: []
  - 示例:
    ```yaml
    - method: winsorize
      winsor_limits: [0.05, 0.05]
      global_normalize: true
    - method: minmax
      global_normalize: true
    ```

**preprocessing_for_group_level**: 群体级别预处理

- `methods`: 预处理方法列表
  - 类型: 列表
  - 必需: 否
  - 默认值: []
  - 示例:
    ```yaml
    - method: binning
      n_bins: 10
      bin_strategy: uniform
      global_normalize: false
    ```

**HabitatsSegmention**: 生境分割设置

- `clustering_mode`: 聚类策略
  - 类型: 字符串
  - 必需: 否
  - 默认值: two_step
  - 可选值: one_step、two_step、direct_pooling
  - 示例: `two_step`

**supervoxel**: 超像素聚类设置

- `algorithm`: 聚类算法
  - 类型: 字符串
  - 必需: 否
  - 默认值: kmeans
  - 可选值: kmeans、gmm、dbscan、spectral、agglomerative
  - 示例: `kmeans`

- `n_clusters`: 聚类数
  - 类型: 整数
  - 必需: 是
  - 示例: `50`

- `random_state`: 随机种子
  - 类型: 整数
  - 必需: 否
  - 默认值: None
  - 示例: `42`

- `max_iter`: 最大迭代次数
  - 类型: 整数
  - 必需: 否
  - 默认值: 300
  - 示例: `300`

- `n_init`: 初始化次数
  - 类型: 整数
  - 必需: 否
  - 默认值: 10
  - 示例: `10`

**one_step_settings**: One-Step 模式设置

- `min_clusters`: 最小聚类数
  - 类型: 整数
  - 必需: 否
  - 默认值: 2
  - 示例: `2`

- `max_clusters`: 最大聚类数
  - 类型: 整数
  - 必需: 否
  - 默认值: 10
  - 示例: `10`

- `selection_method`: 选择方法
  - 类型: 字符串
  - 必需: 否
  - 默认值: inertia
  - 可选值: silhouette、calinski_harabasz、davies_bouldin、inertia
  - 示例: `silhouette`

- `plot_validation_curves`: 是否绘制验证曲线
  - 类型: 布尔值
  - 必需: 否
  - 默认值: true
  - 示例: `true`

**habitat**: 生境聚类设置

- `algorithm`: 聚类算法
  - 类型: 字符串
  - 必需: 否
  - 默认值: kmeans
  - 可选值: kmeans、gmm
  - 示例: `kmeans`

- `max_clusters`: 最大聚类数
  - 类型: 整数
  - 必需: 是
  - 示例: `10`

- `habitat_cluster_selection_method`: 生境聚类选择方法
  - 类型: 列表
  - 必需: 否
  - 默认值: [inertia]
  - 可选值: inertia、silhouette、calinski_harabasz、aic、bic、davies_bouldin
  - 示例: `[inertia, silhouette]`

- `best_n_clusters`: 最佳聚类数
  - 类型: 整数或 null
  - 必需: 否
  - 默认值: null（表示自动选择）
  - 示例: `null`

- `random_state`: 随机种子
  - 类型: 整数
  - 必需: 否
  - 默认值: None
  - 示例: `42`

- `max_iter`: 最大迭代次数
  - 类型: 整数
  - 必需: 否
  - 默认值: 300
  - 示例: `300`

- `n_init`: 初始化次数
  - 类型: 整数
  - 必需: 否
  - 默认值: 10
  - 示例: `10`

**plot_curves**: 是否生成和保存图表

- **类型**: 布尔值
- **必需**: 否
- **默认值**: true
- **示例**: `true`

**save_results_csv**: 是否将结果保存为 CSV 文件

- **类型**: 布尔值
- **必需**: 否
- **默认值**: true
- **示例**: `true`

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
- **示例**: `./parameter.yaml`

**params_file_of_habitat**: 从生境图提取特征的参数文件

- **类型**: 字符串
- **必需**: 是
- **说明**: 使用 pyradiomics 从生境图中提取特征的参数文件
- **示例**: `./parameter_habitat.yaml`

**raw_img_folder**: 原始图像根目录

- **类型**: 字符串
- **必需**: 是
- **说明**: 包含预处理后的图像
- **示例**: `./preprocessed/processed_images`

**habitats_map_folder**: 生境图根目录

- **类型**: 字符串
- **必需**: 是
- **说明**: 包含生成的生境图
- **示例**: `./results/habitat`

**out_dir**: 输出目录

- **类型**: 字符串
- **必需**: 是
- **说明**: 特征文件将保存在此目录
- **示例**: `./results/features`

**n_processes**: 并行进程数

- **类型**: 整数
- **必需**: 否
- **默认值**: 2
- **说明**: 用于并行处理的进程数
- **示例**: `3`

**habitat_pattern**: 生境文件匹配模式

- **类型**: 字符串
- **必需**: 否
- **默认值**: '*_habitats.nrrd'
- **说明**: 用于匹配生境图文件，支持通配符（`*`）
- **示例**: `*_habitats.nrrd`

**feature_types**: 特征类型列表

- **类型**: 列表
- **必需**: 否
- **默认值**: [traditional]
- **可选值**: traditional、non_radiomics、whole_habitat、each_habitat、msi、ith_score
- **示例**: `[traditional, non_radiomics, whole_habitat]`

**n_habitats**: 生境数量

- **类型**: 整数或 null
- **必需**: 否
- **默认值**: null（表示自动检测）
- **说明**: 可以手动指定生境数量
- **示例**: `null`

机器学习配置参数
------------

**配置文件示例：**

.. code-block:: yaml

   run_mode: train
   pipeline_path: ./results/ml/model_pipeline.pkl
   data_dir: ./files_ml.yaml
   out_dir: ./results/ml/train

   FeatureSelection:
     enabled: true
     method: variance
     params:
       threshold: 0.0

   ModelTraining:
     enabled: true
     model_type: RandomForest
     params:
       n_estimators: 100
       max_depth: null
       min_samples_split: 2
       min_samples_leaf:1
       random_state: 42

   ModelEvaluation:
     enabled: true
     metrics:
       - accuracy
       - precision
       - recall
       - f1
       - roc_auc
       - confusion_matrix
     cv:5
     test_size: 0.2
     random_state: 42

   ModelSaving:
     enabled: true
     save_path: ./results/ml/model_pipeline.pkl
     save_format: pkl

   processes: 2
   random_state: 42
   debug: false

**run_mode**: 运行模式

- **类型**: 字符串
- **必需**: 否
- **默认值**: train
- **可选值**: train、predict
- **说明**: train 表示训练新模型，predict 表示使用预训练模型进行预测
- **示例**: `train`

**pipeline_path**: Pipeline 文件路径

- **类型**: 字符串
- **必需**: 否（predict 模式必需）
- **说明**: 指定训练好的 Pipeline 文件路径
- **示例**: `./results/ml/model_pipeline.pkl`

**FeatureSelection**: 特征选择设置

- `enabled`: 是否启用特征选择
  - 类型: 布尔值
  - 必需: 否
  - 默认值: true
  - 示例: `true`

- `method`: 特征选择方法
  - 类型: 字符串
  - 必需: 否
  - 默认值: variance
  - 可选值: variance、correlation、anova、chi2、lasso、rfecv
  - 示例: `variance`

- `params`: 特征选择方法的参数
  - 类型: 字典
  - 必需: 否
  - 默认值: {}
  - 示例: `{threshold: 0.0}`

**ModelTraining**: 模型训练设置

- `enabled`: 是否启用模型训练
  - 类型: 布尔值
  - 必需: 否
  - 默认值: true
  - 示例: `true`

- `model_type`: 模型类型
  - 类型: 字符串
  - 必需: 否
  - 默认值: RandomForest
  - 可选值: LogisticRegression、RandomForest、XGBoost、SVM、KNN、AutoGluon
  - 示例: `RandomForest`

- `params`: 模型参数
  - 类型: 字典
  - 必需: 否
  - 默认值: {}
  - 示例: `{n_estimators: 100, random_state: 42}`

**ModelEvaluation**: 模型评估设置

- `enabled`: 是否启用模型评估
  - 类型: 布尔值
  - 必需: 否
  - 默认值: true
  - 示例: `true`

- `metrics`: 评估指标列表
  - 类型: 列表
  - 必需: 否
  - 默认值: [accuracy]
  - 可选值: accuracy、precision、recall、f1、roc_auc、confusion_matrix
  - 示例: `[accuracy, precision, recall, f1, roc_auc]`

- `cv`: 交叉验证折数
  - 类型: 整数
  - 必需: 否
  - 默认值: 5
  - 示例: `5`

- `test_size`: 测试集比例
  - 类型: 浮点数
  - 必需: 否
  - 默认值: 0.2
  - 范围: (0, 1)
  - 示例: `0.2`

- `random_state`: 随机种子
  - 类型: 整数
  - 必需: 否
  - 默认值: None
  - 示例: `42`

**ModelSaving**: 模型保存设置

- `enabled`: 是否启用模型保存
  - 类型: 布尔值
  - 必需: 否
  - 默认值: true
  - 示例: `true`

- `save_path`: 模型保存路径
  - 类型: 字符串
  - 必需: 否
  - 默认值: ./model_pipeline.pkl
  - 示例: `./results/ml/model_pipeline.pkl`

- `save_format`: 保存格式
  - 类型: 字符串
  - 必需: 否
  - 默认值: pkl
  - 可选值: pkl、joblib
  - 示例: `pkl`

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
       FLAIR: /path/to/subject1/FLAIR/FLAIR.nii.gz
     subject2:
       T1: /path/to/subject2/T1/T1.nii.gz
       T2: /path/to/subject2/T2/T2.nii.gz
       FLAIR: /path/to/subject2/FLAIR/FLAIR.nii.gz

   masks:
     subject1:
       T1: /path/to/subject1/T1/mask_T1.nii.gz
       T2: /path/to/subject2/T2/mask_T2.nii.gz
       FLAIR: /path/to/subject1/FLAIR/mask_FLAIR.nii.gz
     subject2:
       T1: /path/to/subject2/T1/mask_T1.nii.gz
       T2: /path/to/subject2/T2/mask_T2.nii.gz
       FLAIR: /path/to/subject2/FLAIR/mask_FLAIR.nii.gz

**auto_select_first_file**: 是否自动读取目录中的第一个文件

- **类型**: 布尔值
- **必需**: 否
- **默认值**: true
- **说明**: 
  - true: 自动读取目录中的第一个文件（适用于已转换的nii文件等场景）
  - false: 保持目录路径不变（适用于dcm2nii等需要整个文件夹的任务）
- **示例**: `true`

**images**: 图像数据路径

- **类型**: 字典
- **必需**: 是
- **说明**: 嵌套字典，第一层是受试者ID，第二层是图像类型
- **示例**:
  ```yaml
  images:
    subject1:
      T1: /path/to/subject1/T1/T1.nii.gz
      T2: /path/to/subject1/T2/T2.nii.gz
  ```

**masks**: 掩码数据路径

- **类型**: 字典
- **必需**: 否
- **说明**: 嵌套字典，第一层是受试者ID，第二层是图像类型
- **示例**:
  ```yaml
  masks:
    subject1:
      T1: /path/to/subject1/T1/mask_T1.nii.gz
      T2: /path/to/subject1/T2/mask_T2.nii.gz
  ```

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
