# Habitat Analysis Logic and Flow

本文档详细说明 `habitat_analysis.py` 模块的内部逻辑和执行流程。

## 目录

1. [核心概念](#核心概念)
2. [初始化流程](#初始化流程)
3. [主执行流程](#主执行流程)
4. [个体级处理](#个体级处理)
5. [群体级聚类](#群体级聚类)
6. [特征提取系统](#特征提取系统)
7. [聚类策略](#聚类策略)
8. [并行处理](#并行处理)
9. [结果生成](#结果生成)

---

## 核心概念

### 什么是 Habitat Analysis？

Habitat Analysis（生境分析）是一种医学图像分析方法，用于识别肿瘤内部的不同亚区域（生境）。这些生境基于影像特征（如纹理、强度、形态等）进行划分，能够揭示肿瘤内部的异质性，可能具有预后或预测价值。

### 两种聚类策略

#### Two-Step 模式（默认）

```
个体级聚类 → 群体级聚类
(Voxel → Supervoxel → Supervoxel Features → Habitat)
```

1. **第一步：个体级聚类（每个患者独立）**
   - 提取体素特征（每个体素提取 radiomics 特征）
   - 对体素进行聚类，生成超体素（supervoxel）
   - 默认每个肿瘤生成 50 个超体素
   - **计算超体素特征**：
     - 当前实现：对每个超体素内的所有体素特征取平均值
     - 可扩展方式：提取超体素的组学特征、使用其他聚合方法（中位数、加权平均）等
   - 目的：降维并保留空间信息

2. **第二步：群体级聚类（跨所有患者）**
   - 拼接所有患者的超体素特征（例如：50 人 × 50 超体素 = 2500 个超体素）
   - 对超体素特征进行聚类以发现共同的生境模式
   - 自动确定最优生境数量
   - 每个超体素被分配到一个 habitat 标签

#### One-Step 模式

```
个体级聚类 = 最终生境
(Voxel → Habitat)
```

- 仅进行个体级聚类
- 每个肿瘤获得自己的生境分配
- 超体素即为生境（无群体级步骤）
- 适用于需要患者特定模式的情况

#### Direct Pooling 模式（新增）

```
全体体素拼接 → 单次聚类
(All Voxels → Habitat)
```

1. **特征拼接**
   - 对每个患者提取体素特征
   - 将所有患者的体素特征行拼接成一个大矩阵
   - 示例：50 人 × 100 体素 × 140 特征 → 5000 × 140

2. **全局聚类**
   - 在拼接矩阵上直接聚类
   - 获得全体体素的 habitat 标签

3. **回写生境图**
   - 按每位患者的体素顺序回写标签
   - 生成每个患者的 habitat map

> 该模式省略了 supervoxel 生成步骤，适合直接在体素级分析的场景。

---

## 策略模式与扩展机制

为降低 `habitat_analysis.py` 的复杂度并提升可扩展性，引入**策略模式**：

```
HabitatAnalysis
    └── Strategy (BaseHabitatStrategy)
          ├── OneStepStrategy
          ├── TwoStepStrategy
          └── DirectPoolingStrategy
```

### 核心思想
- **入口统一**：`HabitatAnalysis.run()` 只负责选择策略并执行
- **流程拆分**：各策略各自实现 `run()`，封装差异逻辑
- **易扩展**：新增策略只需新增一个类，无需修改主流程

---

## 初始化流程

### `__init__` 方法执行顺序

```python
HabitatAnalysis.__init__()
├── 1. 配置构建
│   ├── 优先使用 HabitatConfig 对象
│   └── 否则从遗留参数构建配置
├── 2. 日志设置 (_setup_logging)
├── 3. 特征配置验证 (_validate_feature_config)
├── 4. 数据路径设置 (_setup_data_paths)
├── 5. 特征提取器初始化 (_init_feature_extractor)
├── 6. 聚类算法初始化 (_init_clustering_algorithms)
├── 7. 选择方法初始化 (_init_selection_methods)
├── 8. 管道初始化 (_init_pipeline)
└── 9. 结果存储初始化 (_init_results_storage)
```

### 1. 配置构建

```python
if config is None:
    config = HabitatConfig.from_dict({
        'root_folder': root_folder,
        'out_folder': out_folder,
        'feature_config': feature_config or {},
        'clustering_strategy': clustering_strategy,
        # ... 其他参数
    })
```

**说明**：支持两种配置方式
- **推荐**：使用 `HabitatConfig` 对象（类型安全，易于维护）
- **兼容**：使用遗留参数（向后兼容旧代码）

### 2. 日志设置 (_setup_logging)

```python
def _setup_logging(self):
    manager = LoggerManager()
    
    if manager.get_log_file() is not None:
        # CLI 已配置日志，使用现有日志记录器
        self.logger = get_module_logger('habitat')
    else:
        # 创建新的日志配置
        self.logger = setup_logger(
            name='habitat',
            output_dir=self.config.io.out_folder,
            log_filename='habitat_analysis.log',
            level=level
        )
```

**关键点**：
- 支持从 CLI 继承日志配置
- 存储日志配置供子进程使用（Windows spawn 模式）
- 日志文件保存到输出目录

### 3. 特征配置验证 (_validate_feature_config)

```python
def _validate_feature_config(self):
    if 'voxel_level' not in self.config.feature_config:
        raise ValueError("voxel_level configuration is required")
```

**要求**：`voxel_level` 配置必须存在

### 4. 数据路径设置 (_setup_data_paths)

```python
def _setup_data_paths(self):
    # 创建输出目录
    os.makedirs(self.config.io.out_folder, exist_ok=True)
    
    # 获取图像和掩码路径
    self.images_paths, self.mask_paths = get_image_and_mask_paths(
        self.config.io.root_folder,
        keyword_of_raw_folder=self.config.io.images_dir,
        keyword_of_mask_folder=self.config.io.masks_dir
    )
    
    # 自动检测图像名称（如果未提供）
    if not self.config.io.image_names:
        self.config.io.image_names = detect_image_names(
            self.images_paths,
            self.mask_paths
        )
```

**数据结构**：
- `images_paths`: `{subject_id: {image_name: image_path}}`
- `mask_paths`: `{subject_id: mask_path}`

### 5. 特征提取器初始化 (_init_feature_extractor)

```python
def _init_feature_extractor(self):
    # 解析体素级特征表达式
    self.voxel_feature_parser = FeatureExpressionParser(
        self.config.feature_config['voxel_level']['method']
    )
    
    # 创建特征提取器
    self.feature_extractor = create_feature_extractor(
        self.voxel_feature_parser,
        self.config.io.image_names,
        self.config.feature_config['voxel_level'].get('params', {})
    )
    
    # 获取参数列表
    self.voxel_params = self.voxel_feature_parser.get_params()
```

**特征表达式示例**：
```python
"concat(raw(img1), raw(img2), kinetic(img1, img2))"
```

### 6. 聚类算法初始化 (_init_clustering_algorithms)

```python
def _init_clustering_algorithms(self):
    # 个体级聚类算法（Voxel → Supervoxel）
    self.voxel2supervoxel_clustering = get_clustering_algorithm(
        self.config.clustering.supervoxel_method,  # 'kmeans', 'gmm', etc.
        n_clusters=self.config.clustering.n_clusters_supervoxel,  # 默认 50
        random_state=self.config.clustering.random_state
    )
    
    # 群体级聚类算法（Supervoxel → Habitat）
    self.supervoxel2habitat_clustering = get_clustering_algorithm(
        self.config.clustering.habitat_method,
        n_clusters=self.config.clustering.n_clusters_habitats_max,  # 默认 10
        random_state=self.config.clustering.random_state
    )
```

**支持的算法**：
- `kmeans`: K-Means 聚类
- `gmm`: 高斯混合模型
- `hierarchical`: 层次聚类
- `spectral`: 谱聚类
- `dbscan`: DBSCAN 密度聚类
- `mean_shift`: 均值漂移
- `affinity_propagation`: 亲和传播

### 7. 选择方法初始化 (_init_selection_methods)

```python
def _init_selection_methods(self):
    # 获取算法支持的验证方法
    validation_info = get_validation_methods(self.config.clustering.habitat_method)
    valid_methods = list(validation_info['methods'].keys())
    default_methods = get_default_methods(self.config.clustering.habitat_method)
    
    # 验证并设置选择方法
    selection_methods = self.config.clustering.selection_methods
    
    if selection_methods is None:
        self.selection_methods = default_methods
    elif isinstance(selection_methods, str):
        # 单个方法
        self.selection_methods = selection_methods.lower()
    elif isinstance(selection_methods, list):
        # 多个方法（投票系统）
        valid = [m.lower() for m in selection_methods if is_valid_method_for_algorithm(...)]
        self.selection_methods = valid
```

**验证方法**：
| 方法 | 说明 | 选择原则 |
|------|------|----------|
| `silhouette` | 轮廓系数（-1 到 1，越高越好） | 最大原则 |
| `calinski_harabasz` | 方差比率（越高越好） | 最大原则 |
| `davies_bouldin` | 聚类分离度（越低越好） | 最小原则 |
| `inertia` | 误差平方和（越低越好） | 肘部法则 |
| `bic` / `aic` | 信息准则（越低越好） | 肘部法则 |
| `gap_statistic` | Gap 统计量（越高越好） | 最大原则 |

**投票系统**：当使用多个方法时，每个方法独立选择最优聚类数，得票最多的聚类数被选中。

### 8. 管道初始化 (_init_pipeline)

```python
def _init_pipeline(self):
    self.pipeline = create_pipeline(
        mode=self.config.runtime.mode,  # 'training' or 'testing'
        config=self.config,
        logger=self.logger
    )
```

**策略模式**：
- `TrainingPipeline`: 训练模式逻辑
- `TestingPipeline`: 测试模式逻辑

### 9. 结果存储初始化 (_init_results_storage)

```python
def _init_results_storage(self):
    self.results_df = None
    self.supervoxel2habitat_clustering_strategy = None
```

---

## 主执行流程

### `run()` 方法

```python
def run(self, subjects=None, save_results_csv=True):
    # 1. 提取特征并进行超体素聚类
    mean_features_all, failed_subjects = self._process_all_subjects(subjects)
    
    # 2. 准备群体级特征
    features_for_clustering = self._prepare_population_features(
        mean_features_all, subjects, failed_subjects
    )
    
    # 3. 执行群体级聚类（或跳过 one-step 模式）
    self.results_df = self._perform_population_clustering(
        mean_features_all, features_for_clustering
    )
    
    # 4. 保存结果
    if save_results_csv:
        self._save_results(subjects, failed_subjects)
    
    return self.results_df
```

### 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                         run() 主流程                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  _process_all_subjects()                     │
        │  - 并行处理所有受试者                          │
        │  - 提取特征                                   │
        │  - 个体级聚类（Voxel → Supervoxel）           │
        │  - 返回: mean_features_all, failed_subjects  │
        └───────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  _prepare_population_features()               │
        │  - 准备群体级聚类特征                          │
        │  - 标准化/预处理                               │
        └───────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  _perform_population_clustering()              │
        │  - one_step: 跳过，超体素=生境                 │
        │  - two_step: 群体级聚类（Supervoxel → Habitat）│
        │  - 确定最优聚类数                              │
        │  - 保存模型（训练模式）                        │
        └───────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  _save_results()                             │
        │  - 保存 CSV 文件                              │
        │  - 保存生境图像                               │
        │  - 保存配置文件                               │
        └───────────────────────────────────────────────┘
```

---

## 个体级处理

### `_process_all_subjects()` 方法

```python
def _process_all_subjects(self, subjects):
    # 并行处理所有受试者
    results, failed_subjects = parallel_map(
        func=self._voxel2supervoxel_clustering,
        items=subjects,
        n_processes=self.config.runtime.n_processes,
        desc="Processing subjects",
        logger=self.logger,
        show_progress=True,
        log_file_path=self._log_file_path,
        log_level=self._log_level,
    )
    
    # 合并结果
    mean_features_all = pd.DataFrame()
    for result in results:
        if result.success and result.result is not None:
            mean_features_all = pd.concat([mean_features_all, result.result], ignore_index=True)
    
    return mean_features_all, failed_subjects
```

### `_voxel2supervoxel_clustering()` 方法

```python
def _voxel2supervoxel_clustering(self, subject):
    # 1. 提取特征
    _, feature_df, raw_df, mask_info = self.extract_voxel_features(subject)
    
    # 2. 应用个体级预处理
    feature_df = self._apply_subject_preprocessing(feature_df)
    
    # 3. 执行聚类
    supervoxel_labels = self._cluster_subject_voxels(subject, feature_df)
    
    # 4. 计算超体素级别特征（当前：平均聚合）
    mean_features_df = self._calculate_supervoxel_means(
        subject, feature_df, raw_df, supervoxel_labels
    )
    
    # 5. 保存超体素图像
    self._save_supervoxel_image(subject, supervoxel_labels, mask_info)
    
    # 6. 可视化（如果启用）
    if self.config.runtime.plot_curves and HAS_VISUALIZATION:
        self._visualize_supervoxel_clustering(subject, feature_df, supervoxel_labels)
    
    return subject, mean_features_df
```

### 个体级处理流程图

```
┌─────────────────────────────────────────────────────────────────┐
│              _voxel2supervoxel_clustering(subject)                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  extract_voxel_features(subject)              │
        │  - 加载图像和掩码                              │
        │  - 提取体素级特征                              │
        │  - 返回: feature_df, raw_df, mask_info       │
        └───────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  _apply_subject_preprocessing(feature_df)     │
        │  - 标准化/归一化                              │
        │  - 其他预处理方法                              │
        └───────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  _cluster_subject_voxels(subject, feature_df) │
        │  - one_step: 确定最优聚类数                   │
        │  - 执行聚类（kmeans/gmm/etc）                 │
        │  - 返回: supervoxel_labels (1-indexed)      │
        └───────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  _calculate_supervoxel_means(...)            │
        │  - 计算超体素级别特征（当前实现：平均聚合）   │
        │  - 可扩展：组学特征提取、其他聚合方法等       │
        │  - 返回: mean_features_df                    │
        └───────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  _save_supervoxel_image(...)                 │
        │  - 保存超体素标签图（NRRD 格式）              │
        └───────────────────────────────────────────────┘
```

### `_cluster_subject_voxels()` 方法

```python
def _cluster_subject_voxels(self, subject, feature_df):
    # one_step 模式：确定最优聚类数
    if self.config.clustering.strategy == 'one_step':
        optimal_n_clusters = self._get_one_step_optimal_clusters(subject, feature_df)
        
        # 更新聚类算法
        self.voxel2supervoxel_clustering = get_clustering_algorithm(
            self.config.clustering.supervoxel_method,
            n_clusters=optimal_n_clusters,
            random_state=self.config.clustering.random_state
        )
    
    # 执行聚类
    self.voxel2supervoxel_clustering.fit(feature_df.values)
    supervoxel_labels = self.voxel2supervoxel_clustering.predict(feature_df.values)
    supervoxel_labels += 1  # 1-indexed
    
    return supervoxel_labels
```

### `_get_one_step_optimal_clusters()` 方法

```python
def _get_one_step_optimal_clusters(self, subject, feature_df):
    one_step = self.config.one_step
    
    # 使用固定聚类数（如果指定）
    if one_step and one_step.best_n_clusters is not None:
        return one_step.best_n_clusters
    
    # 使用验证方法确定最优聚类数
    clusterer = get_clustering_algorithm(
        self.config.clustering.supervoxel_method,
        n_clusters=one_step.max_clusters,
        random_state=self.config.clustering.random_state
    )
    
    # 测试不同聚类数
    scores = {}
    for n_clusters in range(one_step.min_clusters, one_step.max_clusters + 1):
        clusterer.n_clusters = n_clusters
        clusterer.fit(feature_df.values)
        
        # 计算验证分数
        for method in one_step.selection_method:
            if method not in scores:
                scores[method] = []
            scores[method].append(clusterer.score(method))
    
    # 选择最优聚类数（投票系统）
    optimal_n_clusters = self._select_optimal_clusters(scores)
    
    return optimal_n_clusters
```

---

## 群体级聚类

### `_perform_population_clustering()` 方法

```python
def _perform_population_clustering(self, mean_features_all, features):
    # one_step 模式：超体素即为生境
    if self.config.clustering.strategy == 'one_step':
        mean_features_all[ResultColumns.HABITATS] = mean_features_all[ResultColumns.SUPERVOXEL]
        return mean_features_all.copy()
    
    # two_step 模式：执行群体级聚类
    habitat_labels, optimal_n_clusters, scores = self.pipeline.cluster_habitats(
        features, self.supervoxel2habitat_clustering
    )
    
    # 绘制分数曲线（如果可用）
    if scores and self.config.runtime.plot_curves:
        self._plot_habitat_scores(scores, optimal_n_clusters)
    
    # 可视化聚类结果
    if self.config.runtime.plot_curves and HAS_VISUALIZATION:
        self._visualize_habitat_clustering(features, habitat_labels, optimal_n_clusters)
    
    # 保存模型（训练模式）
    if self.config.runtime.mode == 'training':
        self.pipeline.save_model(
            self.supervoxel2habitat_clustering,
            'supervoxel2habitat_clustering_strategy'
        )
    
    # 添加生境标签到结果
    mean_features_all[ResultColumns.HABITATS] = habitat_labels
    
    return mean_features_all.copy()
```

### 群体级聚类流程图

```
┌─────────────────────────────────────────────────────────────────┐
│           _perform_population_clustering()                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  one_step?            │
                    └───────────────────────┘
                      │ Yes              │ No
                      ▼                  ▼
        ┌───────────────────────┐  ┌─────────────────────────────┐
        │  supervoxel = habitat  │  │  pipeline.cluster_habitats()│
        │  直接赋值              │  │  - 测试不同聚类数           │
        │  返回结果              │  │  - 计算验证分数             │
        └───────────────────────┘  │  - 选择最优聚类数           │
                                  │  - 返回: habitat_labels     │
                                  └─────────────────────────────┘
                                              │
                                              ▼
                                  ┌─────────────────────────────┐
                                  │  训练模式?                   │
                                  └─────────────────────────────┘
                                    │ Yes              │ No
                                    ▼                  │
                          ┌───────────────────────┐    │
                          │  保存聚类模型          │    │
                          └───────────────────────┘    │
                                              │
                                              ▼
                                  ┌─────────────────────────────┐
                                  │  添加生境标签到结果          │
                                  │  返回 results_df            │
                                  └─────────────────────────────┘
```

### `TrainingPipeline.cluster_habitats()` 方法

```python
def cluster_habitats(self, features, clustering_algorithm):
    min_clusters = self.config.clustering.n_clusters_habitats_min
    max_clusters = self.config.clustering.n_clusters_habitats_max
    
    scores = {}
    cluster_labels = {}
    
    # 测试不同聚类数
    for n_clusters in range(min_clusters, max_clusters + 1):
        clustering_algorithm.n_clusters = n_clusters
        clustering_algorithm.fit(features.values)
        labels = clustering_algorithm.predict(features.values)
        cluster_labels[n_clusters] = labels
        
        # 计算验证分数
        for method in self.selection_methods:
            if method not in scores:
                scores[method] = []
            scores[method].append(clustering_algorithm.score(method))
    
    # 选择最优聚类数
    optimal_n_clusters = self._select_optimal_clusters(scores)
    
    # 使用最优聚类数重新聚类
    clustering_algorithm.n_clusters = optimal_n_clusters
    clustering_algorithm.fit(features.values)
    habitat_labels = clustering_algorithm.predict(features.values)
    
    return habitat_labels, optimal_n_clusters, scores
```

### `TestingPipeline.cluster_habitats()` 方法

```python
def cluster_habitats(self, features, clustering_algorithm):
    # 加载预训练模型
    model_path = os.path.join(
        self.config.io.out_folder,
        'supervoxel2habitat_clustering_strategy.pkl'
    )
    clustering_algorithm = load_model(model_path)
    
    # 直接预测
    habitat_labels = clustering_algorithm.predict(features.values)
    
    return habitat_labels, clustering_algorithm.n_clusters, None
```

---

## 特征提取系统

### `extract_voxel_features()` 方法

```python
def extract_voxel_features(self, subject):
    # 1. 加载图像和掩码
    images_dict = self._load_subject_images(subject)
    mask = self._load_subject_mask(subject)
    
    # 2. 提取体素级特征
    feature_df = self.feature_extractor.extract(images_dict, mask)
    
    # 3. 获取原始数据（用于计算超体素均值）
    raw_df = self._get_raw_data(images_dict, mask)
    
    # 4. 获取掩码信息
    mask_info = self._get_mask_info(mask)
    
    return images_dict, feature_df, raw_df, mask_info
```

### 特征提取流程

```
┌─────────────────────────────────────────────────────────────────┐
│              extract_voxel_features(subject)                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  _load_subject_images(subject)                │
        │  - 加载所有图像（T1, T2, FLAIR 等）            │
        │  - 返回: {image_name: image_array}           │
        └───────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  _load_subject_mask(subject)                  │
        │  - 加载肿瘤掩码                                │
        │  - 返回: mask_array                          │
        └───────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  feature_extractor.extract(images_dict, mask) │
        │  - 解析特征表达式                              │
        │  - 提取体素级特征                              │
        │  - 返回: feature_df (n_voxels × n_features)  │
        └───────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │  _get_raw_data(images_dict, mask)             │
        │  - 获取原始图像数据                            │
        │  - 返回: raw_df                              │
        └───────────────────────────────────────────────┘
```

### 特征表达式示例

```python
# 简单特征
"raw(T1)"

# 多模态特征拼接
"concat(raw(T1), raw(T2), raw(FLAIR))"

# 动力学特征
"kinetic(T1_pre, T1_post)"

# 熵特征
"local_entropy(T1, window_size=3)"

# 复杂组合
"concat(raw(T1), raw(T2), kinetic(T1_pre, T1_post), local_entropy(FLAIR))"
```

---

## 聚类策略

### Two-Step 模式详细流程

```
受试者 1                  受试者 2                  受试者 N
    │                         │                         │
    ▼                         ▼                         ▼
┌─────────┐              ┌─────────┐              ┌─────────┐
│  体素   │              │  体素   │              │  体素   │
│ 特征提取│              │ 特征提取│              │ 特征提取│
└────┬────┘              └────┬────┘              └────┬────┘
     │                         │                         │
     ▼                         ▼                         ▼
┌─────────┐              ┌─────────┐              ┌─────────┐
│ 个体级  │              │ 个体级  │              │ 个体级  │
│ 聚类    │              │ 聚类    │              │ 聚类    │
└────┬────┘              └────┬────┘              └────┬────┘
     │                         │                         │
     ▼                         ▼                         ▼
┌─────────┐              ┌─────────┐              ┌─────────┐
│ 50 个   │              │ 50 个   │              │ 50 个   │
│ 超体素  │              │ 超体素  │              │ 超体素  │
└────┬────┘              └────┬────┘              └────┬────┘
     │                         │                         │
     └─────────────────────────┼─────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  合并所有超体素      │
                    │  (N × 50)           │
                    └────────┬────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │  群体级聚类          │
                    │  (Supervoxel → Habitat)│
                    └────────┬────────────┘
                             │
                             ▼
                    ┌─────────────────────┐
                    │  K 个生境            │
                    │  (共同模式)          │
                    └─────────────────────┘
```

### One-Step 模式详细流程

```
受试者 1                  受试者 2                  受试者 N
    │                         │                         │
    ▼                         ▼                         ▼
┌─────────┐              ┌─────────┐              ┌─────────┐
│  体素   │              │  体素   │              │  体素   │
│ 特征提取│              │ 特征提取│              │ 特征提取│
└────┬────┘              └────┬────┘              └────┬────┘
     │                         │                         │
     ▼                         ▼                         ▼
┌─────────┐              ┌─────────┐              ┌─────────┐
│ 个体级  │              │ 个体级  │              │ 个体级  │
│ 聚类    │              │ 聚类    │              │ 聚类    │
│ (确定   │              │ (确定   │              │ (确定   │
│  最优   │              │  最优   │              │  最优   │
│  聚类数)│              │  聚类数)│              │  聚类数)│
└────┬────┘              └────┬────┘              └────┬────┘
     │                         │                         │
     ▼                         ▼                         ▼
┌─────────┐              ┌─────────┐              ┌─────────┐
│ K1 个   │              │ K2 个   │              │ KN 个   │
│ 生境    │              │ 生境    │              │ 生境    │
│ (独立)  │              │ (独立)  │              │ (独立)  │
└─────────┘              └─────────┘              └─────────┘
```

---

## 并行处理

### 并行处理架构

```python
def _process_all_subjects(self, subjects):
    results, failed_subjects = parallel_map(
        func=self._voxel2supervoxel_clustering,  # 并行执行的函数
        items=subjects,                           # 处理的项目列表
        n_processes=self.config.runtime.n_processes,  # 进程数
        desc="Processing subjects",               # 进度条描述
        logger=self.logger,                       # 日志记录器
        show_progress=True,                        # 显示进度条
        log_file_path=self._log_file_path,        # 日志文件路径
        log_level=self._log_level,                 # 日志级别
    )
```

### 并行处理流程

```
主进程
    │
    ├─── 分配任务
    │
    ├─── 子进程 1 ────▶ 处理受试者 1 ────▶ 返回结果
    │
    ├─── 子进程 2 ────▶ 处理受试者 2 ────▶ 返回结果
    │
    ├─── 子进程 3 ────▶ 处理受试者 3 ────▶ 返回结果
    │
    ├─── 子进程 4 ────▶ 处理受试者 4 ────▶ 返回结果
    │
    └─── 合并结果
```

### 子进程日志恢复

```python
def _ensure_logging_in_subprocess(self):
    """
    确保子进程中的日志配置正确
    
    在 Windows spawn 模式下，子进程不继承日志配置
    此方法恢复日志配置
    """
    from habit.utils.log_utils import restore_logging_in_subprocess
    
    if hasattr(self, '_log_file_path') and self._log_file_path:
        restore_logging_in_subprocess(self._log_file_path, self._log_level)
```

---

## 结果生成

### `_save_results()` 方法

```python
def _save_results(self, subjects, failed_subjects):
    # 1. 保存生境标签 CSV
    self._save_habitat_csv()
    
    # 2. 保存生境图像
    for subject in subjects:
        if subject not in failed_subjects:
            self._save_habitat_image(subject)
    
    # 3. 保存配置文件
    self._save_config()
    
    # 4. 保存特征统计
    self._save_feature_statistics()
```

### 输出文件

| 文件 | 描述 |
|------|------|
| `habitats.csv` | 包含每个超体素的生境标签 |
| `{subject}_supervoxel.nrrd` | 超体素标签图 |
| `{subject}_habitat.nrrd` | 最终生境标签图 |
| `config.yaml` / `config.json` | 保存的配置 |
| `supervoxel2habitat_clustering_strategy.pkl` | 训练的聚类模型（训练模式） |
| `mean_values_of_all_supervoxels_features.csv` | 特征统计 |
| `visualizations/` | 聚类可视化 |

### CSV 文件结构

```csv
subject,supervoxel,habitat,feature_1,feature_2,...,feature_n
sub001,1,2,0.5,0.3,...,0.8
sub001,2,1,0.2,0.7,...,0.4
...
sub002,1,2,0.4,0.5,...,0.6
...
```

---

## 训练 vs 测试模式

### 训练模式

```python
# 训练模式流程
1. 提取特征和超体素聚类
2. 确定最优生境数量
3. 保存聚类模型和统计信息
4. 生成生境图
```

### 测试模式

```python
# 测试模式流程
1. 提取特征和超体素聚类
2. 加载预训练聚类模型
3. 应用模型获取生境标签
4. 生成生境图
```

### 模式对比

| 特性 | 训练模式 | 测试模式 |
|------|----------|----------|
| 聚类模型 | 训练新模型 | 加载预训练模型 |
| 生境数量 | 自动确定 | 使用训练时确定的数量 |
| 模型保存 | 保存模型 | 不保存 |
| 适用场景 | 训练集 | 测试集/新数据 |

---

## 错误处理

### 错误处理策略

```python
def _voxel2supervoxel_clustering(self, subject):
    try:
        # 处理逻辑
        ...
        return subject, mean_features_df
    except Exception as e:
        self.logger.error(f"Error processing {subject}: {e}")
        return subject, Exception(str(e))
```

### 错误处理特点

1. **健壮的文件 I/O**：检查文件是否存在
2. **每步异常处理**：捕获并记录错误
3. **优雅降级**：跳过失败的受试者
4. **详细错误日志**：记录错误上下文

---

## 性能优化

### 优化策略

1. **并行处理**：对 CPU 密集型操作使用多进程
2. **内存高效**：使用 pandas DataFrames
3. **延迟加载**：可选依赖的延迟加载
4. **批处理**：可配置的批处理大小

### 内存管理

```python
# 清理临时变量
del feature_df, raw_df, supervoxel_labels
```

---

## 扩展性

### 插件架构

```python
# 新增聚类算法
class MyClusteringAlgorithm(BaseClustering):
    def fit(self, X):
        # 实现聚类逻辑
        pass
    
    def predict(self, X):
        # 实现预测逻辑
        pass

# 注册算法
register_clustering_algorithm('my_algorithm', MyClusteringAlgorithm)
```

### 自定义特征提取器

```python
class MyFeatureExtractor(BaseFeatureExtractor):
    def extract(self, images_dict, mask):
        # 实现特征提取逻辑
        pass
```

---

## 总结

Habitat Analysis 模块实现了一个灵活、可扩展的肿瘤生境分析系统：

1. **模块化设计**：配置、管道、聚类、特征提取各司其职
2. **策略模式**：训练/测试模式分离
3. **并行处理**：高效利用多核 CPU
4. **可扩展性**：支持自定义聚类算法和特征提取器
5. **健壮性**：完善的错误处理和日志记录

通过理解这些逻辑和流程，您可以更好地使用和扩展这个模块。
