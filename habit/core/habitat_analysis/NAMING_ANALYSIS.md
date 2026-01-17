# Habitat Analysis 模块命名分析与改进建议

## 一、当前命名问题分析

### 1.1 目录命名问题

#### 问题1: `clustering/` vs `clustering_features/` 命名混淆
- **当前命名**:
  - `clustering/` - 聚类算法目录
  - `clustering_features/` - 用于聚类前的特征提取器目录
- **问题**: 
  - `clustering_features` 听起来像是"聚类的特征"（名词），但实际上它是"用于聚类的特征提取器"（功能）
  - 两个目录都包含"clustering"，容易混淆层级关系
- **职责**:
  - `clustering/`: 实现各种聚类算法（KMeans, GMM, DBSCAN等）
  - `clustering_features/`: 从原始图像中提取特征，用于后续聚类（voxel/supervoxel level）

#### 问题2: `habitat_feature_extraction/` 命名不够清晰
- **当前命名**: `habitat_feature_extraction/`
- **问题**:
  - 名称过长，且"habitat_feature_extraction"听起来像是"habitat的特征提取"
  - 实际上它是"从已生成的habitat map中提取特征用于后续分析"
  - 与 `clustering_features/` 的职责区分不够明显
- **职责**:
  - 从已生成的habitat map中提取特征（radiomics, MSI, ITH等）
  - 用于后续的统计分析，而不是用于聚类

#### 问题3: `strategies/` 与 `clustering_pipeline.py` 的关系不清晰
- **当前命名**:
  - `strategies/` - 包含各种策略（TwoStepStrategy, OneStepStrategy等）
  - `strategies/clustering_pipeline.py` - 包含Pipeline类（TrainingPipeline, TestingPipeline）
- **问题**:
  - Pipeline和Strategy的关系不够清晰
  - `clustering_pipeline.py` 中的Pipeline类实际上是Mode（训练/测试模式），而不是Pipeline

### 1.2 类命名问题

#### 问题1: `BaseFeatureExtractor` vs `HabitatFeatureExtractor` 混淆
- **当前命名**:
  - `clustering_features/base_feature_extractor.py` → `BaseFeatureExtractor`
  - `habitat_feature_extraction/extractor.py` → `HabitatFeatureExtractor`
- **问题**:
  - 两个都叫"FeatureExtractor"，但用途完全不同
  - `HabitatFeatureExtractor` 名称不够具体，不能体现它是"从habitat map提取特征"

#### 问题2: `BasePipeline` vs `BaseHabitatStrategy` 职责重叠
- **当前命名**:
  - `BasePipeline` - 在 `strategies/clustering_pipeline.py` 中
  - `BaseHabitatStrategy` - 在 `strategies/base_strategy.py` 中
- **问题**:
  - Pipeline和Strategy的职责区分不够清晰
  - `BasePipeline` 实际上是Mode（训练/测试模式），不是Pipeline

#### 问题3: 聚类算法类命名不一致
- **当前命名**:
  - `KMeansClustering`, `GMMClustering`, `HierarchicalClustering` 等
  - `MySpectralClustering` (使用了"My"前缀，不一致)
- **问题**:
  - `MySpectralClustering` 命名不规范，应该改为 `SpectralClustering`

### 1.3 文件命名问题

#### 问题1: `habitat_feature_extraction/` 中的文件命名
- **当前命名**:
  - `extractor.py` - 原始实现
  - `new_extractor.py` - 重构后的实现
- **问题**:
  - "new"前缀不专业，应该使用版本号或直接替换
  - 两个文件职责重叠

## 二、改进建议

### 2.1 目录重命名方案

#### 方案A: 基于职责的清晰命名（推荐）

```
habitat_analysis/
├── algorithms/                    # 原 clustering/ - 聚类算法
│   ├── base_clustering.py
│   ├── kmeans_clustering.py
│   └── ...
├── extractors/                    # 原 clustering_features/ - 用于聚类的特征提取器
│   ├── base_extractor.py
│   ├── voxel_extractors/          # voxel level提取器
│   └── supervoxel_extractors/     # supervoxel level提取器
├── analyzers/                     # 原 habitat_feature_extraction/ - 从habitat map分析特征
│   ├── base_analyzer.py
│   ├── radiomics_analyzer.py
│   ├── msi_analyzer.py
│   └── ...
├── strategies/                    # 聚类策略（TwoStep, OneStep等）
│   ├── base_strategy.py
│   ├── two_step_strategy.py
│   └── ...
└── modes/                         # 原 strategies/clustering_pipeline.py - 训练/测试模式
    ├── base_mode.py
    ├── training_mode.py
    └── testing_mode.py
```

**优点**:
- 职责清晰：`extractors/` 用于聚类前，`analyzers/` 用于聚类后
- 层级明确：`algorithms/` 是算法，`extractors/` 是提取器，`analyzers/` 是分析器
- 易于理解：命名直接反映功能

#### 方案B: 基于流程阶段的命名

```
habitat_analysis/
├── clustering/                    # 聚类算法（保持不变）
├── pre_clustering/                # 原 clustering_features/ - 聚类前的特征提取
│   └── extractors/
├── post_clustering/               # 原 habitat_feature_extraction/ - 聚类后的特征分析
│   └── analyzers/
├── strategies/                    # 聚类策略
└── modes/                         # 训练/测试模式
```

**优点**:
- 流程清晰：pre_clustering → clustering → post_clustering
- 阶段明确

**缺点**:
- 命名较长
- "pre"和"post"前缀可能不够直观

#### 方案C: 保持现状但优化命名（最小改动）

```
habitat_analysis/
├── clustering/                    # 保持不变
├── feature_extractors/            # 原 clustering_features/ - 更简洁
├── habitat_analyzers/             # 原 habitat_feature_extraction/ - 更清晰
├── strategies/                    # 保持不变
└── modes/                         # 原 strategies/clustering_pipeline.py
```

**优点**:
- 改动最小
- 命名更清晰

### 2.2 类重命名方案

#### Extractors 模块
```python
# 原: clustering_features/base_feature_extractor.py
BaseFeatureExtractor → BaseClusteringExtractor

# 具体实现类保持不变，但可以添加注释说明用途
# RawFeatureExtractor → 用于voxel level特征提取
# MeanVoxelFeaturesExtractor → 用于supervoxel level特征聚合
```

#### Analyzers 模块
```python
# 原: habitat_feature_extraction/extractor.py
HabitatFeatureExtractor → HabitatMapAnalyzer
# 或
HabitatFeatureExtractor → PostClusteringAnalyzer

# 具体实现类
TraditionalRadiomicsExtractor → HabitatRadiomicsAnalyzer
MSIFeatureExtractor → MSIAnalyzer
ITHFeatureExtractor → ITHAnalyzer
```

#### Pipeline/Mode 模块
```python
# 原: strategies/clustering_pipeline.py
BasePipeline → BaseMode
TrainingPipeline → TrainingMode
TestingPipeline → TestingMode

# 原: strategies/base_strategy.py
BaseHabitatStrategy → BaseClusteringStrategy (更具体)
```

#### 聚类算法类
```python
# 修复不一致的命名
MySpectralClustering → SpectralClustering
```

### 2.3 文件重命名方案

#### habitat_feature_extraction/ 模块
```
extractor.py → 删除或标记为deprecated
new_extractor.py → habitat_analyzer.py (主类)
```

## 三、推荐方案（综合）

### 3.1 目录结构（推荐方案A的简化版）

```
habitat_analysis/
├── algorithms/                    # 聚类算法
│   ├── base_clustering.py
│   ├── kmeans_clustering.py
│   ├── gmm_clustering.py
│   └── ...
├── extractors/                    # 用于聚类的特征提取器
│   ├── base_extractor.py          # 原 base_feature_extractor.py
│   ├── voxel_extractors/          # voxel level提取器
│   │   ├── raw_extractor.py
│   │   ├── kinetic_extractor.py
│   │   └── ...
│   └── supervoxel_extractors/     # supervoxel level提取器
│       ├── mean_voxel_extractor.py
│       └── ...
├── analyzers/                     # 从habitat map分析特征
│   ├── base_analyzer.py
│   ├── habitat_analyzer.py        # 主类（原 new_extractor.py）
│   ├── radiomics_analyzer.py
│   ├── msi_analyzer.py
│   └── ith_analyzer.py
├── strategies/                    # 聚类策略
│   ├── base_strategy.py
│   ├── two_step_strategy.py
│   ├── one_step_strategy.py
│   └── direct_pooling_strategy.py
├── modes/                         # 训练/测试模式
│   ├── base_mode.py
│   ├── training_mode.py
│   └── testing_mode.py
├── managers/                      # 管理器（保持不变）
│   ├── feature_manager.py
│   ├── clustering_manager.py
│   └── result_manager.py
├── config.py
└── habitat_analysis.py
```

### 3.2 类命名映射表

| 原命名 | 新命名 | 位置 |
|--------|--------|------|
| `BaseFeatureExtractor` | `BaseClusteringExtractor` | `extractors/base_extractor.py` |
| `HabitatFeatureExtractor` | `HabitatMapAnalyzer` | `analyzers/habitat_analyzer.py` |
| `BasePipeline` | `BaseMode` | `modes/base_mode.py` |
| `TrainingPipeline` | `TrainingMode` | `modes/training_mode.py` |
| `TestingPipeline` | `TestingMode` | `modes/testing_mode.py` |
| `BaseHabitatStrategy` | `BaseClusteringStrategy` | `strategies/base_strategy.py` |
| `MySpectralClustering` | `SpectralClustering` | `algorithms/spectral_clustering.py` |
| `TraditionalRadiomicsExtractor` | `HabitatRadiomicsAnalyzer` | `analyzers/radiomics_analyzer.py` |
| `MSIFeatureExtractor` | `MSIAnalyzer` | `analyzers/msi_analyzer.py` |
| `ITHFeatureExtractor` | `ITHAnalyzer` | `analyzers/ith_analyzer.py` |

### 3.3 命名原则总结

1. **职责清晰**: 目录和类名直接反映其功能
   - `extractors/` - 提取特征用于聚类
   - `analyzers/` - 分析已生成的habitat map
   - `algorithms/` - 聚类算法实现
   - `strategies/` - 聚类策略（TwoStep, OneStep等）
   - `modes/` - 运行模式（Training, Testing）

2. **层级明确**: 避免命名混淆
   - 不使用"clustering_features"这种容易误解的复合词
   - 使用单数形式表示功能类别（extractor, analyzer, algorithm）

3. **一致性**: 同类组件使用统一的命名模式
   - 所有聚类算法: `*Clustering`
   - 所有提取器: `*Extractor` (用于聚类前)
   - 所有分析器: `*Analyzer` (用于聚类后)
   - 所有策略: `*Strategy`
   - 所有模式: `*Mode`

4. **简洁性**: 避免冗余前缀
   - 不使用"My"、"New"等临时性前缀
   - 类名直接反映功能，不需要额外说明

## 四、迁移计划

### 阶段1: 目录重命名（向后兼容）
1. 创建新目录结构
2. 移动文件并更新导入
3. 在旧位置创建符号链接或导入重定向（保持向后兼容）

### 阶段2: 类重命名（向后兼容）
1. 创建新类名，旧类名作为别名
2. 更新所有内部引用
3. 添加deprecation警告

### 阶段3: 清理（可选）
1. 移除旧命名
2. 更新文档和示例

## 五、总结

主要问题：
1. `clustering_features/` 命名容易误解
2. `habitat_feature_extraction/` 命名不够清晰
3. `BasePipeline` 实际上是Mode，不是Pipeline
4. `BaseFeatureExtractor` 和 `HabitatFeatureExtractor` 容易混淆

推荐方案：
- `clustering/` → `algorithms/`
- `clustering_features/` → `extractors/`
- `habitat_feature_extraction/` → `analyzers/`
- `strategies/clustering_pipeline.py` → `modes/`
- 类名相应调整以反映新职责
