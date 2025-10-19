# Habitat Analysis 使用指南

## 概述

Habitat Analysis 模块用于识别和表征肿瘤内部具有不同影像表型的亚区域（"生境"）。该模块支持两种聚类策略：

### 🎯 聚类模式对比

| 特性 | 一步法 (One-Step) | 二步法 (Two-Step) |
|------|------------------|------------------|
| **聚类流程** | 直接从体素聚类到生境 | 先聚类到supervoxels，再聚类到habitats |
| **聚类层级** | 单层级（个体水平） | 双层级（个体+群体水平） |
| **聚类数目** | 每个肿瘤自动确定最优数目 | supervoxels数目固定，habitats数目可优化 |
| **生境标签** | 每个患者的生境标签独立 | 所有患者共享统一的生境标签体系 |
| **跨患者比较** | 需要额外的对应分析 | 可直接比较相同编号的生境 |
| **计算复杂度** | 较低（仅个体聚类） | 较高（个体+群体两次聚类） |
| **适用场景** | 个体异质性分析、小样本研究 | 队列研究、跨患者生境模式识别 |

---

## 🚀 快速开始

### 使用CLI（推荐）

```bash
# 二步法（默认）
habit habitat --config config/config_getting_habitat.yaml

# 一步法
# 先修改配置文件中的 clustering_mode: one_step
habit habitat --config config/config_getting_habitat.yaml
```

### 使用传统脚本

```bash
python scripts/app_getting_habitat_map.py --config config/config_getting_habitat.yaml
```

---

## 📋 配置文件说明

**📖 配置文件链接**：
- 📄 [当前配置文件](../config/config_getting_habitat.yaml) - 实际使用的精简配置
- 📖 [详细注释模板](../config_templates/config_getting_habitat_annotated.yaml) - 包含完整英文注释和使用说明

### 关键配置项

```yaml
HabitatsSegmention:
  # 聚类策略选择
  clustering_mode: two_step  # one_step 或 two_step
  
  # 第一步：个体水平聚类
  supervoxel:
    algorithm: kmeans  # 聚类算法：kmeans 或 gmm
    n_clusters: 50     # 二步法的固定聚类数
    
    # 一步法专用设置
    one_step_settings:
      min_clusters: 2               # 最小聚类数
      max_clusters: 10              # 最大聚类数
      selection_method: silhouette  # 评估方法
      plot_validation_curves: true  # 是否绘制验证曲线
  
  # 第二步：群体水平聚类（仅在two_step模式使用）
  habitat:
    mode: training  # training 或 testing
    algorithm: kmeans
    max_clusters: 10
    habitat_cluster_selection_method: inertia
    best_n_clusters: 4  # 指定聚类数，或设为null自动选择
```

---

## 🎨 一步法详解

### 工作原理

1. **体素特征提取**: 计算每个体素的组学特征
2. **个体聚类**: 对每个患者的肿瘤单独聚类
3. **自动选择聚类数**: 使用验证指标（如轮廓系数）确定最佳聚类数
4. **生成个性化生境图**: 每个患者获得独特的生境分割

### 聚类数选择方法

| 方法 | 说明 | 优化方向 |
|------|------|---------|
| `silhouette` | 轮廓系数，衡量聚类紧密度和分离度 | 越大越好 |
| `calinski_harabasz` | 方差比率，类间/类内方差 | 越大越好 |
| `davies_bouldin` | 簇间平均相似度 | 越小越好 |
| `inertia` | 簇内平方和 | 越小越好 |

### 输出文件

```
output_dir/
├── {subject}_supervoxel.nrrd           # 生境地图（每个患者）
├── {subject}_validation_plots/         # 验证曲线（如果启用）
│   └── {subject}_cluster_validation.png
├── results_all_samples.csv             # 所有患者的聚类结果
└── clustering_summary.csv              # 聚类摘要统计
```

### 示例配置（一步法）

```yaml
HabitatsSegmention:
  clustering_mode: one_step
  
  supervoxel:
    algorithm: kmeans
    random_state: 42
    
    one_step_settings:
      min_clusters: 3              # 测试3-8个聚类
      max_clusters: 8
      selection_method: silhouette  # 使用轮廓系数
      plot_validation_curves: true  # 绘制每个患者的验证曲线
```

---

## 📊 二步法详解

### 工作原理

1. **体素→超体素**: 每个患者的肿瘤聚类为supervoxels
2. **超体素→生境**: 跨患者聚类，识别共通的生境模式
3. **群体一致性**: 所有患者使用相同的生境定义

### 优势

- ✅ 跨患者可比较性
- ✅ 识别共通模式
- ✅ 适合队列研究
- ✅ 便于统计分析

### 输出文件

```
output_dir/
├── {subject}_supervoxel.nrrd              # 超体素地图
├── {subject}_habitat.nrrd                 # 生境地图
├── mean_values_of_all_supervoxels_features.csv  # 超体素特征均值
├── results_all_samples.csv                # 最终结果
├── supervoxel2habitat_clustering_model.pkl  # 聚类模型
└── habitat_clustering_scores.png          # 聚类评估曲线
```

### 示例配置（二步法）

```yaml
HabitatsSegmention:
  clustering_mode: two_step
  
  supervoxel:
    algorithm: kmeans
    n_clusters: 50  # 每个患者固定50个supervoxels
    random_state: 42
  
  habitat:
    mode: training
    algorithm: kmeans
    max_clusters: 10
    habitat_cluster_selection_method: silhouette
    best_n_clusters: null  # 自动选择
```

---

## 🔧 高级用法

### 使用预训练模型（二步法）

对于新的测试数据，可以使用之前训练的模型：

```yaml
habitat:
  mode: testing  # 切换到测试模式
  # 模型会自动从 out_dir/supervoxel2habitat_clustering_model.pkl 加载
```

### 多进程加速

```yaml
processes: 4  # 使用4个进程并行处理
```

### 自定义特征提取 (Custom Feature Extraction)

`HABIT` 提供了一个灵活的特征提取框架，允许您在体素（Voxel）层面组合使用多种特征，用于后续的生境（Habitat）聚类分析。所有特征相关的配置都在配置文件的 `FeatureConstruction` 部分完成。

### 语法简介

特征提取的语法被设计为一种表达式，格式为 `method(arguments)`。

- **单特征**: 直接使用一种方法，如 `raw(pre_contrast)`。
- **多特征组合**: 将多种特征作为另一个特征提取器（如 `concat`）的输入，例如 `concat(raw(pre_contrast), gabor(pre_contrast))`。
- **跨模态特征**: 一些特征提取器（如 `kinetic`）可以接受多个不同模态的图像作为输入。

所有在方法中使用的参数（如 `timestamps` 或 `params_file`）都必须在 `params` 字段中定义。

### 体素级特征 (`voxel_level`)

这是生境分析的第一步，用于从原始图像中为每个体素提取一个或多个特征值，形成特征向量。

以下是所有可用的体素级特征提取方法：

| 方法 (`method`) | 功能描述 | 主要参数 |
| :--- | :--- | :--- |
| `raw` | **原始强度 (Raw Intensity)**<br>直接提取每个体素在指定图像中的原始信号强度值。这是最基础、最直接的特征。 | 无 |
| `kinetic` | **动力学特征 (Kinetic Features)**<br>从一个时间序列的图像（如多期相增强扫描）中计算动力学特征，例如灌注速率（wash-in/wash-out slope）。 | `timestamps`: 一个指向 `.xlsx` 文件的路径。该文件需包含每个受试者 (subject) 对应的多期相扫描的时间点。 |
| `voxel_radiomics` | **体素级组学 (Voxel-level Radiomics)**<br>使用 PyRadiomics 在每个体素的局部邻域内计算影像组学特征。可提取纹理（如GLCM, Gabor）、强度分布等多种高级特征。 | `params_file`: 指向一个 PyRadiomics 的 `.yaml` 参数文件，用于精确控制要提取的特征类别、滤波器及相关设置。 |
| `local_entropy` | **局部熵 (Local Entropy)**<br>计算每个体素邻域内的局部信息熵。这是一个衡量局部区域纹理复杂度和随机性的指标。 | `kernel_size`: (可选, 默认值: 3) 定义计算熵的邻域大小，如 `3` 代表 3x3x3 的立方体。<br>`bins`: (可选, 默认值: 32) 计算直方图时的分箱数。 |

---

### 示例配置

下面是一些具体的 `yaml` 配置示例，展示了如何使用这些方法。

#### 示例 1: 使用单一原始图像强度

这是最简单的配置，仅使用 `pre_contrast` 图像的原始像素值作为特征。

```yaml
FeatureConstruction:
  voxel_level:
    method: 'raw(pre_contrast)'
    image_names: ['pre_contrast']
    params: {}
```

#### 示例 2: 组合使用多种原始图像强度

您可以将多个模态的原始强度值拼接成一个特征向量。这里我们使用 `concat` 方法来组合三个不同期相的图像。

```yaml
FeatureConstruction:
  voxel_level:
    method: 'concat(raw(pre_contrast), raw(LAP), raw(PVP))'
    image_names: ['pre_contrast', 'LAP', 'PVP']
    params: {}
```

#### 示例 3: 计算动力学特征

使用 `kinetic` 方法需要提供一个 `timestamps` 文件。该方法会自动处理 `raw()` 包装的多个图像。

```yaml
FeatureConstruction:
  voxel_level:
    method: 'kinetic(raw(pre_contrast), raw(LAP), raw(PVP), raw(delay_3min), timestamps)'
    image_names: ['pre_contrast', 'LAP', 'PVP', 'delay_3min']
    params:
      timestamps: './config/scan_times.xlsx' # 指向您的时间戳文件
```

#### 示例 4: 提取体素级组学特征

使用 `voxel_radiomics` 需要提供一个 PyRadiomics 参数文件。

```yaml
FeatureConstruction:
  voxel_level:
    method: 'voxel_radiomics(pre_contrast, radiomics_params)'
    image_names: ['pre_contrast']
    params:
      radiomics_params: './config/radiomics_params.yaml' # 指向您的组学参数文件
```

#### 示例 5: 提取局部熵特征

使用 `local_entropy` 并自定义邻域大小和分箱数。

```yaml
FeatureConstruction:
  voxel_level:
    method: 'local_entropy(pre_contrast, kernel_size, bins)'
    image_names: ['pre_contrast']
    params:
      kernel_size: 5
      bins: 64
```

---

## 🎯 使用建议

### 选择一步法当...

✅ 关注个体肿瘤的异质性  
✅ 不需要跨患者比较  
✅ 每个患者样本量充足（足够体素数）  
✅ 探索性研究，了解个体差异  

### 选择二步法当...

✅ 需要跨患者统计分析  
✅ 识别群体共通的生境类型  
✅ 建立可复用的生境模型  
✅ 进行队列研究或临床预测  

---

## 🐛 常见问题

### Q1: 一步法中每个患者的聚类数都不同，如何比较？

**A**: 一步法关注的是个体内的异质性，不是跨个体比较。如果需要比较，应该：
- 比较聚类数量（作为异质性指标）
- 提取每个生境的特征进行统计
- 使用二步法获得统一的生境定义

### Q2: 如何选择合适的聚类数范围？

**A**: 建议：
- 最小值：2-3（至少要有明显分类）
- 最大值：10-15（避免过度分割）
- 考虑肿瘤大小（小肿瘤用较少聚类数）

### Q3: 验证曲线看起来不稳定怎么办？

**A**: 可能原因：
- 样本量不足（体素太少）
- 特征选择不合适
- 尝试不同的validation method
- 增加聚类算法的 `n_init` 参数

---

## 📚 相关文档

- [特征提取配置](./app_extracting_habitat_features.md)
- [ICC可重复性分析](./app_icc_analysis.md)
- [CLI使用指南](../HABIT_CLI.md)

---

## 📖 参考文献

**二步法（经典Habitat方法）**:
- Wu J, et al. "Intratumoral spatial heterogeneity at perfusion MR imaging predicts recurrence-free survival in locally advanced breast cancer treated with neoadjuvant chemotherapy." Radiology, 2018.

**一步法（个性化分析）**:
- Nomogram for Predicting Neoadjuvant Chemotherapy  Response in Breast Cancer Using MRI-based Intratumoral  Heterogeneity Quantification

---

*最后更新: 2025-10-19*

