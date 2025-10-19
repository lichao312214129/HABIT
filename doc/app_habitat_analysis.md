# Habitat Analysis 使用指南

## 概述

Habitat Analysis 模块用于识别和表征肿瘤内部具有不同影像表型的亚区域（"生境"）。该模块支持两种聚类策略：

### 🎯 聚类模式对比

| 特性 | 一步法 (One-Step) | 二步法 (Two-Step) |
|------|------------------|------------------|
| **聚类层级** | 仅个体水平 | 个体 + 群体水平 |
| **适用场景** | 个性化肿瘤分析 | 跨患者生境识别 |
| **聚类数确定** | 每个肿瘤自动确定 | 在群体水平统一确定 |
| **结果一致性** | 每个患者可能不同 | 所有患者使用相同生境 |
| **文献参考** | 近期个性化研究 | 经典Habitat方法 |

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

### 自定义特征提取

```yaml
FeatureConstruction:
  voxel_level:
    # 使用动力学特征
    method: kinetic(raw(pre_contrast), raw(LAP), raw(PVP), timestamps)
    params:
      timestamps: ./scan_times.xlsx
  
  # 个体水平预处理（可选）
  preprocessing_for_subject_level:
    methods:
      - method: winsorize
        winsor_limits: [0.05, 0.05]
      - method: minmax
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
- Zhou M, et al. "Radiomics in Brain Tumor: Image Assessment, Quantitative Feature Descriptors, and Machine-Learning Approaches." AJNR, 2018.

**一步法（个性化分析）**:
- 近期多项研究采用基于个体的聚类方法进行肿瘤异质性分析
- 适用于精准医疗和个性化治疗研究

---

*最后更新: 2025-10-19*

