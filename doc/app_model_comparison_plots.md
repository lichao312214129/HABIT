# app_model_comparison_plots.py 功能文档

## 功能概述

`app_model_comparison_plots.py` 是HABIT工具包中用于生成机器学习模型比较和评估图表的专用工具。它支持多种可视化方法，可用于比较不同模型的性能、特征重要性、决策边界等，帮助研究人员理解和解释模型结果。

## 用法

```bash
python scripts/app_model_comparison_plots.py --config <config_file_path>
```

## 命令行参数

| 参数 | 描述 |
|-----|-----|
| `--config` | YAML配置文件路径（必需） |

## 配置文件格式

`app_model_comparison_plots.py` 使用YAML格式的配置文件，包含以下主要部分：

### 基本配置

```yaml
# 数据和输出路径
input: <输入数据文件路径>
results_path: <模型结果目录>
output: <图表输出目录>
label: <标签列名>
random_seed: <随机种子>
```

### 模型配置

```yaml
models:
  - name: <模型名称1>
    path: <模型文件路径1>
  - name: <模型名称2>
    path: <模型文件路径2>
  ...
```

### 绘图配置

```yaml
plots:
  performance_comparison:
    enabled: <是否启用>
    metrics: <指标列表>
    
  roc_curves:
    enabled: <是否启用>
    
  pr_curves:
    enabled: <是否启用>
    
  confusion_matrices:
    enabled: <是否启用>
    normalize: <是否归一化>
    
  feature_importance:
    enabled: <是否启用>
    n_features: <显示的特征数量>
    
  decision_boundaries:
    enabled: <是否启用>
    feature_pairs: <特征对列表>
    resolution: <决策边界分辨率>
    
  calibration_curves:
    enabled: <是否启用>
    n_bins: <分箱数量>
    
  learning_curves:
    enabled: <是否启用>
    train_sizes: <训练集大小列表>
    
  dimension_reduction:
    enabled: <是否启用>
    method: <降维方法>
    n_components: <组件数量>
    
  shap_analysis:
    enabled: <是否启用>
    max_display: <最大显示特征数>
    plot_types: <SHAP图表类型列表>
```

## 支持的绘图功能

### 1. 性能比较图 (performance_comparison)

生成不同模型的各项评估指标对比图，支持条形图和雷达图。

**参数：**
- `metrics`: 要比较的评估指标列表
- `plot_type`: 图表类型，可选 "bar" 或 "radar"
- `sort_by`: 用于排序的指标

### 2. ROC曲线 (roc_curves)

绘制所有模型的ROC曲线（受试者工作特征曲线）。

**参数：**
- `micro_average`: 是否计算微平均
- `macro_average`: 是否计算宏平均
- `conf_intervals`: 是否显示置信区间
- `n_bootstraps`: Bootstrap重采样次数

### 3. PR曲线 (pr_curves)

绘制所有模型的精确率-召回率曲线。

**参数：**
- `micro_average`: 是否计算微平均
- `macro_average`: 是否计算宏平均
- `baseline`: 是否显示基线

### 4. 混淆矩阵 (confusion_matrices)

生成每个模型的混淆矩阵可视化。

**参数：**
- `normalize`: 是否归一化混淆矩阵
- `colormap`: 颜色映射
- `include_values`: 是否在单元格中显示数值

### 5. 特征重要性图 (feature_importance)

显示每个模型的特征重要性排名。

**参数：**
- `n_features`: 显示的特征数量
- `sort`: 是否对特征进行排序
- `method`: 计算特征重要性的方法，可选 "builtin", "permutation", "shap"

### 6. 决策边界图 (decision_boundaries)

绘制二维特征空间中的模型决策边界。

**参数：**
- `feature_pairs`: 要可视化的特征对列表
- `resolution`: 决策边界的分辨率
- `colormap`: 颜色映射

### 7. 校准曲线 (calibration_curves)

检查分类模型概率校准的可靠性曲线。

**参数：**
- `n_bins`: 分箱数量
- `strategy`: 分箱策略，可选 "uniform", "quantile"

### 8. 学习曲线 (learning_curves)

显示模型性能随训练数据量变化的趋势。

**参数：**
- `train_sizes`: 训练集大小列表或比例
- `scoring`: 评分方法
- `n_jobs`: 并行任务数量

### 9. 降维可视化 (dimension_reduction)

将高维特征空间降维并可视化样本分布。

**参数：**
- `method`: 降维方法，可选 "pca", "tsne", "umap"
- `n_components`: 降维后的维度数
- `perplexity`: t-SNE的困惑度参数
- `n_neighbors`: UMAP的邻居数量

### 10. SHAP分析图 (shap_analysis)

使用SHAP值解释模型预测，提供多种可视化方式。

**参数：**
- `max_display`: 最大显示特征数
- `plot_types`: SHAP图表类型列表，可选 "summary", "bar", "beeswarm", "waterfall", "force", "decision"
- `sample_idx`: 用于局部解释的样本索引

## 完整配置示例

```yaml
# 基本配置
input: ./data/radiomics_features.csv
results_path: ./results/models
output: ./results/visualization
label: cancer_type
random_seed: 42

# 模型配置
models:
  - name: LogisticRegression
    path: ./results/models/logistic_regression.pkl
  - name: RandomForest
    path: ./results/models/random_forest.pkl
  - name: XGBoost
    path: ./results/models/xgboost.pkl

# 绘图配置
plots:
  performance_comparison:
    enabled: true
    metrics: ["accuracy", "precision", "recall", "f1", "roc_auc"]
    plot_type: "radar"
    
  roc_curves:
    enabled: true
    micro_average: true
    macro_average: true
    conf_intervals: true
    n_bootstraps: 1000
    
  pr_curves:
    enabled: true
    micro_average: true
    macro_average: true
    baseline: true
    
  confusion_matrices:
    enabled: true
    normalize: true
    colormap: "Blues"
    include_values: true
    
  feature_importance:
    enabled: true
    n_features: 20
    sort: true
    method: "permutation"
    
  decision_boundaries:
    enabled: true
    feature_pairs: [["feature1", "feature2"], ["feature3", "feature4"]]
    resolution: 100
    colormap: "RdBu"
    
  calibration_curves:
    enabled: true
    n_bins: 10
    strategy: "uniform"
    
  learning_curves:
    enabled: true
    train_sizes: [0.1, 0.3, 0.5, 0.7, 0.9]
    scoring: "accuracy"
    n_jobs: -1
    
  dimension_reduction:
    enabled: true
    method: "tsne"
    n_components: 2
    perplexity: 30
    
  shap_analysis:
    enabled: true
    max_display: 15
    plot_types: ["summary", "beeswarm", "waterfall"]
    sample_idx: 0
```

## 执行流程

1. 加载配置文件和数据
2. 加载指定的模型文件
3. 根据配置生成性能评估和比较图表
4. 保存所有图表到输出目录

## 输出结果

程序执行后，将在指定的输出目录生成以下内容：

1. `performance_comparison/`: 性能比较图
2. `roc_curves/`: ROC曲线图
3. `pr_curves/`: PR曲线图
4. `confusion_matrices/`: 混淆矩阵可视化
5. `feature_importance/`: 特征重要性图
6. `decision_boundaries/`: 决策边界图
7. `calibration_curves/`: 校准曲线图
8. `learning_curves/`: 学习曲线图
9. `dimension_reduction/`: 降维可视化
10. `shap_analysis/`: SHAP分析图

每个子目录中包含对应类型的图表文件（PNG和PDF格式）。

## 自定义主题和样式

配置文件中支持设置图表主题和样式：

```yaml
visualization:
  theme: <主题名称>  # 如 "default", "dark", "light", "pastel", "journal"
  figsize: [<宽>, <高>]  # 默认图表尺寸
  dpi: <分辨率>  # 图表分辨率
  font_family: <字体>  # 如 "Arial", "Times New Roman"
  colorblind_friendly: <是否启用色盲友好模式>
```

## 注意事项

1. 确保模型文件路径正确
2. 对于大数据集，降维可视化可能需要较长时间
3. SHAP分析对计算资源要求较高
4. 决策边界图仅适用于二维特征空间
5. 某些图表类型可能仅适用于特定类型的模型（如分类或回归） 