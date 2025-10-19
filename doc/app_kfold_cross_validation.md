# K-Fold交叉验证使用指南

本文档介绍如何使用habit软件包中的K-Fold交叉验证功能进行模型评估和训练。

## 概述

K-Fold交叉验证是一种常用的模型评估方法，它将数据集分成K个子集（fold），每次使用其中K-1个子集进行训练，剩余1个子集进行验证，这样重复K次，每个子集都会被用作验证集一次。

### K-Fold交叉验证的优势

1. **更可靠的性能评估**: 相比单次训练-测试划分，K-Fold交叉验证能提供更稳定和可靠的性能估计
2. **充分利用数据**: 每个样本都会被用于训练和验证，特别适合小样本数据集
3. **减少偶然性**: 通过多次验证减少单次划分的偶然性影响
4. **评估模型稳定性**: 可以通过标准差评估模型在不同数据子集上的稳定性

### 与标准建模流程的区别

| 特性 | 标准建模流程 | K-Fold交叉验证 |
|------|-------------|---------------|
| 数据划分 | 一次划分（训练集+测试集） | K次划分 |
| 训练次数 | 每个模型训练1次 | 每个模型训练K次 |
| 评估指标 | 单个测试集结果 | K个验证集结果的平均值±标准差 |
| 特征选择 | 在整个训练集上进行 | 在每个fold内独立进行（避免数据泄露） |
| 适用场景 | 数据量大、需要快速原型 | 数据量小、需要可靠评估 |
| 计算时间 | 快 | 慢（K倍） |

## 配置文件

K-Fold交叉验证使用专门的配置文件 `config_machine_learning_kfold.yaml`。

### 配置文件示例

```yaml
# Data Input Configuration
input:
  - path: ./ml_data/breast_cancer_dataset.csv
    name: clinical_
    subject_id_col: subjID
    label_col: label
    features:

# Output Directory Configuration
output: ./ml_data/kfold_results

# K-Fold Cross-Validation Configuration
n_splits: 5          # Number of folds (commonly 5 or 10)
stratified: true     # Whether to use stratified k-fold (recommended for imbalanced data)
random_state: 42     # Random seed for reproducibility

# Normalization Configuration
normalization:
  method: z_score    # Normalization method

# Feature Selection Configuration
# Note: Feature selection is performed within each fold to avoid data leakage
feature_selection_methods:
  - method: correlation
    params:
      threshold: 0.80
      method: spearman
      visualize: false
      before_z_score: false

# Model Configuration
models:
  LogisticRegression:
    params:
      random_state: 42
      max_iter: 1000
      C: 1.0
  
  RandomForest:
    params:
      random_state: 42
      n_estimators: 100
  
  XGBoost:
    params:
      random_state: 42
      n_estimators: 100
      max_depth: 3

# Visualization and Saving Configuration
is_visualize: false
is_save_model: false
```

### 重要参数说明

#### n_splits
- **描述**: K-Fold中的fold数量
- **常用值**: 5或10
- **选择建议**:
  - 数据量小（<100样本）: 使用10-fold或更多
  - 数据量中等（100-1000样本）: 使用5-fold或10-fold
  - 数据量大（>1000样本）: 使用5-fold
  - 极小样本: 考虑Leave-One-Out CV（n_splits = 样本数）

#### stratified
- **描述**: 是否使用分层K-Fold
- **默认值**: true
- **建议**:
  - 类别不平衡问题：必须使用true
  - 类别平衡问题：建议使用true
  - 回归问题：设为false

## 使用方法

### 1. 命令行使用

创建一个Python脚本（例如 `run_kfold_cv.py`）:

```python
from habit.core.machine_learning.machine_learning_kfold import run_kfold_modeling
import yaml

# Load configuration
with open('config/config_machine_learning_kfold.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Run k-fold cross-validation
modeling = run_kfold_modeling(config)

print("K-Fold cross-validation completed!")
```

然后运行：
```bash
python run_kfold_cv.py
```

### 2. 交互式使用

```python
from habit.core.machine_learning.machine_learning_kfold import ModelingKFold
import yaml

# Load configuration
with open('config/config_machine_learning_kfold.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create modeling instance
modeling = ModelingKFold(config)

# Run pipeline step by step
modeling.read_data()
modeling.preprocess_data()
modeling._create_kfold_splits()
modeling.run_kfold_cv()

# Access results
print("Aggregated results:", modeling.cv_results['aggregated'])
```

## 输出结果

### 1. 控制台输出

运行过程中会显示：
- 每个fold的处理进度
- 每个fold中每个模型的性能指标
- 所有fold的聚合结果

示例输出：
```
================================================================================
Processing Fold 1/5
================================================================================
Selected 25 features for fold 1

Training LogisticRegression on fold 1...
LogisticRegression - Val AUC: 0.8523, Acc: 0.7800

Training RandomForest on fold 1...
RandomForest - Val AUC: 0.8912, Acc: 0.8200

...

LogisticRegression - Overall AUC: 0.8456 ± 0.0234
RandomForest - Overall AUC: 0.8834 ± 0.0189
```

### 2. 结果文件

#### kfold_cv_results.json
包含完整的交叉验证结果：
```json
{
  "n_splits": 5,
  "stratified": true,
  "aggregated": {
    "LogisticRegression": {
      "fold_metrics": {
        "auc_mean": 0.8456,
        "auc_std": 0.0234,
        "accuracy_mean": 0.7820,
        "accuracy_std": 0.0156
      },
      "overall_metrics": {
        "auc": 0.8467,
        "accuracy": 0.7830,
        "sensitivity": 0.8100,
        "specificity": 0.7560
      }
    }
  }
}
```

#### kfold_performance_summary.csv
性能汇总表格：

| Model | AUC_mean | AUC_std | AUC_overall | Accuracy_mean | Accuracy_std | Accuracy_overall |
|-------|----------|---------|-------------|---------------|--------------|------------------|
| LogisticRegression | 0.8456 | 0.0234 | 0.8467 | 0.7820 | 0.0156 | 0.7830 |
| RandomForest | 0.8834 | 0.0189 | 0.8845 | 0.8230 | 0.0123 | 0.8240 |

### 3. Fold级别输出

每个fold会在输出目录下创建子目录：
```
kfold_results/
├── fold_1/
│   └── feature_selection/
│       └── correlation_heatmap.pdf
├── fold_2/
│   └── feature_selection/
├── ...
├── kfold_cv_results.json
└── kfold_performance_summary.csv
```

## 结果解读

### 性能指标解释

1. **xxx_mean**: 各fold上的平均值
   - 表示模型在不同数据子集上的平均性能
   
2. **xxx_std**: 各fold上的标准差
   - 表示模型性能的稳定性
   - 标准差越小，模型越稳定
   
3. **xxx_overall**: 所有fold预测的整体指标
   - 将所有fold的预测结果合并后计算
   - 通常比平均值更准确

### 如何选择模型

1. **优先考虑overall指标**: 
   - Overall AUC最高的模型通常是最佳选择

2. **考虑稳定性**:
   - 标准差较小的模型更可靠
   - 例如: 模型A (AUC=0.85±0.02) 优于 模型B (AUC=0.86±0.08)

3. **权衡性能和复杂度**:
   - 如果两个模型性能接近，选择更简单的模型
   - 简单模型更容易解释和部署

## 避免数据泄露

K-Fold交叉验证中的一个关键问题是**数据泄露**。本实现通过以下方式避免数据泄露：

### 1. Fold内特征选择

**错误做法**（会导致数据泄露）：
```python
# 在所有数据上进行特征选择
selected_features = feature_selection(X, y)

# 然后进行k-fold交叉验证
for train_idx, val_idx in kfold.split(X):
    X_train = X[train_idx][selected_features]
    X_val = X[val_idx][selected_features]
    # 训练和评估...
```

**正确做法**（本实现采用的方法）：
```python
# 在每个fold内独立进行特征选择
for train_idx, val_idx in kfold.split(X):
    X_train = X[train_idx]
    X_val = X[val_idx]
    
    # 只在训练集上选择特征
    selected_features = feature_selection(X_train, y_train)
    
    # 应用到训练集和验证集
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    # 训练和评估...
```

### 2. Fold内数据标准化

标准化也必须在每个fold内独立进行：

```python
for train_idx, val_idx in kfold.split(X):
    # 在训练集上拟合scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 用训练集的scaler转换验证集
    X_val_scaled = scaler.transform(X_val)
```

## 最佳实践

### 1. 选择合适的K值

```python
# 小数据集（< 100样本）
config['n_splits'] = 10  # 或更多

# 中等数据集（100-1000样本）
config['n_splits'] = 5   # 或10

# 大数据集（> 1000样本）
config['n_splits'] = 5
```

### 2. 使用分层K-Fold

对于分类问题，特别是类别不平衡的情况：
```yaml
stratified: true  # 保持每个fold中的类别比例
```

### 3. 固定随机种子

确保结果可重复：
```yaml
random_state: 42  # 固定随机种子
```

### 4. 合理选择模型数量

- 不要同时训练太多模型（会很慢）
- 建议每次训练3-5个模型进行比较

### 5. 特征选择策略

```yaml
feature_selection_methods:
  # 先去除高度相关的特征
  - method: correlation
    params:
      threshold: 0.90
      
  # 再进行重要性筛选
  - method: anova
    params:
      n_features_to_select: 20
```

## 与标准流程的选择

### 何时使用K-Fold交叉验证

1. **小样本数据集**（< 500样本）
2. **需要可靠的性能评估**
3. **模型选择和超参数调优**
4. **发表论文需要严谨评估**
5. **数据收集成本高**

### 何时使用标准流程

1. **大数据集**（> 10000样本）
2. **快速原型开发**
3. **计算资源有限**
4. **已经确定模型和参数**
5. **实时预测系统**

## 常见问题

### Q: K-Fold交叉验证的计算时间是多少？
A: 约为标准流程的K倍（例如5-fold需要5倍时间）。可以通过以下方式加速：
   - 减少模型数量
   - 减少特征选择方法
   - 使用更简单的模型
   - 减少K值

### Q: 每个fold选出的特征会不同吗？
A: 是的，这是正常的。每个fold的训练集略有不同，因此特征选择结果也会有差异。最终特征重要性应该综合所有fold的结果。

### Q: 如何保存K-Fold交叉验证的模型？
A: K-Fold主要用于评估，通常不保存模型。如果需要最终模型，建议：
   1. 使用K-Fold评估选出最佳模型和参数
   2. 在全部数据上重新训练该模型
   3. 保存重新训练的模型

### Q: 标准差很大说明什么？
A: 标准差大表示：
   - 模型在不同数据子集上性能不稳定
   - 可能对数据分布敏感
   - 需要考虑：
     - 增加样本量
     - 使用更稳定的模型
     - 改进特征工程
     - 调整超参数

### Q: K-Fold与留一法(Leave-One-Out)的区别？
A: 
   - Leave-One-Out是K-Fold的特例（K=样本数）
   - 优点：充分利用数据
   - 缺点：计算量极大
   - 建议：仅在极小样本（<30）时使用

## 与模型比较工具的兼容性

K-Fold交叉验证的结果可以直接用于模型比较分析，无需额外的格式转换。

### 输出文件

运行K-Fold交叉验证后，会自动生成以下文件：

#### 必需输出

1. **kfold_cv_results.json** - 详细的交叉验证结果
2. **kfold_performance_summary.csv** - 性能摘要表
3. **all_prediction_results.csv** - 兼容格式的预测结果（可用于模型比较）

#### 可视化输出（当 `is_visualize: true` 时）

4. **kfold_roc_curves.pdf** - ROC曲线对比图
5. **kfold_calibration_curves.pdf** - 校准曲线
6. **kfold_dca_curves.pdf** - 决策曲线分析（DCA）
7. **kfold_confusion_matrix_{模型名}.pdf** - 各模型的混淆矩阵

**配置可视化**：
```yaml
# 在配置文件中设置
is_visualize: true  # 启用可视化（默认为 false）
is_save_model: false  # 是否保存每个fold的模型
```

### 预测结果格式

`all_prediction_results.csv` 文件格式与标准机器学习流程完全兼容：

```csv
subject_id,true_label,split,RandomForest_pred,RandomForest_prob,XGBoost_pred,XGBoost_prob
patient_001,1,Test set,1,0.85,1,0.88
patient_002,0,Test set,0,0.23,0,0.19
...
```

**注意**: 
- `split` 列标记为 "Test set"，表示这些是验证集的预测结果
- 在K-Fold中，每个样本都会在某个fold中作为验证集被预测一次
- 因此所有样本的 `split` 都是 "Test set"
- 如果启用了 `is_visualize: true`，还会自动生成 ROC、DCA、校准曲线和混淆矩阵等可视化图表
- 可视化图表基于所有 fold 的聚合预测结果生成，能够全面反映模型在整个数据集上的表现

### 使用模型比较工具

运行完K-Fold交叉验证后，可以直接使用模型比较工具：

```bash
# 1. 运行K-Fold交叉验证
python -m habit kfold -c config/config_machine_learning_kfold.yaml

# 2. 使用比较工具分析结果
python -m habit compare -c config/config_model_comparison.yaml
```

在 `config_model_comparison.yaml` 中配置：

```yaml
output_dir: ./results/model_comparison

files_config:
  - path: ./ml_data/kfold_results/all_prediction_results.csv
    model_names:
      - RandomForest
      - LogisticRegression
      - XGBoost
    prob_suffix: "_prob"
    pred_suffix: "_pred"
    label_col: "true_label"
    split_col: "split"

split:
  enabled: false  # K-Fold结果通常不需要再分组

statistical_tests:
  enabled: true
  methods:
    - delong
    - mcnemar

visualization:
  enabled: true
  plot_types:
    - roc
    - calibration
    - confusion
    - dca
```

### 对比不同训练方式

您还可以对比K-Fold交叉验证和标准训练/测试集划分的结果：

```yaml
files_config:
  # 标准训练结果
  - path: ./results/standard_ml/all_prediction_results.csv
    model_names: [RandomForest, XGBoost]
    # ...
  
  # K-Fold验证结果
  - path: ./results/kfold/all_prediction_results.csv
    model_names: [RandomForest, XGBoost]
    # ...
```

这样可以评估：
- 不同训练策略对模型性能的影响
- 模型在不同评估方法下的稳定性
- 选择最适合您数据的训练方法

## 相关文档

- [机器学习配置文件说明](app_of_machine_learning.md)
- [机器学习模型指南](app_machine_learning_models.md)
- [模型比较工具使用](app_model_comparison_plots.md)
- [特征选择方法](../habit/core/machine_learning/feature_selectors/README.md)
- [CLI使用指南](../HABIT_CLI.md)

