# 机器学习分析

机器学习分析是HABIT工作流程的最后阶段，用于建立基于生境特征的预测模型。本节介绍如何使用HABIT的机器学习模块进行特征选择、模型训练和评估。

## 工作原理

HABIT的机器学习模块提供了端到端的建模流程，包括：

1. **数据加载与预处理**：读取特征文件，处理缺失值，标准化特征
2. **特征选择**：使用多种方法筛选最相关特征
3. **模型训练**：支持多种算法进行模型训练
4. **模型评估**：使用交叉验证评估模型性能
5. **结果可视化**：生成ROC曲线、PR曲线等评估图表

## 使用场景

机器学习分析适用于以下场景：

1. **诊断模型**：预测疾病状态或分类
2. **预后模型**：预测患者预后或生存率
3. **风险分层**：根据风险水平分层患者
4. **特征重要性分析**：确定对预测最有价值的特征

## 命令行使用

HABIT的机器学习脚本支持三种操作模式：训练、预测和评估。

### 训练模式

```bash
python scripts/run_machine_learning.py \
  --config config/ml_config.yaml \
  --mode train \
  --output_dir results/ml_models
```

### 预测模式

```bash
python scripts/run_machine_learning.py \
  --config config/ml_config.yaml \
  --mode predict \
  --model_file models/best_model.pkl \
  --predict_data new_data.csv \
  --output_dir results/predictions
```

### 评估模式

```bash
python scripts/run_machine_learning.py \
  --config config/ml_config.yaml \
  --mode evaluate \
  --model_file models/best_model.pkl \
  --output_dir results/evaluations
```

### 参数说明

- **--config**：配置文件路径（必需）
- **--mode**：操作模式（train, predict, evaluate）
- **--output_dir**：输出目录
- **--model_file**：模型文件路径（预测和评估模式必需）
- **--predict_data**：预测数据文件路径（预测模式必需）
- **--debug**：启用调试模式（可选）

## 配置文件

机器学习分析通过YAML配置文件设置。以下是典型配置示例：

```yaml
# 输入数据配置
input:
  - path: /path/to/features.csv
    name: features
    subject_id_col: PatientID
    label_col: Label

# 输出目录
output: /path/to/ml_results

# 数据分割配置
test_size: 0.3
random_state: 42
split_method: stratified  # stratified, random, k_fold

# 特征预处理
scaler: standard  # standard, minmax, robust, none

# 特征选择方法
feature_selection_methods:
  - name: variance_threshold
    params:
      threshold: 0.1
  - name: select_k_best
    params:
      k: 20
      score_func: f_classif

# 模型配置
models:
  - name: random_forest
    params:
      n_estimators: 100
      max_depth: 5
  - name: svm
    params:
      C: 1.0
      kernel: rbf
  - name: xgboost
    params:
      learning_rate: 0.1
      n_estimators: 100

# 可视化和保存选项
is_visualize: true
is_save_model: true
model_file: /path/to/save/model.pkl
```

## 输出结果

机器学习分析会产生以下输出：

1. **模型文件**：训练好的模型（.pkl格式）
2. **评估结果**：模型性能指标（CSV格式）
3. **可视化图表**：
   - ROC曲线（在多类别问题中为每类一条曲线）
   - 精确率-召回率曲线
   - 混淆矩阵
   - 特征重要性图
4. **预测结果**：对新数据的预测（CSV格式）

## 支持的机器学习算法

HABIT支持多种机器学习算法：

1. **分类算法**
   - 随机森林（Random Forest）
   - 支持向量机（SVM）
   - XGBoost
   - 逻辑回归（Logistic Regression）
   - 决策树（Decision Tree）
   - K近邻（KNN）

2. **回归算法**
   - 线性回归（Linear Regression）
   - 随机森林回归（Random Forest Regressor）
   - 支持向量回归（SVR）
   - 弹性网（Elastic Net）

## 特征选择方法

HABIT支持以下特征选择方法：

1. **方差阈值法**：移除方差低于阈值的特征
2. **单变量特征选择**：基于统计检验选择最佳特征
3. **递归特征消除（RFE）**：基于模型反馈递归移除特征
4. **基于模型的特征选择**：使用树模型的特征重要性
5. **ICC选择**：结合ICC值筛选稳定特征

## 高级建模技巧

### 处理类别不平衡

对于不平衡数据集，配置文件可以添加以下选项：

```yaml
# 处理类别不平衡
class_weight: balanced  # none, balanced, 或自定义权重字典
sampling_strategy: 
  method: smote  # none, smote, adasyn, random_over, random_under
  params:
    k_neighbors: 5
```

### 超参数优化

对于模型调优，可以使用交叉验证网格搜索：

```yaml
hyperparameter_tuning:
  enabled: true
  method: grid_search  # grid_search, random_search
  cv: 5
  scoring: roc_auc
  param_grid:
    random_forest:
      n_estimators: [50, 100, 200]
      max_depth: [3, 5, 7, null]
```

### 集成学习

可以组合多个模型形成集成：

```yaml
ensemble:
  method: voting  # voting, stacking
  voting: soft  # hard, soft
  weights: [1, 1, 2]  # 对应models列表中的权重
```

## 注意事项

1. 使用ICC分析筛选稳定特征，提高模型可重复性
2. 对于小数据集，考虑使用交叉验证而非单一训练测试集分割
3. 避免数据泄漏，确保测试集在训练前不被使用
4. 选择合适的评估指标，如对于不平衡数据集使用AUC而非准确率
5. 保存完整的预处理和特征选择管道，确保预测时使用相同的处理 