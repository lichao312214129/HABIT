# app_of_machine_learning.py 功能文档

## 功能概述

`app_of_machine_learning.py` 是HABIT工具包中用于放射组学建模和预测的入口程序。该模块提供了完整的机器学习流程，包括数据预处理、特征选择、模型训练、性能评估和新数据预测。支持多种机器学习算法，可用于放射组学特征的分类和回归任务。

## 用法

### 训练模式

```bash
python scripts/app_of_machine_learning.py --config <config_file_path> --mode train
```

### 预测模式

```bash
python scripts/app_of_machine_learning.py --config <config_file_path> --mode predict --model <model_file_path> --data <data_file_path> [--output <output_dir>] [--model_name <model_name>] [--evaluate]
```

## 命令行参数

| 参数 | 描述 |
|-----|-----|
| `--config` | YAML配置文件路径（必需） |
| `--mode` | 运行模式: 'train' (训练) 或 'predict' (预测)，默认为'train' |
| `--model` | 模型包文件路径 (.pkl)，预测模式必需 |
| `--data` | 预测数据文件路径 (.csv)，预测模式必需 |
| `--output` | 预测结果保存路径 |
| `--model_name` | 用于预测的特定模型名称 |
| `--evaluate` | 是否评估模型性能并生成图表 |

## 配置文件格式

`app_of_machine_learning.py` 使用YAML格式的配置文件，包含以下主要部分：

### 基本配置

```yaml
# 数据和输出路径
input: <输入数据文件路径>
output: <输出目录路径>
label: <标签列名>
task_type: <任务类型，分类(classification)或回归(regression)>
random_seed: <随机种子>
```

### 数据预处理配置

```yaml
preprocessing:
  drop_features: <要删除的特征列表>
  drop_features_with_nan_rate: <删除空值率超过此阈值的特征>
  drop_correlated_features: <是否删除高相关特征>
  correlation_threshold: <相关性阈值>
  one_hot_encode: <是否进行独热编码>
  categorical_features: <分类特征列表>
  impute_method: <缺失值填充方法>
```

### 数据分割配置

```yaml
data_split:
  method: <分割方法>
  test_size: <测试集比例>
  n_splits: <交叉验证折数>
  group_col: <分组列名>
```

### 特征选择配置

```yaml
feature_selection:
  method: <特征选择方法>
  n_features: <选择的特征数量>
  alpha: <alpha参数>
  step: <步长>
```

### 机器学习模型配置

```yaml
models:
  - name: <模型名称>
    type: <模型类型>
    hyperparameters:
      <参数1>: <值1>
      <参数2>: <值2>
      ...
  - name: <模型名称2>
    ...
```

### 模型评估配置

```yaml
evaluation:
  metrics: <评估指标列表>
  n_permutations: <置换测试次数>
  generate_plots: <是否生成图表>
  calibration: <是否进行校准>
```

## 支持的数据预处理方法

### 缺失值填充方法

- `mean`: 均值填充
- `median`: 中位数填充
- `most_frequent`: 众数填充
- `constant`: 常数填充
- `knn`: K近邻填充

### 特征选择方法

- `univariate`: 单变量特征选择（基于统计测试）
- `rfe`: 递归特征消除
- `lasso`: Lasso正则化特征选择
- `pca`: 主成分分析
- `variance_threshold`: 方差阈值特征选择
- `boruta`: Boruta算法特征选择
- `none`: 不进行特征选择

## 支持的机器学习模型

### 分类模型

- `logistic_regression`: 逻辑回归
- `svm`: 支持向量机
- `random_forest`: 随机森林
- `xgboost`: XGBoost
- `lightgbm`: LightGBM
- `decision_tree`: 决策树
- `naive_bayes`: 朴素贝叶斯
- `knn`: K近邻分类器
- `neural_network`: 神经网络

### 回归模型

- `linear_regression`: 线性回归
- `lasso`: Lasso回归
- `ridge`: Ridge回归
- `elastic_net`: 弹性网络回归
- `svr`: 支持向量回归
- `random_forest_regressor`: 随机森林回归
- `xgboost_regressor`: XGBoost回归
- `lightgbm_regressor`: LightGBM回归
- `decision_tree_regressor`: 决策树回归
- `knn_regressor`: K近邻回归

## 支持的评估指标

### 分类指标

- `accuracy`: 准确率
- `precision`: 精确率
- `recall`: 召回率
- `f1`: F1分数
- `roc_auc`: ROC AUC
- `pr_auc`: PR AUC (精确率-召回率曲线下面积)
- `sensitivity`: 敏感度
- `specificity`: 特异度

### 回归指标

- `r2`: R²决定系数
- `mae`: 平均绝对误差
- `mse`: 均方误差
- `rmse`: 均方根误差
- `explained_variance`: 解释方差

## 完整配置示例

### 分类任务

```yaml
# 基本配置
input: ./data/radiomics_features.csv
output: ./results/classification_results
label: label
task_type: classification
random_seed: 42

# 数据预处理
preprocessing:
  drop_features: ["patient_id", "date", "image_path"]
  drop_features_with_nan_rate: 0.2
  drop_correlated_features: true
  correlation_threshold: 0.9
  one_hot_encode: true
  categorical_features: ["gender", "stage"]
  impute_method: mean

# 数据分割
data_split:
  method: stratified
  test_size: 0.3
  n_splits: 5

# 特征选择
feature_selection:
  method: lasso
  n_features: 20
  alpha: 0.01
  step: 1

# 模型配置
models:
  - name: LogisticRegression
    type: logistic_regression
    hyperparameters:
      C: 1.0
      penalty: l2
      solver: liblinear
      
  - name: RandomForest
    type: random_forest
    hyperparameters:
      n_estimators: 100
      max_depth: 5
      min_samples_split: 5
      
  - name: XGBoost
    type: xgboost
    hyperparameters:
      n_estimators: 100
      max_depth: 3
      learning_rate: 0.1

# 评估配置
evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "roc_auc"]
  n_permutations: 100
  generate_plots: true
  calibration: true
```

### 回归任务

```yaml
# 基本配置
input: ./data/radiomics_features.csv
output: ./results/regression_results
label: survival_time
task_type: regression
random_seed: 42

# 数据预处理
preprocessing:
  drop_features: ["patient_id", "date", "image_path"]
  drop_features_with_nan_rate: 0.2
  drop_correlated_features: true
  correlation_threshold: 0.9
  impute_method: median

# 数据分割
data_split:
  method: random
  test_size: 0.3
  n_splits: 5

# 特征选择
feature_selection:
  method: rfe
  n_features: 15

# 模型配置
models:
  - name: LinearRegression
    type: linear_regression
    
  - name: Ridge
    type: ridge
    hyperparameters:
      alpha: 1.0
      
  - name: RandomForestRegressor
    type: random_forest_regressor
    hyperparameters:
      n_estimators: 100
      max_depth: 5

# 评估配置
evaluation:
  metrics: ["r2", "mae", "rmse"]
  n_permutations: 100
  generate_plots: true
```

## 执行流程

### 训练模式

1. 读取配置文件和数据
2. 数据预处理（填充缺失值、删除高相关特征等）
3. 数据分割（训练集、测试集或交叉验证）
4. 特征标准化/归一化
5. 特征选择
6. 模型训练（支持多个模型同时训练）
7. 模型评估（计算性能指标、生成图表）
8. 模型解释（特征重要性分析）
9. 保存模型和结果

### 预测模式

1. 加载训练好的模型包
2. 读取新数据
3. 应用预处理和特征选择流程
4. 使用模型生成预测结果
5. 可选：评估预测性能（如果提供了真实标签）
6. 保存预测结果

## 输出结果

程序执行后，将在指定的输出目录生成以下内容：

1. `models/`: 保存训练好的模型文件
2. `feature_selection/`: 特征选择结果
3. `evaluation/`: 模型评估结果和图表
4. `predictions/`: 测试集和新数据的预测结果
5. `model_package.pkl`: 完整模型包，包含预处理、特征选择和模型参数
6. `results_summary.csv`: 所有模型的性能指标摘要

## 注意事项

1. 确保输入数据格式正确，标签列必须存在
2. 对于分类任务，标签应该是分类变量（可以是数字、字符串或布尔值）
3. 对于回归任务，标签应该是连续数值
4. 使用预测模式时，新数据应包含与训练数据相同的特征列（除标签列外）
5. 建议在配置文件中指定随机种子以确保结果可重复性 