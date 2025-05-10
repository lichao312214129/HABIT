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
input:
  - path: <输入数据文件路径>
    name: <特征名称前缀，默认为空>
    subject_id_col: <患者ID列名>
    label_col: <标签列名>
    features: <可选的特定特征列表>
output: <输出目录路径>
```

### 数据分割配置

```yaml
# 数据分割方法：random(随机分割)、stratified(分层分割)或custom(自定义分割)
split_method: <分割方法>
test_size: <测试集比例>  # 当split_method为random或stratified时使用

# 当split_method为custom时使用
train_ids_file: <训练集ID文件路径>
test_ids_file: <测试集ID文件路径>
```

### 特征选择配置

```yaml
feature_selection_methods:
  # 可以配置多个特征选择方法，它们将按顺序执行
  - method: <特征选择方法名称>
    params:
      <参数1>: <值1>
      <参数2>: <值2>
      ...
```

### 机器学习模型配置

```yaml
models:
  <模型名称>:
    params:
      <参数1>: <值1>
      <参数2>: <值2>
      ...
```

### 可视化和保存配置

```yaml
is_visualize: <是否生成性能可视化图表>
is_save_model: <是否保存训练好的模型>
```

## 支持的数据预处理方法

### 缺失值填充方法

- `mean`: 均值填充
- `median`: 中位数填充
- `most_frequent`: 众数填充
- `constant`: 常数填充
- `knn`: K近邻填充

## 支持的特征选择方法

### ICC (Intraclass Correlation Coefficient) 方法
- `method: 'icc'`: 基于特征重复性选择特征
- 参数:
  - `icc_results`: ICC结果JSON文件路径
  - `keys`: 使用的ICC结果键
  - `threshold`: 保留特征的最小ICC值(0.0-1.0)

### VIF (Variance Inflation Factor) 方法
- `method: 'vif'`: 移除具有高多重共线性的特征
- 参数:
  - `max_vif`: 最大允许的VIF值
  - `visualize`: 是否生成VIF值可视化

### 相关性方法
- `method: 'correlation'`: 移除高度相关的特征
- 参数:
  - `threshold`: 相关性阈值
  - `method`: 相关性计算方法('pearson', 'spearman'或'kendall')
  - `visualize`: 是否生成相关性热图

### ANOVA方法
- `method: 'anova'`: 基于ANOVA F值选择特征
- 参数:
  - `n_features_to_select`: 选择的顶级特征数量
  - `plot_importance`: 是否绘制特征重要性

### mRMR (Minimum Redundancy Maximum Relevance) 方法
- `method: 'mrmr'`: 选择与目标高度相关但特征间冗余度低的特征
- 参数:
  - `n_features_to_select`: 选择的特征数量
  - `visualize`: 是否生成选定特征的可视化

### LASSO (L1正则化) 方法
- `method: 'lasso'`: 使用L1正则化进行特征选择
- 参数:
  - `cv`: 选择最优alpha的交叉验证折数
  - `n_alphas`: 尝试的alpha值数量
  - `random_state`: 随机种子
  - `visualize`: 是否生成特征系数可视化

### Boruta方法
- `method: 'boruta'`: 基于随机森林重要性的全相关特征选择
- 参数:
  - `n_estimators`: 随机森林中的树数量
  - `max_iter`: 最大迭代次数
  - `random_state`: 随机种子
  - `include_tentative`: 是否包括暂定重要特征

### 单变量逻辑回归方法
- `method: 'univariate_logistic'`: 基于单变量逻辑回归p值选择特征
- 参数:
  - `threshold`: 特征选择的最大p值阈值

### 逐步特征选择方法
- `method: 'stepwise'`: 使用AIC/BIC准则进行逐步特征选择
- 参数:
  - `Rhome`: R安装路径（逐步选择需要）
  - `direction`: 逐步选择方向：'forward'(前向)、'backward'(后向)或'both'(双向)

## 支持的机器学习模型

### 分类模型

- `LogisticRegression`: 逻辑回归
- `SVM`: 支持向量机
- `RandomForest`: 随机森林
- `XGBoost`: XGBoost

### 回归模型

- `LinearRegression`: 线性回归
- `Ridge`: Ridge回归
- `RandomForestRegressor`: 随机森林回归
- `XGBoostRegressor`: XGBoost回归

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
input:
  - path: ./data/radiomics_features.csv
    name: ''
    subject_id_col: 'subjID'
    label_col: 'label'
    features: []
output: ./results/classification_results

# 数据分割
split_method: 'custom'  # 使用自定义分割方法
train_ids_file: './data/train_ids.txt'  # 训练集ID文件
test_ids_file: './data/test_ids.txt'  # 测试集ID文件

# 特征选择
feature_selection_methods:
  - method: 'univariate_logistic'  # 基于单变量逻辑回归p值选择特征
    params:
      threshold: 0.1  # p值阈值，低于此值的特征被保留
      
  - method: 'stepwise'  # 逐步特征选择
    params:
      Rhome: 'E:/software/R'  # R安装路径
      direction: 'backward'  # 使用后向选择方法

# 模型配置
models:
  LogisticRegression:
    params:
      random_state: 42
      max_iter: 1000
      C: 1.0
      penalty: "l2"
      solver: "lbfgs"

# 可视化和保存配置
is_visualize: true  # 生成性能可视化图表
is_save_model: true  # 保存训练好的模型
```

### 回归任务

```yaml
# 基本配置
input:
  - path: ./data/radiomics_features.csv
    name: ''
    subject_id_col: 'subjID'
    label_col: 'survival_time'
    features: []
output: ./results/regression_results

# 数据分割
split_method: 'random'
test_size: 0.3
random_state: 42

# 特征选择
feature_selection_methods:
  - method: 'lasso'
    params:
      cv: 5
      n_alphas: 100
      random_state: 42
      visualize: true

# 模型配置
models:
  LinearRegression:
    params: {}
    
  Ridge:
    params:
      alpha: 1.0
      
  RandomForestRegressor:
    params:
      n_estimators: 100
      max_depth: 5

# 可视化和保存配置
is_visualize: true
is_save_model: true
```

## 执行流程

### 训练模式

1. 读取配置文件和数据
2. 数据预处理
3. 数据分割（训练集、测试集）
4. 特征标准化/归一化
5. 特征选择
6. 模型训练（支持多个模型同时训练）
7. 模型评估（计算性能指标、生成图表）
8. 模型解释（特征重要性分析、SHAP值）
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