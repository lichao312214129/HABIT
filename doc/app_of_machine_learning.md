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

### 数据标准化配置

```yaml
# 数据标准化/归一化配置
normalization:
  method: <标准化方法名称>  # 支持多种标准化方法
  params:
    <参数1>: <值1>  # 特定标准化方法的参数
    <参数2>: <值2>
```

## 支持的数据预处理方法

### 缺失值填充方法

- `mean`: 均值填充
- `median`: 中位数填充
- `most_frequent`: 众数填充
- `constant`: 常数填充
- `knn`: K近邻填充

## 支持的特征选择方法

### 特征选择时机选择
所有特征选择方法都支持一个新的参数 `before_z_score`，用于控制该方法是在Z-score标准化前还是标准化后执行：
- `before_z_score: true` - 方法将在Z-score标准化前执行
- `before_z_score: false` - 方法将在Z-score标准化后执行（默认行为）

对于方差敏感的方法（如方差阈值过滤器），建议设置 `before_z_score: true`，因为Z-score标准化会使所有特征的方差变为1，导致方差过滤失效。

### ICC (Intraclass Correlation Coefficient) 方法
- `method: 'icc'`: 基于特征重复性选择特征
- 参数:
  - `icc_results`: ICC结果JSON文件路径
  - `keys`: 使用的ICC结果键
  - `threshold`: 保留特征的最小ICC值(0.0-1.0)
  - `before_z_score`: 是否在Z-score标准化前执行，默认为false

### VIF (Variance Inflation Factor) 方法
- `method: 'vif'`: 移除具有高多重共线性的特征
- 参数:
  - `max_vif`: 最大允许的VIF值
  - `visualize`: 是否生成VIF值可视化
  - `before_z_score`: 是否在Z-score标准化前执行，默认为false

### 相关性方法
- `method: 'correlation'`: 移除高度相关的特征
- 参数:
  - `threshold`: 相关性阈值
  - `method`: 相关性计算方法('pearson', 'spearman'或'kendall')
  - `visualize`: 是否生成相关性热图
  - `before_z_score`: 是否在Z-score标准化前执行，默认为false

### ANOVA方法
- `method: 'anova'`: 基于ANOVA F值选择特征
- 参数:
  - `p_threshold`: P值阈值，默认为0.05（选择p值小于阈值的特征）
  - `n_features_to_select`: 可选参数，要选择的特征数量（如果指定，则覆盖p_threshold）
  - `plot_importance`: 是否绘制特征重要性图，默认为True
  - `before_z_score`: 是否在Z-score标准化前执行，默认为false

### Chi2方法
- `method: 'chi2'`: 基于卡方统计量选择特征（适用于非负特征的分类问题）
- 参数:
  - `p_threshold`: P值阈值，默认为0.05（选择p值小于阈值的特征）
  - `n_features_to_select`: 可选参数，要选择的特征数量（如果指定，则覆盖p_threshold）
  - `plot_importance`: 是否绘制特征重要性图，默认为True
  - `visualize`: 是否生成可视化结果
  - `before_z_score`: 是否在Z-score标准化前执行，默认为false

### 统计检验方法
- `method: 'statistical_test'`: 基于统计检验（t检验或Mann-Whitney U检验）选择特征
- 参数:
  - `p_threshold`: P值阈值，默认为0.05（选择p值小于阈值的特征）
  - `n_features_to_select`: 可选参数，要选择的特征数量（如果指定，则覆盖p_threshold）
  - `normality_test_threshold`: Shapiro-Wilk正态性检验阈值，默认为0.05
  - `plot_importance`: 是否绘制特征重要性图，默认为True
  - `force_test`: 强制使用特定检验方法，可选值为'ttest'或'mannwhitney'，默认自动选择
  - `before_z_score`: 是否在Z-score标准化前执行，默认为false

### 方差阈值方法
- `method: 'variance'`: 基于特征方差选择特征，移除低方差特征
- 参数:
  - `threshold`: 方差阈值，默认为0.0（保留方差高于阈值的特征）
  - `plot_variances`: 是否绘制特征方差图，默认为True
  - `before_z_score`: 是否在Z-score标准化前执行，推荐设置为true，因为标准化后所有特征的方差都为1
  - `top_k`: 选择方差最大的前k个特征（如果指定，将覆盖threshold参数）
  - `top_percent`: 选择方差最大的前百分比特征（0-100之间，如果指定，将覆盖threshold参数）

使用示例 - 基于阈值选择:
```yaml
feature_selection_methods:
  - method: 'variance'
    params:
      before_z_score: true
      threshold: 0.1
      plot_variances: true
```

使用示例 - 选择前k个特征:
```yaml
feature_selection_methods:
  - method: 'variance'
    params:
      before_z_score: true
      top_k: 20  # 选择方差最大的前20个特征
      plot_variances: true
```

使用示例 - 选择前百分比特征:
```yaml
feature_selection_methods:
  - method: 'variance'
    params:
      before_z_score: true
      top_percent: 10  # 选择方差最大的前10%特征
      plot_variances: true
```

### mRMR (Minimum Redundancy Maximum Relevance) 方法
- `method: 'mrmr'`: 选择与目标高度相关但特征间冗余度低的特征
- 参数:
  - `target`: 目标变量名
  - `n_features`: 要选择的特征数量（默认10）
  - `method`: MRMR方法，可选'MIQ'（互信息商）或'MID'（互信息差）
  - `visualize`: 是否生成可视化结果
  - `outdir`: 输出目录（用于保存可视化结果）
  - `before_z_score`: 是否在Z-score标准化前执行，默认为false

### LASSO (L1正则化) 方法
- `method: 'lasso'`: 使用L1正则化进行特征选择
- 参数:
  - `cv`: 选择最优alpha的交叉验证折数
  - `n_alphas`: 尝试的alpha值数量
  - `random_state`: 随机种子
  - `visualize`: 是否生成特征系数可视化
  - `before_z_score`: 是否在Z-score标准化前执行，默认为false

### RFECV (Recursive Feature Elimination with Cross-Validation) 方法
- `method: 'rfecv'`: 使用递归特征消除与交叉验证进行特征选择
- 参数:
  - `estimator`: 基础估计器，支持以下模型：
    - 分类任务：
      - `LogisticRegression`: 逻辑回归
      - `RandomForestClassifier`: 随机森林分类器
      - `SVC`: 支持向量机分类器
      - `GradientBoostingClassifier`: 梯度提升分类器
      - `XGBClassifier`: XGBoost分类器
      - `LGBMClassifier`: LightGBM分类器
    - 回归任务：
      - `LinearRegression`: 线性回归
      - `RandomForestRegressor`: 随机森林回归器
      - `SVR`: 支持向量机回归器
      - `GradientBoostingRegressor`: 梯度提升回归器
      - `XGBRegressor`: XGBoost回归器
      - `LGBMRegressor`: LightGBM回归器
  - `step`: 每次迭代要移除的特征数量（默认为1）
  - `cv`: 交叉验证折数（默认为5）
  - `scoring`: 评估指标
    - 分类任务：'accuracy', 'f1', 'roc_auc', 'precision', 'recall'
    - 回归任务：'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
  - `min_features_to_select`: 最少保留的特征数量（默认为1）
  - `n_jobs`: 并行计算的作业数（默认为-1，使用所有CPU）
  - `random_state`: 随机种子
  - `visualize`: 是否生成特征数量与性能关系图
  - `before_z_score`: 是否在Z-score标准化前执行，默认为false

使用示例：
```yaml
feature_selection_methods:
  - method: 'rfecv'
    params:
      estimator: 'RandomForestClassifier'  # 使用随机森林分类器
      step: 1
      cv: 5
      scoring: 'roc_auc'
      min_features_to_select: 5
      n_jobs: -1
      random_state: 42
      visualize: true
      before_z_score: false  # 在Z-score标准化后执行
```

### 单变量逻辑回归方法
- `method: 'univariate_logistic'`: 基于单变量逻辑回归p值选择特征
- 参数:
  - `threshold`: 特征选择的最大p值阈值
  - `before_z_score`: 是否在Z-score标准化前执行，默认为false

### 逐步特征选择方法
- `method: 'stepwise'`: 使用AIC/BIC准则进行逐步特征选择
- 参数:
  - `direction`: 逐步选择方向：'forward'(前向)、'backward'(后向)或'both'(双向)
  - `criterion`: 选择准则：'aic'(AIC)、'bic'(BIC)或'pvalue'(p值)
  - `before_z_score`: 是否在Z-score标准化前执行，默认为false

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
  - method: 'variance'  # 方差阈值过滤，在Z-score前执行
    params:
      threshold: 0.1  # 方差阈值
      plot_variances: true  # 生成方差可视化
      before_z_score: true  # 在Z-score标准化前执行
      
  - method: 'univariate_logistic'  # 基于单变量逻辑回归p值选择特征
    params:
      threshold: 0.1  # p值阈值，低于此值的特征被保留
      before_z_score: false  # 在Z-score标准化后执行
      
  - method: 'stepwise'  # 逐步特征选择
    params:
      Rhome: 'E:/software/R'  # R安装路径
      direction: 'backward'  # 使用后向选择方法
      before_z_score: false  # 在Z-score标准化后执行

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
4. 特征选择（第一阶段，处理`before_z_score: true`的方法）
5. Z-score特征标准化
6. 特征选择（第二阶段，处理`before_z_score: false`或未指定的方法）
7. 模型训练（支持多个模型同时训练）
8. 模型评估（计算性能指标、生成图表）
9. 模型解释（特征重要性分析、SHAP值）
10. 保存模型和结果

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

## 支持的标准化/归一化方法

### Z-Score标准化 (StandardScaler)
- `method: 'z_score'`: 标准化特征为零均值和单位方差
- 参数: 无需额外参数

```yaml
normalization:
  method: z_score
```

### Min-Max缩放 (MinMaxScaler)
- `method: 'min_max'`: 将特征缩放到指定范围内
- 参数:
  - `feature_range`: 缩放范围，默认为[0, 1]

```yaml
normalization:
  method: min_max
  params:
    feature_range: [0, 1]  # 缩放到0-1范围
```

### 稳健缩放 (RobustScaler)
- `method: 'robust'`: 使用对异常值不敏感的统计量缩放特征
- 参数:
  - `quantile_range`: 用于计算缩放的百分位范围，默认为[25.0, 75.0]
  - `with_centering`: 是否在缩放前中心化数据，默认为True
  - `with_scaling`: 是否缩放数据到四分位距，默认为True

```yaml
normalization:
  method: robust
  params:
    quantile_range: [25.0, 75.0]
    with_centering: true
    with_scaling: true
```

### 最大绝对值缩放 (MaxAbsScaler)
- `method: 'max_abs'`: 按每个特征的最大绝对值缩放
- 参数: 无需额外参数

```yaml
normalization:
  method: max_abs
```

### 样本归一化 (Normalizer)
- `method: 'normalizer'`: 将样本缩放为单位范数
- 参数:
  - `norm`: 使用的范数，可选'l1'、'l2'或'max'，默认为'l2'

```yaml
normalization:
  method: normalizer
  params:
    norm: l2  # 使用L2范数归一化
```

### 分位数变换 (QuantileTransformer)
- `method: 'quantile'`: 将特征转换为均匀或正态分布
- 参数:
  - `n_quantiles`: 用于量化的分位数数量，默认为1000
  - `output_distribution`: 输出分布，可选'uniform'或'normal'，默认为'uniform'

```yaml
normalization:
  method: quantile
  params:
    n_quantiles: 1000
    output_distribution: uniform  # 或 normal
```

### 幂变换 (PowerTransformer)
- `method: 'power'`: 应用幂变换使数据更接近高斯分布
- 参数:
  - `method`: 变换方法，可选'yeo-johnson'或'box-cox'，默认为'yeo-johnson'
  - `standardize`: 是否标准化转换后的数据为零均值和单位方差，默认为True

```yaml
normalization:
  method: power
  params:
    method: yeo-johnson
    standardize: true
``` 