# 机器学习模型使用指南

本文档介绍habit软件包中机器学习模块支持的各种模型及其使用方法。

## 概述

habit的机器学习模块提供了丰富的分类模型，包括：
- 线性模型（Logistic Regression）
- 支持向量机（SVM）
- 树模型（Decision Tree, Random Forest, XGBoost, GradientBoosting, AdaBoost）
- 基于距离的模型（KNN）
- 神经网络（MLP）
- 概率模型（Naive Bayes系列）
- 自动机器学习（AutoGluon）

## 可用模型列表

### 1. Logistic Regression（逻辑回归）

**适用场景**: 线性可分问题、特征相对独立、需要可解释性

**配置示例**:
```yaml
LogisticRegression:
  params:
    C: 1.0                    # Regularization strength
    penalty: "l2"             # l1, l2, elasticnet, none
    solver: "lbfgs"           # lbfgs, liblinear, newton-cg, sag, saga
    max_iter: 1000            # Maximum iterations
    random_state: 42
    class_weight: "balanced"  # Handle class imbalance
```

**优点**:
- 训练速度快
- 可解释性强
- 支持特征重要性分析

**缺点**:
- 只能处理线性可分问题
- 对特征工程要求较高

---

### 2. Support Vector Machine（支持向量机）

**适用场景**: 小样本、高维数据、非线性问题

**配置示例**:
```yaml
SVM:
  params:
    C: 1.0                    # Regularization parameter
    kernel: "rbf"             # linear, poly, rbf, sigmoid
    gamma: "scale"            # Kernel coefficient
    probability: true         # Enable probability estimates
    random_state: 42
    class_weight: "balanced"
```

**优点**:
- 在高维空间表现优异
- 适合小样本问题
- 通过核函数处理非线性问题

**缺点**:
- 训练速度相对较慢
- 对参数敏感
- 不提供特征重要性

---

### 3. Random Forest（随机森林）

**适用场景**: 通用分类问题、需要特征重要性、对过拟合不敏感

**配置示例**:
```yaml
RandomForest:
  params:
    n_estimators: 100         # Number of trees
    max_depth: null           # Maximum depth (null = unlimited)
    min_samples_split: 2      # Min samples to split
    min_samples_leaf: 1       # Min samples in leaf
    max_features: "sqrt"      # Features per split
    random_state: 42
    class_weight: "balanced"
```

**优点**:
- 不容易过拟合
- 提供特征重要性
- 对缺失值和异常值鲁棒
- 不需要特征标准化

**缺点**:
- 模型文件较大
- 预测速度相对较慢

---

### 4. XGBoost（梯度提升树）

**适用场景**: 竞赛、需要高性能、结构化数据

**配置示例**:
```yaml
XGBoost:
  params:
    n_estimators: 100         # Number of trees
    max_depth: 3              # Maximum depth
    learning_rate: 0.1        # Learning rate
    subsample: 0.8            # Sample ratio
    colsample_bytree: 0.8     # Feature ratio
    random_state: 42
    objective: "binary:logistic"
    eval_metric: "logloss"
```

**优点**:
- 性能通常很好
- 提供特征重要性
- 支持并行计算
- 内置正则化

**缺点**:
- 参数较多，调参复杂
- 可能过拟合

---

### 5. K-Nearest Neighbors（K近邻）

**适用场景**: 简单分类、小数据集、非参数方法

**配置示例**:
```yaml
KNN:
  params:
    n_neighbors: 5            # Number of neighbors
    weights: "uniform"        # uniform or distance
    algorithm: "auto"         # auto, ball_tree, kd_tree, brute
    metric: "minkowski"       # Distance metric
    p: 2                      # Power for Minkowski
```

**优点**:
- 简单直观
- 无需训练
- 适合不规则决策边界

**缺点**:
- 预测速度慢
- 对特征尺度敏感（需要标准化）
- 不提供特征重要性
- 对高维数据效果差

---

### 6. Multi-layer Perceptron（多层感知机）

**适用场景**: 复杂非线性问题、大数据集

**配置示例**:
```yaml
MLP:
  params:
    hidden_layer_sizes: [100, 50]  # Hidden layer sizes
    activation: "relu"             # relu, tanh, logistic
    solver: "adam"                 # adam, sgd, lbfgs
    alpha: 0.0001                  # L2 penalty
    learning_rate: "constant"      # constant, adaptive
    learning_rate_init: 0.001      # Initial learning rate
    max_iter: 200                  # Maximum iterations
    random_state: 42
    early_stopping: false          # Use early stopping
```

**优点**:
- 可以学习复杂的非线性关系
- 适合大规模数据

**缺点**:
- 训练时间长
- 需要大量数据
- 对参数敏感
- 黑盒模型，可解释性差

---

### 7. Naive Bayes（朴素贝叶斯）

#### 7.1 Gaussian Naive Bayes（高斯朴素贝叶斯）

**适用场景**: 连续特征、假设特征服从高斯分布

**配置示例**:
```yaml
GaussianNB:
  params:
    var_smoothing: 1.0e-9     # Variance smoothing
```

#### 7.2 Multinomial Naive Bayes（多项式朴素贝叶斯）

**适用场景**: 离散特征、文本分类、计数数据

**配置示例**:
```yaml
MultinomialNB:
  params:
    alpha: 1.0                # Smoothing parameter
    fit_prior: true           # Learn class prior
```

**注意**: 需要非负特征值

#### 7.3 Bernoulli Naive Bayes（伯努利朴素贝叶斯）

**适用场景**: 二值特征、布尔数据

**配置示例**:
```yaml
BernoulliNB:
  params:
    alpha: 1.0                # Smoothing parameter
    binarize: 0.0             # Binarization threshold
    fit_prior: true           # Learn class prior
```

**朴素贝叶斯系列优点**:
- 训练速度极快
- 对小数据集效果好
- 可解释性强

**朴素贝叶斯系列缺点**:
- 假设特征独立（往往不现实）
- 对特征相关性敏感

---

### 8. Gradient Boosting（梯度提升）

**适用场景**: 通用分类问题、需要高性能

**配置示例**:
```yaml
GradientBoosting:
  params:
    n_estimators: 100         # Number of boosting stages
    learning_rate: 0.1        # Learning rate
    max_depth: 3              # Maximum depth
    subsample: 1.0            # Sample fraction
    min_samples_split: 2      # Min samples to split
    random_state: 42
```

**优点**:
- 性能优异
- 提供特征重要性
- 灵活性高

**缺点**:
- 训练速度慢
- 容易过拟合
- 参数多

---

### 9. AdaBoost（自适应提升）

**适用场景**: 弱学习器集成、需要鲁棒性

**配置示例**:
```yaml
AdaBoost:
  params:
    n_estimators: 50          # Number of weak learners
    learning_rate: 1.0        # Learning rate
    algorithm: "SAMME.R"      # SAMME or SAMME.R
    random_state: 42
```

**优点**:
- 简单有效
- 对参数不太敏感
- 提供特征重要性

**缺点**:
- 对噪声和异常值敏感
- 可能过拟合

---

### 10. Decision Tree（决策树）

**适用场景**: 需要高可解释性、作为基学习器

**配置示例**:
```yaml
DecisionTree:
  params:
    criterion: "gini"         # gini or entropy
    splitter: "best"          # best or random
    max_depth: null           # Maximum depth
    min_samples_split: 2      # Min samples to split
    min_samples_leaf: 1       # Min samples in leaf
    random_state: 42
    class_weight: "balanced"
```

**优点**:
- 高度可解释
- 不需要特征标准化
- 可以处理非线性关系
- 提供特征重要性

**缺点**:
- 容易过拟合
- 对数据变化敏感
- 单棵树性能有限

---

## 模型选择建议

### 按任务类型选择

1. **线性可分问题**: Logistic Regression
2. **小样本问题**: SVM, Naive Bayes
3. **大样本问题**: MLP, XGBoost, Random Forest
4. **需要可解释性**: Logistic Regression, Decision Tree, Naive Bayes
5. **追求高性能**: XGBoost, GradientBoosting, Random Forest
6. **快速原型**: KNN, GaussianNB, Decision Tree

### 按数据特征选择

1. **高维数据**: SVM, Logistic Regression
2. **低维数据**: KNN, Decision Tree
3. **非线性数据**: SVM (RBF kernel), MLP, 树模型
4. **类别不平衡**: 使用`class_weight='balanced'`参数的模型
5. **有缺失值**: 树模型（RandomForest, XGBoost）
6. **需要特征标准化**: KNN, SVM, MLP, Logistic Regression
7. **不需要特征标准化**: 树模型, Naive Bayes

### 组合策略

可以同时训练多个模型并比较：

```yaml
models:
  LogisticRegression:
    params:
      C: 1.0
      random_state: 42
  
  RandomForest:
    params:
      n_estimators: 100
      random_state: 42
  
  XGBoost:
    params:
      n_estimators: 100
      random_state: 42
```

## 使用示例

### 基本使用

```python
from habit.core.machine_learning.machine_learning import Modeling
import yaml

# Load configuration
with open('config_machine_learning.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Run modeling pipeline
modeling = Modeling(config)
modeling.read_data()\
    .preprocess_data()\
    ._split_data()\
    .feature_selection_before_normalization()\
    .normalization()\
    .feature_selection()\
    .modeling()\
    .evaluate_models()
```

### K-Fold交叉验证

```python
from habit.core.machine_learning.machine_learning_kfold import run_kfold_modeling
import yaml

# Load configuration
with open('config_machine_learning_kfold.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Run k-fold CV
modeling = run_kfold_modeling(config)
```

## 特征重要性分析

支持特征重要性的模型：
- Logistic Regression（基于系数）
- Random Forest（基于Gini重要性）
- XGBoost（基于增益）
- GradientBoosting（基于分裂改进）
- AdaBoost（基于弱学习器重要性）
- Decision Tree（基于Gini重要性或信息增益）

不支持特征重要性的模型：
- SVM
- KNN
- MLP
- Naive Bayes系列

## 注意事项

1. **数据预处理**:
   - KNN、SVM、MLP需要特征标准化
   - MultinomialNB需要非负特征
   - 树模型不需要特征标准化

2. **类别不平衡**:
   - 使用`class_weight='balanced'`参数
   - 或使用采样技术（SMOTE等）

3. **超参数调优**:
   - 使用网格搜索（GridSearchCV）
   - 或贝叶斯优化（Optuna等）

4. **过拟合控制**:
   - 使用正则化（L1/L2）
   - 限制模型复杂度（max_depth等）
   - 使用交叉验证
   - 增加训练数据

5. **计算资源**:
   - 大数据集避免使用KNN
   - 有限资源时使用简单模型
   - 使用并行计算（n_jobs=-1）

## 常见问题

### Q: 应该选择哪个模型？
A: 没有最好的模型，只有最适合的模型。建议：
   1. 从简单模型开始（Logistic Regression）
   2. 尝试树模型（Random Forest, XGBoost）
   3. 根据结果选择最佳模型
   4. 使用k-fold交叉验证评估

### Q: 如何处理类别不平衡？
A: 可以：
   1. 使用`class_weight='balanced'`
   2. 使用SMOTE等采样技术
   3. 使用适当的评估指标（AUC, F1等）

### Q: 模型训练太慢怎么办？
A: 可以：
   1. 减少特征数量
   2. 减少样本数量
   3. 使用更快的模型
   4. 使用并行计算
   5. 减少模型复杂度

### Q: 如何提高模型性能？
A: 可以：
   1. 增加训练数据
   2. 特征工程
   3. 超参数调优
   4. 模型融合（ensemble）
   5. 使用更复杂的模型

## 相关文档

- [机器学习配置文件说明](app_of_machine_learning.md)
- [特征选择方法](../habit/core/machine_learning/feature_selectors/README.md)
- [模型API文档](../habit/core/machine_learning/models/README.md)

