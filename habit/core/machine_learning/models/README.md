# 模型注册系统使用指南

本模块提供了统一的机器学习模型注册和管理系统，使用工厂模式设计，便于扩展和自定义。

## 基本用法

### 1. 在配置文件中指定模型

模型通常通过配置文件来指定，以下是一个示例配置文件：

```yaml
# 模型配置
models:
  LogisticRegression:
    type: LogisticRegression
    params:
      C: 1.0
      penalty: 'l2'
      solver: 'liblinear'
      max_iter: 1000
      class_weight: 'balanced'
  
  SVM:
    type: SVM
    params:
      C: 1.0
      kernel: 'rbf'
      gamma: 'scale'
      probability: true
      class_weight: 'balanced'
      
  XGBoost:
    type: XGBoost
    params:
      n_estimators: 100
      max_depth: 3
      learning_rate: 0.1
      subsample: 0.8
      colsample_bytree: 0.8

  RandomForest:
    type: RandomForest
    params:
      n_estimators: 100
      max_depth: null
      max_features: 'sqrt'

  KNN:
    type: KNN
    params:
      n_neighbors: 5
      weights: 'uniform'

  MLP:
    type: MLP
    params:
      hidden_layer_sizes: [100, 50]
      activation: 'relu'
      max_iter: 200

  GaussianNB:
    type: GaussianNB
    params:
      var_smoothing: 1.0e-9

  GradientBoosting:
    type: GradientBoosting
    params:
      n_estimators: 100
      learning_rate: 0.1
      max_depth: 3

  AdaBoost:
    type: AdaBoost
    params:
      n_estimators: 50
      learning_rate: 1.0

  DecisionTree:
    type: DecisionTree
    params:
      criterion: 'gini'
      max_depth: null
```

### 2. 在代码中直接创建模型

```python
from habit.core.machine_learning.models.factory import ModelFactory

# 查看可用的模型
available_models = ModelFactory.get_available_models()
print(f"Available models: {available_models}")

# 创建逻辑回归模型
log_reg = ModelFactory.create_model(
    'LogisticRegression',
    {'params': {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'}}
)

# 创建SVM模型
svm = ModelFactory.create_model(
    'SVM',
    {'params': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}}
)

# 使用模型
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)
```

## 可用的模型

本模块提供以下预定义模型：

### 1. 逻辑回归 (`LogisticRegression`)

基于逻辑回归算法的分类模型。

**模型类型**: `linear`

**参数**:
- `C`: 正则化强度的倒数，默认为1.0
- `penalty`: 惩罚项类型，'l1'、'l2'、'elasticnet'或'none'，默认为'l2'
- `solver`: 优化算法，默认为'liblinear'
- `max_iter`: 最大迭代次数，默认为1000
- `random_state`: 随机种子，默认为42
- `class_weight`: 类别权重，None、'balanced'或字典

**示例**:
```python
model = ModelFactory.create_model(
    'LogisticRegression',
    {'params': {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear'}}
)
```

**特征重要性**: 支持（基于系数）

---

### 2. 支持向量机 (`SVM`)

基于支持向量机算法的分类模型。

**模型类型**: `kernel-based`

**参数**:
- `C`: 正则化参数，默认为1.0
- `kernel`: 核函数类型，'linear'、'poly'、'rbf'或'sigmoid'，默认为'rbf'
- `gamma`: 核系数，'scale'、'auto'或浮点数，默认为'scale'
- `probability`: 是否启用概率估计，默认为True
- `class_weight`: 类别权重，None、'balanced'或字典
- `random_state`: 随机种子，默认为42

**示例**:
```python
model = ModelFactory.create_model(
    'SVM',
    {'params': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}}
)
```

**特征重要性**: 不支持

---

### 3. 随机森林 (`RandomForest`)

基于随机森林算法的分类模型。

**模型类型**: `tree`

**参数**:
- `n_estimators`: 森林中树的数量，默认为100
- `max_depth`: 树的最大深度，默认为None
- `min_samples_split`: 分裂内部节点所需的最小样本数，默认为2
- `min_samples_leaf`: 叶节点所需的最小样本数，默认为1
- `max_features`: 寻找最佳分裂时考虑的特征数，默认为'sqrt'
- `bootstrap`: 是否使用bootstrap样本，默认为True
- `class_weight`: 类别权重，None、'balanced'、'balanced_subsample'或字典
- `random_state`: 随机种子，默认为42

**示例**:
```python
model = ModelFactory.create_model(
    'RandomForest',
    {'params': {'n_estimators': 100, 'max_depth': None, 'max_features': 'sqrt'}}
)
```

**特征重要性**: 支持（基于Gini重要性）

---

### 4. XGBoost (`XGBoost`)

基于梯度提升决策树的分类模型。

**模型类型**: `tree`

**参数**:
- `n_estimators`: 提升轮数，默认为100
- `max_depth`: 树的最大深度，默认为3
- `learning_rate`: 学习率，默认为0.1
- `subsample`: 训练样本的子采样比例，默认为0.8
- `colsample_bytree`: 构建树时特征的子采样比例，默认为0.8
- `objective`: 学习任务目标，默认为'binary:logistic'
- `eval_metric`: 评估指标，默认为'logloss'
- `random_state`: 随机种子，默认为42

**示例**:
```python
model = ModelFactory.create_model(
    'XGBoost',
    {'params': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1}}
)
```

**特征重要性**: 支持（基于增益）

---

### 5. K近邻 (`KNN`)

基于K近邻算法的分类模型。

**模型类型**: `distance-based`

**参数**:
- `n_neighbors`: 近邻数量，默认为5
- `weights`: 权重函数，'uniform'或'distance'，默认为'uniform'
- `algorithm`: 计算最近邻的算法，'auto'、'ball_tree'、'kd_tree'或'brute'，默认为'auto'
- `leaf_size`: 传递给BallTree或KDTree的叶子大小，默认为30
- `p`: Minkowski度量的幂参数，默认为2（欧氏距离）
- `metric`: 距离度量，默认为'minkowski'
- `n_jobs`: 并行作业数，默认为-1（使用所有CPU）

**示例**:
```python
model = ModelFactory.create_model(
    'KNN',
    {'params': {'n_neighbors': 5, 'weights': 'uniform'}}
)
```

**特征重要性**: 不支持

---

### 6. 多层感知机 (`MLP`)

基于神经网络的分类模型。

**模型类型**: `neural-network`

**参数**:
- `hidden_layer_sizes`: 隐藏层大小的元组，默认为(100,)
- `activation`: 激活函数，'identity'、'logistic'、'tanh'或'relu'，默认为'relu'
- `solver`: 权重优化的求解器，'lbfgs'、'sgd'或'adam'，默认为'adam'
- `alpha`: L2正则化参数，默认为0.0001
- `batch_size`: 批量大小，默认为'auto'
- `learning_rate`: 学习率调度，'constant'、'invscaling'或'adaptive'，默认为'constant'
- `learning_rate_init`: 初始学习率，默认为0.001
- `max_iter`: 最大迭代次数，默认为200
- `shuffle`: 是否在每次迭代中打乱样本，默认为True
- `random_state`: 随机种子，默认为42
- `early_stopping`: 是否使用早停，默认为False
- `validation_fraction`: 用于早停的验证集比例，默认为0.1

**示例**:
```python
model = ModelFactory.create_model(
    'MLP',
    {'params': {'hidden_layer_sizes': (100, 50), 'activation': 'relu', 'max_iter': 200}}
)
```

**特征重要性**: 不支持

---

### 7. 高斯朴素贝叶斯 (`GaussianNB`)

基于高斯朴素贝叶斯算法的分类模型，假设特征服从高斯分布。

**模型类型**: `probabilistic`

**参数**:
- `priors`: 类的先验概率，默认为None（根据数据计算）
- `var_smoothing`: 方差平滑参数，默认为1e-9

**示例**:
```python
model = ModelFactory.create_model(
    'GaussianNB',
    {'params': {'var_smoothing': 1e-9}}
)
```

**特征重要性**: 不支持

---

### 8. 多项式朴素贝叶斯 (`MultinomialNB`)

基于多项式朴素贝叶斯算法的分类模型，适用于离散特征（如文本分类）。

**模型类型**: `probabilistic`

**参数**:
- `alpha`: 平滑参数，默认为1.0
- `fit_prior`: 是否学习类先验概率，默认为True
- `class_prior`: 类的先验概率，默认为None

**示例**:
```python
model = ModelFactory.create_model(
    'MultinomialNB',
    {'params': {'alpha': 1.0}}
)
```

**特征重要性**: 不支持

**注意**: 特征必须为非负值

---

### 9. 伯努利朴素贝叶斯 (`BernoulliNB`)

基于伯努利朴素贝叶斯算法的分类模型，适用于二值特征。

**模型类型**: `probabilistic`

**参数**:
- `alpha`: 平滑参数，默认为1.0
- `binarize`: 二值化阈值，默认为0.0（None表示假设输入已二值化）
- `fit_prior`: 是否学习类先验概率，默认为True
- `class_prior`: 类的先验概率，默认为None

**示例**:
```python
model = ModelFactory.create_model(
    'BernoulliNB',
    {'params': {'alpha': 1.0, 'binarize': 0.0}}
)
```

**特征重要性**: 不支持

---

### 10. 梯度提升 (`GradientBoosting`)

基于梯度提升决策树的分类模型。

**模型类型**: `tree`

**参数**:
- `loss`: 损失函数，默认为'log_loss'
- `learning_rate`: 学习率，默认为0.1
- `n_estimators`: 提升轮数，默认为100
- `subsample`: 用于拟合基学习器的样本比例，默认为1.0
- `criterion`: 分裂质量的度量标准，默认为'friedman_mse'
- `min_samples_split`: 分裂内部节点所需的最小样本数，默认为2
- `min_samples_leaf`: 叶节点所需的最小样本数，默认为1
- `max_depth`: 树的最大深度，默认为3
- `max_features`: 寻找最佳分裂时考虑的特征数，默认为None
- `random_state`: 随机种子，默认为42

**示例**:
```python
model = ModelFactory.create_model(
    'GradientBoosting',
    {'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}}
)
```

**特征重要性**: 支持（基于分裂改进）

---

### 11. AdaBoost (`AdaBoost`)

基于AdaBoost算法的集成分类模型。

**模型类型**: `ensemble`

**参数**:
- `n_estimators`: 弱学习器的数量，默认为50
- `learning_rate`: 学习率，默认为1.0
- `algorithm`: 提升算法，'SAMME'或'SAMME.R'，默认为'SAMME.R'
- `random_state`: 随机种子，默认为42

**示例**:
```python
model = ModelFactory.create_model(
    'AdaBoost',
    {'params': {'n_estimators': 50, 'learning_rate': 1.0}}
)
```

**特征重要性**: 支持（基于弱学习器的重要性）

---

### 12. 决策树 (`DecisionTree`)

基于决策树算法的分类模型。

**模型类型**: `tree`

**参数**:
- `criterion`: 分裂质量的度量标准，'gini'或'entropy'，默认为'gini'
- `splitter`: 分裂策略，'best'或'random'，默认为'best'
- `max_depth`: 树的最大深度，默认为None
- `min_samples_split`: 分裂内部节点所需的最小样本数，默认为2
- `min_samples_leaf`: 叶节点所需的最小样本数，默认为1
- `max_features`: 寻找最佳分裂时考虑的特征数，默认为None
- `class_weight`: 类别权重，None、'balanced'或字典
- `random_state`: 随机种子，默认为42

**示例**:
```python
model = ModelFactory.create_model(
    'DecisionTree',
    {'params': {'criterion': 'gini', 'max_depth': None}}
)
```

**特征重要性**: 支持（基于Gini重要性或信息增益）

---

### 13. AutoGluon (`AutoGluon`)

基于AutoGluon的自动机器学习模型（需要安装autogluon）。

**模型类型**: `automl`

**参数**:
- 请参考AutoGluon文档

**示例**:
```python
model = ModelFactory.create_model(
    'AutoGluon',
    {'params': {'time_limit': 60, 'presets': 'best_quality'}}
)
```

**特征重要性**: 支持

---

## 模型对比

| 模型 | 类型 | 特征重要性 | 概率输出 | 适用场景 | 训练速度 | 可解释性 |
|------|------|-----------|---------|---------|---------|---------|
| LogisticRegression | 线性 | ✓ | ✓ | 线性可分问题 | 快 | 高 |
| SVM | 核方法 | ✗ | ✓ | 小样本、高维 | 中 | 低 |
| RandomForest | 树集成 | ✓ | ✓ | 通用 | 中 | 中 |
| XGBoost | 树集成 | ✓ | ✓ | 通用、竞赛 | 中 | 中 |
| KNN | 距离 | ✗ | ✓ | 简单分类 | 快(训练)慢(预测) | 中 |
| MLP | 神经网络 | ✗ | ✓ | 复杂非线性 | 慢 | 低 |
| GaussianNB | 概率 | ✗ | ✓ | 高斯分布数据 | 快 | 高 |
| MultinomialNB | 概率 | ✗ | ✓ | 文本分类 | 快 | 高 |
| BernoulliNB | 概率 | ✗ | ✓ | 二值特征 | 快 | 高 |
| GradientBoosting | 树集成 | ✓ | ✓ | 通用 | 慢 | 中 |
| AdaBoost | 集成 | ✓ | ✓ | 通用 | 中 | 中 |
| DecisionTree | 树 | ✓ | ✓ | 简单分类、基学习器 | 快 | 高 |

---

## 自定义模型

您可以通过编写自己的模型类并使用`@ModelFactory.register`装饰器注册来扩展功能。以下是一个简单示例：

```python
# my_custom_model.py
from habit.core.machine_learning.models.base import BaseModel
from habit.core.machine_learning.models.factory import ModelFactory
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class MyCustomClassifier(BaseEstimator, ClassifierMixin):
    """Custom classifier implementation"""
    def __init__(self, param1=1.0, param2='value'):
        self.param1 = param1
        self.param2 = param2
        
    def fit(self, X, y):
        # Implement training logic
        return self
        
    def predict(self, X):
        # Implement prediction logic
        return np.zeros(len(X))
        
    def predict_proba(self, X):
        # Implement probability prediction
        proba = np.zeros((len(X), 2))
        proba[:, 1] = 0.5
        return proba

@ModelFactory.register('MyCustomModel')
class MyCustomModel(BaseModel):
    """Wrapper for custom classifier"""
    
    @property
    def model_type(self) -> str:
        return 'custom'
    
    def __init__(self, config: dict):
        super().__init__(config)
        params = config.get('params', {})
        self.model = MyCustomClassifier(
            param1=params.get('param1', 1.0),
            param2=params.get('param2', 'value')
        )
    
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        return {}
```

然后在您的代码中导入这个模块，即可自动注册：

```python
# Import custom model to register it
import my_custom_model

# Use custom model
from habit.core.machine_learning.models.factory import ModelFactory
model = ModelFactory.create_model('MyCustomModel', {'params': {'param1': 2.0}})
```

---

## 完整工作流程示例

以下是一个完整的模型训练和评估工作流程示例：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import model factory
from habit.core.machine_learning.models.factory import ModelFactory

# 1. Load data
data = pd.read_csv('my_features.csv')
X = data.drop('label', axis=1)
y = data['label']

# 2. Split train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# 4. Create models
models = {
    'LogisticRegression': ModelFactory.create_model(
        'LogisticRegression',
        {'params': {'C': 1.0, 'penalty': 'l2', 'class_weight': 'balanced'}}
    ),
    'SVM': ModelFactory.create_model(
        'SVM',
        {'params': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}}
    ),
    'RandomForest': ModelFactory.create_model(
        'RandomForest',
        {'params': {'n_estimators': 100, 'max_depth': None}}
    ),
    'XGBoost': ModelFactory.create_model(
        'XGBoost',
        {'params': {'n_estimators': 100, 'max_depth': 3}}
    ),
    'KNN': ModelFactory.create_model(
        'KNN',
        {'params': {'n_neighbors': 5, 'weights': 'uniform'}}
    ),
    'MLP': ModelFactory.create_model(
        'MLP',
        {'params': {'hidden_layer_sizes': (100, 50), 'max_iter': 200}}
    ),
    'GaussianNB': ModelFactory.create_model(
        'GaussianNB',
        {'params': {}}
    ),
    'GradientBoosting': ModelFactory.create_model(
        'GradientBoosting',
        {'params': {'n_estimators': 100, 'learning_rate': 0.1}}
    ),
    'AdaBoost': ModelFactory.create_model(
        'AdaBoost',
        {'params': {'n_estimators': 50}}
    ),
    'DecisionTree': ModelFactory.create_model(
        'DecisionTree',
        {'params': {'max_depth': 5}}
    )
}

# 5. Train and evaluate models
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)
    if len(y_train_proba.shape) > 1:
        y_train_proba = y_train_proba[:, 1]
    
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)
    if len(y_test_proba.shape) > 1:
        y_test_proba = y_test_proba[:, 1]
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    results[name] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc
    }
    
    # Print report
    print(f"{name} Performance:")
    print(f"  Train Accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")

# 6. Compare model performance
results_df = pd.DataFrame(results).T
results_df = results_df.reset_index().rename(columns={'index': 'Model'})
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))
```

---

## 注意事项

1. **数据预处理**: 不同模型对数据的要求不同：
   - KNN、SVM、MLP需要特征标准化
   - MultinomialNB要求非负特征
   - 树模型不需要特征标准化

2. **类别不平衡**: 对于类别不平衡问题，建议使用`class_weight='balanced'`参数（适用于LogisticRegression、SVM、RandomForest、DecisionTree）

3. **超参数调优**: 建议使用网格搜索或贝叶斯优化来调整模型超参数

4. **特征重要性**: 只有部分模型支持特征重要性分析（树模型、线性模型）

5. **模型选择**: 
   - 线性可分问题：LogisticRegression
   - 小样本高维：SVM
   - 通用问题：RandomForest、XGBoost、GradientBoosting
   - 快速原型：KNN、GaussianNB、DecisionTree
   - 复杂非线性：MLP、XGBoost
