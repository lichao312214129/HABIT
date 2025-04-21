# 模型注册系统使用指南

本模块提供了统一的机器学习模型注册和管理系统，使用工厂模式设计，便于扩展和自定义。

## 基本用法

### 1. 在配置文件中指定模型

模型通常通过配置文件来指定，以下是一个示例配置文件：

```yaml
# 模型配置
models:
  LogisticRegression:
    type: LogisticRegression  # 模型类型名称，对应注册的类型
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
```

### 2. 在代码中直接创建模型

```python
from models import create_model, get_available_models

# 查看可用的模型
available_models = get_available_models()
print(f"Available models: {available_models}")

# 创建逻辑回归模型
log_reg = create_model(
    'LogisticRegression',
    C=1.0,
    penalty='l2',
    solver='liblinear'
)

# 创建SVM模型
svm = create_model(
    'SVM',
    C=1.0,
    kernel='rbf',
    gamma='scale'
)

# 使用模型
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
```

## 可用的模型

本模块提供以下预定义模型：

### 1. 逻辑回归 (`LogisticRegression`)

基于逻辑回归算法的分类模型。

**参数**:
- `C`: 正则化强度的倒数，默认为1.0
- `penalty`: 惩罚项类型，'l1'、'l2'、'elasticnet'或'none'，默认为'l2'
- `solver`: 优化算法，默认为'liblinear'
- `max_iter`: 最大迭代次数，默认为1000
- `random_state`: 随机种子，默认为42
- `class_weight`: 类别权重，None、'balanced'或字典

**示例**:
```python
model = create_model(
    'LogisticRegression',
    C=1.0,
    penalty='l2',
    solver='liblinear'
)
```

### 2. 支持向量机 (`SVM`)

基于支持向量机算法的分类模型。

**参数**:
- `C`: 正则化参数，默认为1.0
- `kernel`: 核函数类型，'linear'、'poly'、'rbf'或'sigmoid'，默认为'rbf'
- `gamma`: 核系数，'scale'、'auto'或浮点数，默认为'scale'
- `probability`: 是否启用概率估计，默认为True
- `class_weight`: 类别权重，None、'balanced'或字典
- `random_state`: 随机种子，默认为42

**示例**:
```python
model = create_model(
    'SVM',
    C=1.0,
    kernel='rbf',
    gamma='scale'
)
```

### 3. XGBoost (`XGBoost`)

基于梯度提升决策树的分类模型。

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
model = create_model(
    'XGBoost',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1
)
```

### 4. 随机森林 (`RandomForest`)

基于随机森林算法的分类模型。

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
model = create_model(
    'RandomForest',
    n_estimators=100,
    max_depth=None,
    max_features='sqrt'
)
```

### 5. 自定义集成模型 (`CustomEnsemble`)

组合多个基础模型的集成分类器。

**参数**:
- `models`: 模型配置列表，每个元素包含'name'和'params'
- `weights`: 模型权重列表，默认为None
- `voting`: 投票方式，'hard'或'soft'，默认为'soft'

**示例**:
```python
model = create_model(
    'CustomEnsemble',
    models=[
        {'name': 'LogisticRegression', 'params': {'C': 1.0}},
        {'name': 'SVM', 'params': {'kernel': 'linear'}},
        {'name': 'RandomForest', 'params': {'n_estimators': 100}}
    ],
    weights=[0.3, 0.3, 0.4],
    voting='soft'
)
```

## 自定义模型

您可以通过编写自己的模型创建函数并使用`@register_model`装饰器注册来扩展功能。以下是一个简单示例：

```python
# my_model.py
from models import register_model
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class MyCustomModel(BaseEstimator, ClassifierMixin):
    """自定义模型"""
    def __init__(self, param1=1.0, param2='value'):
        self.param1 = param1
        self.param2 = param2
        
    def fit(self, X, y):
        # 实现训练逻辑
        return self
        
    def predict(self, X):
        # 实现预测逻辑
        return np.zeros(len(X))
        
    def predict_proba(self, X):
        # 实现概率预测
        proba = np.zeros((len(X), 2))
        proba[:, 1] = 0.5  # 默认概率值
        return proba

@register_model('MyCustomModel')
def my_custom_model(
        param1: float = 1.0,
        param2: str = 'value',
        **kwargs
    ) -> MyCustomModel:
    """
    创建自定义模型
    
    Args:
        param1: 自定义参数1
        param2: 自定义参数2
        **kwargs: 其他参数
        
    Returns:
        MyCustomModel: 模型实例
    """
    return MyCustomModel(param1=param1, param2=param2)
```

然后在您的代码中导入这个模块，即可自动注册：

```python
# 导入自定义模型
import my_model

# 查看可用模型（现在应该包含'MyCustomModel'）
from models import get_available_models
print(get_available_models())

# 使用自定义模型
from models import create_model
model = create_model('MyCustomModel', param1=2.0, param2='custom')
```

## 完整工作流程示例

以下是一个完整的模型训练和评估工作流程示例：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 导入模型创建功能
from models import create_model, get_available_models

# 1. 加载数据
data = pd.read_csv('my_features.csv')
X = data.drop('label', axis=1)
y = data['label']

# 2. 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 创建模型列表
models = {
    'LogisticRegression': create_model(
        'LogisticRegression',
        C=1.0,
        penalty='l2',
        class_weight='balanced'
    ),
    'SVM': create_model(
        'SVM',
        C=1.0,
        kernel='rbf',
        gamma='scale'
    ),
    'RandomForest': create_model(
        'RandomForest',
        n_estimators=100,
        max_depth=None
    ),
    'XGBoost': create_model(
        'XGBoost',
        n_estimators=100,
        max_depth=3
    ),
    'Ensemble': create_model(
        'CustomEnsemble',
        models=[
            {'name': 'LogisticRegression', 'params': {'C': 1.0}},
            {'name': 'SVM', 'params': {'kernel': 'linear'}},
            {'name': 'RandomForest', 'params': {'n_estimators': 100}}
        ],
        weights=[0.3, 0.3, 0.4]
    )
}

# 5. 训练和评估模型
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # 训练模型
    model.fit(X_train_scaled, y_train)
    
    # 训练集预测
    y_train_pred = model.predict(X_train_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    
    # 测试集预测
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # 计算性能指标
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
    
    # 打印报告
    print(f"{name} Performance:")
    print(f"  Train Accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred))

# 6. 比较模型性能
import matplotlib.pyplot as plt
import seaborn as sns

# 创建性能比较表格
results_df = pd.DataFrame(results).T
results_df = results_df.reset_index().rename(columns={'index': 'Model'})

# 绘制性能对比图
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='test_auc', data=results_df)
plt.title('Test AUC Comparison')
plt.ylim(0.5, 1.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_performance_comparison.pdf')

print("\nModel evaluation completed!") 