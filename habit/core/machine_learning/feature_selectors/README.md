# 特征选择器模块使用指南

本模块提供了多种特征选择方法，以统一的接口进行调用。通过注册机制，可以轻松添加和使用各种特征选择器。

## 基本用法

### 1. 在配置文件中指定特征选择方法

特征选择通常通过配置文件来指定，以下是一个示例配置文件：

```yaml
# 特征选择配置
feature_selection_methods:
  - method: icc
    params:
      icc_threshold: 0.8
      outdir: "./output/icc_selection"
  
  - method: correlation
    params:
      method: "pearson"
      threshold: 0.8
      outdir: "./output/correlation_selection"
      
  - method: vif
    params:
      vif_threshold: 5
      outdir: "./output/vif_selection"
      
  - method: lasso
    params:
      alpha: 0.01
      outdir: "./output/lasso_selection"
```

### 2. 在代码中直接调用

```python
from feature_selectors import run_selector, get_available_selectors

# 查看可用的特征选择器
available_selectors = get_available_selectors()
print(f"Available selectors: {available_selectors}")

# 运行LASSO特征选择
selected_features = run_selector(
    'lasso',
    X_train,  # 特征数据（DataFrame）
    y_train,  # 目标变量（Series）
    original_features,  # 原始特征列表
    alpha=0.01,  # LASSO参数
    outdir='./output/lasso_selection'  # 输出目录
)

print(f"Selected features: {selected_features}")
```

## 可用的特征选择器

本模块提供以下特征选择器：

### 1. ICC 选择器 (`icc`)

基于组内相关系数的特征选择，适用于评估特征的可重复性。

**参数**:
- `icc_threshold`: ICC阈值，默认为0.8
- `outdir`: 输出目录

**示例**:
```python
selected_features = run_selector(
    'icc',
    X,
    y,
    selected_features,
    icc_threshold=0.8,
    outdir='./output/icc_selection'
)
```

### 2. 相关性选择器 (`correlation`)

基于特征间相关性的选择，移除高度相关的特征。

**参数**:
- `method`: 相关系数方法，'pearson'、'spearman'或'kendall'，默认为'pearson'
- `threshold`: 相关性阈值，默认为0.8
- `outdir`: 输出目录

**示例**:
```python
selected_features = run_selector(
    'correlation',
    X,
    y,
    selected_features,
    method='pearson',
    threshold=0.8,
    outdir='./output/correlation_selection'
)
```

### 3. VIF选择器 (`vif`)

基于方差膨胀因子的多重共线性检测。

**参数**:
- `vif_threshold`: VIF阈值，默认为5
- `outdir`: 输出目录

**示例**:
```python
selected_features = run_selector(
    'vif',
    X,
    y,
    selected_features,
    vif_threshold=5,
    outdir='./output/vif_selection'
)
```

### 4. LASSO选择器 (`lasso`)

基于L1正则化的特征选择。

**参数**:
- `alpha`: 正则化强度，默认为0.01
- `outdir`: 输出目录

**示例**:
```python
selected_features = run_selector(
    'lasso',
    X,
    y,
    selected_features,
    alpha=0.01,
    outdir='./output/lasso_selection'
)
```

### 5. 逐步回归选择器 (`stepwise`)

使用逐步回归方法选择特征。

**参数**:
- `direction`: 方向，'forward'、'backward'或'bidirectional'，默认为'bidirectional'
- `criterion`: 评价标准，'aic'、'bic'或'r2'，默认为'aic'
- `outdir`: 输出目录

**示例**:
```python
selected_features = run_selector(
    'stepwise',
    X,
    y,
    selected_features,
    direction='bidirectional',
    criterion='aic',
    outdir='./output/stepwise_selection'
)
```

### 6. 单变量逻辑回归选择器 (`univariate_logistic`)

基于单变量逻辑回归的P值选择特征。

**参数**:
- `p_threshold`: P值阈值，默认为0.05
- `outdir`: 输出目录

**示例**:
```python
selected_features = run_selector(
    'univariate_logistic',
    X,
    y,
    selected_features,
    p_threshold=0.05,
    outdir='./output/univariate_selection'
)
```

### 7. mRMR选择器 (`mrmr`)

最小冗余最大相关算法，平衡特征相关性和冗余度。

**参数**:
- `n_features_to_select`: 要选择的特征数量，默认为20
- `outdir`: 输出目录

**示例**:
```python
selected_features = run_selector(
    'mrmr',
    X,
    y,
    selected_features,
    n_features_to_select=20,
    outdir='./output/mrmr_selection'
)
```

### 8. ANOVA选择器 (`anova`)

基于方差分析的F值特征选择。

**参数**:
- `n_features_to_select`: 要选择的特征数量，默认为20
- `plot_importance`: 是否绘制特征重要性图，默认为True
- `outdir`: 输出目录

**示例**:
```python
selected_features = run_selector(
    'anova',
    X,
    y,
    selected_features,
    n_features_to_select=20,
    plot_importance=True,
    outdir='./output/anova_selection'
)
```

### 9. Boruta选择器 (`boruta`)

基于随机森林的Boruta算法进行特征选择。

**参数**:
- `n_estimators`: 随机森林的树数量，默认为100
- `max_iter`: 最大迭代次数，默认为100
- `perc`: 保留的影子特征百分比，默认为100
- `alpha`: 显著性水平，默认为0.05
- `random_state`: 随机数种子，默认为42
- `include_tentative`: 是否包含可能相关的特征，默认为True
- `plot_importance`: 是否绘制特征重要性图，默认为True
- `outdir`: 输出目录

**示例**:
```python
selected_features = run_selector(
    'boruta',
    X,
    y,
    selected_features,
    n_estimators=100,
    max_iter=100,
    include_tentative=True,
    outdir='./output/boruta_selection'
)
```

## 自定义特征选择器

您可以通过编写自己的特征选择器函数并使用`@register_selector`装饰器注册来扩展功能。以下是一个简单示例：

```python
# my_selector.py
from feature_selectors import register_selector
import numpy as np
import pandas as pd
from typing import List

@register_selector('my_custom_selector')
def my_custom_selector(
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: List[str],
        my_param: float = 0.5,
        outdir: str = None,
        **kwargs
    ) -> List[str]:
    """
    自定义特征选择器
    
    Args:
        X: 特征数据
        y: 目标变量
        selected_features: 预选特征列表
        my_param: 自定义参数
        outdir: 输出目录
        **kwargs: 其他参数
        
    Returns:
        List[str]: 选择的特征名列表
    """
    # 实现自定义的特征选择逻辑
    # ...
    
    # 返回选择的特征名列表
    return selected_features_subset
```

然后在您的代码中导入这个模块，即可自动注册：

```python
# 导入自定义选择器
import my_selector

# 查看可用选择器（现在应该包含'my_custom_selector'）
from feature_selectors import get_available_selectors
print(get_available_selectors())

# 使用自定义选择器
from feature_selectors import run_selector
selected = run_selector('my_custom_selector', X, y, features, my_param=0.7)
```

## 完整工作流程示例

以下是一个完整的特征选择工作流程示例：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 导入特征选择器
from feature_selectors import run_selector, get_available_selectors

# 1. 加载数据
data = pd.read_csv('my_features.csv')
X = data.drop('label', axis=1)
y = data['label']

# 2. 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 初始特征列表
original_features = X_train.columns.tolist()

# 4. 特征选择流程
# 4.1 ICC筛选 - 去除不可重复的特征
selected_features = run_selector(
    'icc',
    X_train,
    y_train,
    original_features,
    icc_threshold=0.8,
    outdir='./output/step1_icc'
)
print(f"After ICC: {len(selected_features)} features")

# 4.2 相关性筛选 - 去除高度相关的特征
selected_features = run_selector(
    'correlation',
    X_train,
    y_train,
    selected_features,
    method='pearson',
    threshold=0.8,
    outdir='./output/step2_correlation'
)
print(f"After correlation: {len(selected_features)} features")

# 4.3 单变量逻辑回归 - 去除不显著的特征
selected_features = run_selector(
    'univariate_logistic',
    X_train,
    y_train,
    selected_features,
    p_threshold=0.05,
    outdir='./output/step3_univariate'
)
print(f"After univariate: {len(selected_features)} features")

# 4.4 LASSO选择 - 最终特征选择
final_features = run_selector(
    'lasso',
    X_train,
    y_train,
    selected_features,
    alpha=0.01,
    outdir='./output/step4_lasso'
)
print(f"Final features: {final_features}")

# 5. 保存选择的特征
import json
with open('./output/selected_features.json', 'w') as f:
    json.dump({
        'selected_features': final_features,
        'n_features': len(final_features)
    }, f, indent=4)

# 6. 可视化最终选择的特征
import matplotlib.pyplot as plt
import seaborn as sns

X_final = X_train[final_features]
correlation = X_final.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation of Selected Features')
plt.tight_layout()
plt.savefig('./output/selected_features_correlation.pdf')
plt.close()

print("Feature selection completed successfully!") 