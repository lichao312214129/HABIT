# 聚类验证方法模块

## 简介

`cluster_validation_methods.py`模块提供了不同聚类算法与其适用的最优聚类数确定方法之间的映射关系。该模块主要解决以下问题：

1. 为每种聚类算法定义了支持的验证方法集合
2. 能够验证用户输入的聚类算法和验证方法是否匹配
3. 提供默认的验证方法配置
4. 获取验证方法的描述和优化方向

## 当前支持的聚类算法

目前支持以下聚类算法：

- **kmeans**: K-Means聚类算法
  - 默认验证方法: `silhouette`, `calinski_harabasz`, `inertia`
  - 支持的验证方法: `silhouette`, `calinski_harabasz`, `inertia`, `gap`, `davies_bouldin`

- **gmm**: 高斯混合模型
  - 默认验证方法: `bic`, `aic`, `silhouette`
  - 支持的验证方法: `bic`, `aic`, `silhouette`, `calinski_harabasz`

- **spectral**: 谱聚类
  - 默认验证方法: `silhouette`, `calinski_harabasz`
  - 支持的验证方法: `silhouette`, `calinski_harabasz`, `davies_bouldin`

- **hierarchical**: 层次聚类
  - 默认验证方法: `silhouette`, `calinski_harabasz`
  - 支持的验证方法: `silhouette`, `calinski_harabasz`, `davies_bouldin`, `cophenetic`

- **dbscan**: DBSCAN密度聚类
  - 默认验证方法: `silhouette`, `calinski_harabasz`
  - 支持的验证方法: `silhouette`, `calinski_harabasz`

- **som**: 自组织映射网络
  - 默认验证方法: `quantization_error`, `topographic_error`
  - 支持的验证方法: `quantization_error`, `topographic_error`, `silhouette`

## 使用方法

### 1. 获取聚类算法支持的验证方法

```python
from habitat_clustering.clustering.cluster_validation_methods import get_validation_methods

# 获取kmeans支持的验证方法
validation_info = get_validation_methods('kmeans')
print(f"默认方法: {validation_info['default']}")
print(f"所有支持的方法: {list(validation_info['methods'].keys())}")
```

### 2. 检查验证方法是否适用于特定聚类算法

```python
from habitat_clustering.clustering.cluster_validation_methods import is_valid_method_for_algorithm

# 检查inertia方法是否适用于kmeans
valid = is_valid_method_for_algorithm('kmeans', 'inertia')
print(f"'inertia'方法对kmeans是否有效: {valid}")  # True

# 检查aic方法是否适用于kmeans
valid = is_valid_method_for_algorithm('kmeans', 'aic')
print(f"'aic'方法对kmeans是否有效: {valid}")  # False
```

### 3. 获取聚类算法的默认验证方法

```python
from habitat_clustering.clustering.cluster_validation_methods import get_default_methods

# 获取gmm的默认验证方法
default_methods = get_default_methods('gmm')
print(f"GMM的默认验证方法: {default_methods}")  # ['bic', 'aic', 'silhouette']
```

### 4. 获取所有支持的聚类算法

```python
from habitat_clustering.clustering.cluster_validation_methods import get_all_clustering_algorithms

# 获取所有支持的聚类算法
algorithms = get_all_clustering_algorithms()
print(f"支持的聚类算法: {algorithms}")
```

### 5. 获取验证方法的优化方向

```python
from habitat_clustering.clustering.cluster_validation_methods import get_optimization_direction

# 获取inertia的优化方向
direction = get_optimization_direction('kmeans', 'inertia')
print(f"inertia的优化方向: {direction}")  # 'minimize'

# 获取silhouette的优化方向
direction = get_optimization_direction('kmeans', 'silhouette')
print(f"silhouette的优化方向: {direction}")  # 'maximize'
```

### 6. 获取验证方法的描述

```python
from habitat_clustering.clustering.cluster_validation_methods import get_method_description

# 获取inertia的描述
desc = get_method_description('kmeans', 'inertia')
print(f"inertia的描述: {desc}")  # '惯性(Inertia)，即样本到最近聚类中心的距离平方和，值越小越好'
```

## 在HabitatAnalysis中使用

`HabitatAnalysis`类已经集成了验证方法的检查功能。当用户使用不匹配的聚类算法和验证方法时，会自动使用适合当前聚类算法的默认验证方法，并输出警告信息。

```python
from habitat_clustering.habitat_analysis import HabitatAnalysis

# 使用kmeans聚类，但指定了bic验证方法（不适用于kmeans）
analysis = HabitatAnalysis(
    root_folder="data_dir",
    habitat_clustering_method="kmeans",
    habitat_cluster_selection_method="bic",  # 不适用于kmeans
    verbose=True
)

# 输出警告并使用默认的方法（silhouette, calinski_harabasz, inertia）
```

## 添加自定义聚类算法和验证方法

如果需要添加新的聚类算法和验证方法，可以修改`CLUSTERING_VALIDATION_METHODS`字典：

```python
# 在cluster_validation_methods.py中添加
CLUSTERING_VALIDATION_METHODS['my_clustering'] = {
    'default': ['my_method', 'silhouette'],
    'methods': {
        'my_method': {
            'description': '我的自定义方法，值越大越好',
            'optimization': 'maximize'
        },
        'silhouette': {
            'description': '轮廓系数(Silhouette Score)，值越大越好',
            'optimization': 'maximize'
        }
    }
}
``` 