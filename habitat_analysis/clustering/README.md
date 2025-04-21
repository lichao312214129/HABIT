# 聚类算法模块

本模块提供了栖息地分析中使用的聚类算法实现。

## 内置算法

当前已实现以下聚类算法：

- `kmeans`: K均值聚类算法
- `gmm`: 高斯混合模型聚类算法

## 如何自定义新的聚类算法

您可以通过以下步骤定义自己的聚类算法：

1. 在`clustering`目录下创建一个新的Python文件，命名为`your_algorithm_clustering.py`
2. 从`BaseClustering`类继承一个新类，并使用`register_clustering`装饰器注册它
3. 实现所有必需的方法：`fit`, `predict`, `find_optimal_clusters`

### 示例

您可以参考`custom_clustering_template.py`作为模板：

```python
from habitat_clustering.clustering.base_clustering import BaseClustering, register_clustering

@register_clustering('your_algorithm')
class YourAlgorithmClustering(BaseClustering):
    def __init__(self, n_clusters=None, random_state=0, **kwargs):
        super().__init__(n_clusters, random_state)
        # 自定义初始化代码
        
    def fit(self, X):
        # 实现聚类训练逻辑
        return self
        
    def predict(self, X):
        # 实现预测逻辑
        return labels
        
    def find_optimal_clusters(self, X, min_clusters=2, max_clusters=10, methods=None, show_progress=True):
        # 实现找到最佳聚类数的逻辑
        return best_n_clusters, self.scores
```

### 自动发现与注册

您不需要修改`__init__.py`文件！只要您的文件命名符合规范（`*_clustering.py`），系统会在运行时自动发现并注册您的算法。

### 使用自定义算法

一旦注册，您可以在配置文件中指定您的算法：

```yaml
HabitatsSegmention:
  supervoxel:
    algorithm: your_algorithm  # 这里使用您注册的算法名
    n_clusters: 50
```

或者在代码中直接使用：

```python
from habitat_clustering.clustering.base_clustering import get_clustering_algorithm

# 创建您的算法实例
clustering = get_clustering_algorithm('your_algorithm', n_clusters=50)
``` 