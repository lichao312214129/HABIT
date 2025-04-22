# 聚类算法模块

## 设计原则

此模块遵循以下软件设计原则：

1. **单一职责原则**：每个聚类算法类只负责实现特定算法的核心功能
2. **开放封闭原则**：通过基类和注册机制，可以轻松扩展新的聚类算法而无需修改现有代码
3. **依赖倒置原则**：高层模块依赖抽象接口而非具体实现
4. **DRY原则**：通用功能集中在基类实现，避免代码重复

## 模块结构

### 1. 基类与注册机制

- `BaseClustering`：所有聚类算法的抽象基类
- `register_clustering`：装饰器，用于注册聚类算法
- `get_clustering_algorithm`：工厂函数，通过名称获取聚类算法实例

### 2. 验证方法配置

- `cluster_validation_methods.py`：定义不同聚类算法支持的验证方法和默认配置
- 通过此配置实现算法与验证方法的解耦，避免硬编码

### 3. 具体算法实现

每个聚类算法类只需实现以下三个核心方法：

- `__init__`：初始化算法参数
- `fit`：训练聚类模型
- `predict`：预测聚类标签

所有其他功能，包括寻找最优聚类数、计算各种评分指标、绘制评分曲线等，均由基类提供。这种设计极大地减少了代码重复，提高了维护性。

## 扩展新算法

要添加新的聚类算法，按照以下步骤：

1. 创建新的类，继承自`BaseClustering`
2. 使用`@register_clustering`装饰器注册算法
3. 仅实现三个核心方法：`__init__`、`fit`和`predict`
4. 在`cluster_validation_methods.py`中添加该算法支持的验证方法配置

示例：

```python
@register_clustering('new_algorithm')
class NewAlgorithmClustering(BaseClustering):
    def __init__(self, n_clusters=None, random_state=0, **kwargs):
        super().__init__(n_clusters, random_state)
        # 算法特有的初始化参数
        self.specific_param = kwargs.get('specific_param', 'default_value')
        self.kwargs = kwargs
        
    def fit(self, X):
        # 实现训练逻辑
        self.model = YourAlgorithm(n_clusters=self.n_clusters, random_state=self.random_state)
        self.model.fit(X)
        self.labels_ = self.model.labels_
        return self
        
    def predict(self, X):
        # 实现预测逻辑
        return self.model.predict(X)
```

## 通用功能（由基类提供）

基类`BaseClustering`提供以下通用功能，所有聚类算法无需重新实现：

1. `find_optimal_clusters`：寻找最优聚类数
2. `auto_select_best_n_clusters`：自动选择最佳聚类数
3. `plot_scores`：绘制各种评分方法的曲线图
4. 各种评分计算方法：
   - `calculate_silhouette_scores`：计算轮廓系数
   - `calculate_calinski_harabasz_scores`：计算Calinski-Harabasz指数
   - `calculate_inertia_scores`：计算K-Means惯性（SSE）
   - `calculate_bic_scores`：计算GMM贝叶斯信息准则
   - `calculate_aic_scores`：计算GMM赤池信息准则

## 用法示例

```python
from habitat_analysis.clustering import get_clustering_algorithm

# 创建聚类算法实例
clustering = get_clustering_algorithm('kmeans', random_state=42)

# 寻找最优聚类数
best_n, scores = clustering.find_optimal_clusters(X, min_clusters=2, max_clusters=10)

# 使用最优聚类数训练模型
clustering.fit(X)

# 预测标签
labels = clustering.predict(X)

# 绘制评分曲线
clustering.plot_scores(scores, show=True, save_path='scores.png')
``` 