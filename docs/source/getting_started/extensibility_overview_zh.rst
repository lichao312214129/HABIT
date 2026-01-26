扩展性概述
==========

HABIT 的核心设计理念是"开箱即用，无限扩展"。通过工厂模式和注册机制，用户可以轻松添加自定义组件，无需修改核心代码。

扩展机制概述
------------

HABIT 使用工厂模式（Factory Pattern）和注册机制（Registry Pattern）来实现灵活的扩展系统。

**工厂模式：**

工厂模式是一种创建型设计模式，它提供了一种创建对象的最佳方式。在 HABIT 中，每个可扩展的组件都有对应的工厂类。

**注册机制：**

注册机制允许用户通过装饰器（Decorator）注册自定义组件，系统会自动发现和加载这些组件。

**扩展的优势：**

1. **无需修改核心代码**: 自定义组件独立于核心代码
2. **自动发现**: 系统自动发现和加载注册的组件
3. **统一接口**: 所有自定义组件都遵循统一的接口规范
4. **易于维护**: 自定义组件可以独立维护和升级

可扩展的组件
------------

HABIT 支持以下类型的自定义扩展：

**1. 自定义预处理器**

- **用途**: 添加自定义的图像预处理方法
- **工厂类**: `PreprocessorFactory`
- **注册装饰器**: `@PreprocessorFactory.register("name")`
- **基类**: `BasePreprocessor`
- **模板文件**: `habit/core/preprocessing/custom_preprocessor_template.py`

**示例：**

.. code-block:: python

   from habit.core.preprocessing.preprocessor_factory import PreprocessorFactory
   from habit.core.preprocessing.base_preprocessor import BasePreprocessor

   @PreprocessorFactory.register("my_preprocessor")
   class MyPreprocessor(BasePreprocessor):
       def __init__(self, keys, **kwargs):
           super().__init__(keys=keys)
           # 初始化参数

       def __call__(self, data):
           # 实现预处理逻辑
           return data

**2. 自定义特征提取器**

- **用途**: 添加自定义的聚类特征提取方法
- **工厂类**: `FeatureExtractorFactory`
- **注册装饰器**: `@register_feature_extractor("name")`
- **基类**: `BaseClusteringExtractor`
- **模板文件**: `habit/core/habit_analysis/extractors/custom_feature_extractor_template.py`

**示例：**

.. code-block:: python

   from habit.core.habit_analysis.extractors.base_extractor import BaseClusteringExtractor
   from habit.core.habit_analysis.extractors.base_extractor import register_feature_extractor

   @register_feature_extractor('my_feature_extractor')
   class MyFeatureExtractor(BaseClusteringExtractor):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           self.feature_names = ['feature1', 'feature2', 'feature3']

       def extract_features(self, image_data, **kwargs):
           # 实现特征提取逻辑
           return features

**3. 自定义聚类算法**

- **用途**: 添加自定义的聚类算法
- **工厂类**: `ClusteringFactory`
- **注册装饰器**: `@register_clustering_algorithm("name")`
- **基类**: `BaseClusteringAlgorithm`
- **模板文件**: `habit/core/habit_analysis/algorithms/custom_clustering_template.py`

**示例：**

.. code-block:: python

   from habit.core.habit_analysis.algorithms.base_clustering import BaseClusteringAlgorithm
   from habit.core.habit_analysis.algorithms.base_clustering import register_clustering_algorithm

   @register_clustering_algorithm('my_clustering')
   class MyClusteringAlgorithm(BaseClusteringAlgorithm):
       def __init__(self, n_clusters, random_state, **kwargs):
           super().__init__(n_clusters, random_state, **kwargs)
           # 初始化参数

       def fit_predict(self, X):
           # 实现聚类逻辑
           return labels

**4. 自定义策略**

- **用途**: 添加自定义的生境分割策略
- **工厂类**: `StrategyFactory`
- **注册方式**: 在 `STRATEGY_REGISTRY` 中注册
- **基类**: `BaseClusteringStrategy`
- **模板文件**: `habit/core/habit_analysis/strategies/custom_strategy_template.py`

**示例：**

.. code-block:: python

   from habit.core.habit_analysis.strategies.base_strategy import BaseClusteringStrategy

   class MyStrategy(BaseClusteringStrategy):
       def run(self, subjects, **kwargs):
           # 实现策略逻辑
           return results

   # 在策略工厂中注册
   from habit.core.habit_analysis.strategies import STRATEGY_REGISTRY
   STRATEGY_REGISTRY["my_strategy"] = MyStrategy

**5. 自定义机器学习模型**

- **用途**: 添加自定义的机器学习模型
- **工厂类**: `ModelFactory`
- **注册装饰器**: `@ModelFactory.register("name")`
- **基类**: `BaseModel`
- **模板文件**: `habit/core/machine_learning/models/custom_model_template.py`

**示例：**

.. code-block:: python

   from habit.core.machine_learning.models.base import BaseModel
   from habit.core.machine_learning.models.factory import ModelFactory

   @ModelFactory.register("my_model")
   class MyModel(BaseModel):
       def __init__(self, config):
           super().__init__(config)
           # 初始化模型

       def fit(self, X, y):
           # 实现训练逻辑
           pass

       def predict(self, X):
           # 实现预测逻辑
           return predictions

       def predict_proba(self, X):
           # 实现概率预测逻辑
           return probabilities

       def get_feature_importance(self):
           # 返回特征重要性
           return importance_dict

**6. 自定义特征选择器**

- **用途**: 添加自定义的特征选择方法
- **工厂类**: `SelectorRegistry`
- **注册装饰器**: `@register_selector("name")`
- **基类**: 无特定基类，使用函数式接口
- **模板文件**: `habit/core/machine_learning/feature_selectors/custom_selector_template.py`

**示例：**

.. code-block:: python

   from habit.core.machine_learning.feature_selectors.selector_registry import register_selector

   @register_selector("my_selector", display_name="My Selector", default_before_z_score=False)
   def my_selector(X, y, selected_features, **kwargs):
       # 实现特征选择逻辑
       return selected_features

扩展步骤
--------

**步骤 1: 复制模板文件**

从对应的模板文件开始，模板文件位于：

- `habit/core/preprocessing/custom_preprocessor_template.py`
- `habit/core/habit_analysis/extractors/custom_feature_extractor_template.py`
- `habit/core/habit_analysis/algorithms/custom_clustering_template.py`
- `habit/core/machine_learning/models/custom_model_template.py`
- `habit/core/machine_learning/feature_selectors/custom_selector_template.py`

**步骤 2: 修改类名和注册名称**

将模板类名和注册名称修改为您自己的名称。

**步骤 3: 实现核心逻辑**

根据基类的接口要求，实现核心逻辑。

**步骤 4: 测试自定义组件**

在配置文件中使用您的自定义组件，测试是否正常工作。

**步骤 5: 分享和贡献**

如果您的自定义组件对其他用户也有价值，可以考虑贡献到 HABIT 项目。

配置文件中的使用
--------------

在配置文件中使用自定义组件非常简单，只需指定注册名称即可。

**预处理器示例：**

.. code-block:: yaml

   Preprocessing:
     my_preprocessor:
       images: [T1, T2]
       param1: value1
       param2: value2

**2. 在配置文件中使用：**

.. code-block:: yaml

   FeatureConstruction:
     voxel_level:
       method: my_feature_extractor(raw(T1), raw(T2))
       params:
         param1: value1

**聚类算法示例：**

.. code-block:: yaml

   HabitatsSegmention:
     supervoxel:
       algorithm: my_clustering
       n_clusters: 50

**模型示例：**

.. code-block:: yaml

   models:
     MyModel:
       params:
         param1: value1
         param2: value2

**特征选择器示例：**

.. code-block:: yaml

   feature_selection_methods:
     - method: my_selector
       params:
         param1: value1

自动发现机制
------------

HABIT 使用自动发现机制来加载自定义组件。当您在配置文件中使用自定义组件时，系统会：

1. 检查组件是否已在注册表中
2. 如果未注册，尝试动态导入对应的模块
3. 自动注册发现的组件
4. 创建组件实例

**发现规则：**

- 预处理器: `habit/core/preprocessing/` 目录下的 Python 文件
- 特征提取器: `habit/core/habit_analysis/extractors/` 目录下的 Python 文件
- 聚类算法: `habit/core/habit_analysis/algorithms/` 目录下的 Python 文件
- 模型: `habit/core/machine_learning/models/` 目录下的 Python 文件
- 特征选择器: `habit/core/machine_learning/feature_selectors/` 目录下的 Python 文件

**命名规则：**

自定义组件的文件名应该遵循以下命名规则：

- 预处理器: `{name}_preprocessor.py`
- 特征提取器: `{name}_feature_extractor.py`
- 聚类算法: `{name}_clustering.py`
- 模型: `{name}.py`（类名与文件名匹配）
- 特征选择器: `{name}_selector.py`

扩展最佳实践
------------

**1. 遵循接口规范**

确保您的自定义组件完全实现了基类的所有接口方法。

**2. 提供清晰的文档**

为您的自定义组件提供清晰的文档，包括：

- 功能描述
- 参数说明
- 使用示例
- 适用场景

**3. 处理异常情况**

在实现中添加适当的异常处理，确保组件的健壮性。

**4. 提供默认参数**

为所有参数提供合理的默认值，降低使用门槛。

**5. 添加单元测试**

为您的自定义组件编写单元测试，确保功能的正确性。

**6. 性能优化**

考虑性能优化，特别是对于计算密集型的组件。

**7. 日志记录**

添加适当的日志记录，便于调试和问题排查。

下一步
-------

现在您已经了解了 HABIT 的扩展机制，可以：

- 查看 :doc:`../customization/index_zh` 了解详细的扩展指南
- 参考 :doc:`../user_guide/index_zh` 了解各个模块的使用方法
- 查看 :doc:`../tutorials/index_zh` 学习完整的教程
