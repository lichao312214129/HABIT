二次开发与组件扩展指南
======================

HABIT 的底层架构高度解耦，允许开发者在不修改核心源码的情况下，通过**注册表（Registry）**机制注入自定义的算法组件。

本指南基于 HABIT 的底层代码契约（Code Contracts），详细说明如何扩展预处理器、机器学习模型和特征选择器。

扩展机制的核心契约
------------------

HABIT 的扩展严格遵循以下三个契约：

1. **逻辑契约**：继承指定的基类（如 ``BasePreprocessor``）或实现特定的函数签名（如 ``SelectorContext -> List[str]``）。
2. **注册契约**：使用对应的装饰器（如 ``@PreprocessorFactory.register``）将组件名称注册到全局工厂中。
3. **参数契约**：定义一个 Pydantic Schema，并通过 ``ParamSchemaRegistry.register`` 注册，以实现 YAML 参数的强类型校验和 GUI 表单的自动生成。

.. important::
   
   **模块导入机制**：注册装饰器仅在 Python 模块被 ``import`` 时执行。如果你在 HABIT 源码外部开发插件，必须确保在运行前导入了你的模块。若在源码内开发，请在对应子目录的 ``__init__.py`` 中导入。

实战一：自定义预处理步骤 (ClassRegistry)
----------------------------------------

预处理器是基于类的组件，由 ``PreprocessorFactory`` 管理。

**1. 实现逻辑与注册名称**

继承 ``BasePreprocessor``，实现 ``__call__`` 方法，并使用 ``@PreprocessorFactory.register`` 注册。

.. code-block:: python

   import numpy as np
   from scipy.ndimage import gaussian_filter
   from habit.core.preprocessing.preprocessor_factory import PreprocessorFactory
   from habit.core.preprocessing.base_preprocessor import BasePreprocessor

   @PreprocessorFactory.register("my_gaussian_filter")
   class MyGaussianFilter(BasePreprocessor):
       def __init__(self, keys, allow_missing_keys=False, **kwargs):
           super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
           # 提取参数
           self.sigma = kwargs.get('sigma', 1.0)

       def __call__(self, data):
           self._check_keys(data)
           for key in self.keys:
               data[key] = gaussian_filter(data[key], sigma=self.sigma)
           return data

**2. 定义参数 Schema 并注册**

在 ``habit/core/schemas/steps/preprocessing.py``\ （或你的插件模块中）定义参数模型并注册。

.. code-block:: python

   from pydantic import BaseModel, Field
   from habit.core.schemas.registry import ParamSchemaRegistry

   class MyGaussianFilterParams(BaseModel):
       sigma: float = Field(default=1.0, description="Gaussian kernel standard deviation.")

   # 注册到 ParamSchemaRegistry (domain="preprocessing")
   ParamSchemaRegistry.register("preprocessing", "my_gaussian_filter", MyGaussianFilterParams)

**3. 在 YAML 中调用**

.. code-block:: yaml

   preprocessing:
     my_gaussian_filter:
       images: [T1, T2]
       sigma: 2.5

实战二：自定义特征选择器 (CallableRegistry)
-------------------------------------------

与 scikit-learn 要求编写完整的 ``BaseEstimator`` 类不同，**HABIT 的特征选择器被设计为纯函数**。底层 ``pipeline_builder.py`` 会自动将这些函数包装为 sklearn 兼容的 Transformer。

**1. 实现函数与注册名称**

特征选择器函数接收一个 ``SelectorContext`` 对象（包含 ``X``, ``y``, ``selected_features`` 等），并返回保留的特征名列表 ``List[str]``。

.. code-block:: python

   from typing import List
   from habit.core.machine_learning.feature_selectors.selector_registry import (
       SelectorRegistry, 
       SelectorContext
   )

   @SelectorRegistry.register("my_variance_selector", display_name="Custom Variance")
   def my_variance_selector(context: SelectorContext, threshold: float = 0.0) -> List[str]:
       """
       自定义方差特征选择器。
       """
       X = context.X
       # 计算方差
       variances = X.var(axis=0)
       # 筛选大于阈值的特征
       retained_features = variances[variances > threshold].index.tolist()
       
       context.logger.info(f"Retained {len(retained_features)} features.")
       return retained_features

**2. 定义参数 Schema 并注册**

.. code-block:: python

   from pydantic import BaseModel, Field
   from habit.core.schemas.registry import ParamSchemaRegistry

   class MyVarianceParams(BaseModel):
       threshold: float = Field(default=0.0, description="Variance threshold.")

   # 注册到 ParamSchemaRegistry (domain="feature_selection")
   ParamSchemaRegistry.register("feature_selection", "my_variance_selector", MyVarianceParams)

**3. 在 YAML 中调用**

.. code-block:: yaml

   feature_selection_methods:
     - method: my_variance_selector
       params:
         threshold: 0.5

实战三：自定义机器学习模型 (ModelFactory)
-----------------------------------------

模型工厂 ``ModelFactory`` 继承自 ``ClassRegistry[BaseModel]``。注意其构造函数契约：它接收一个单一的 ``config`` 字典。

**1. 实现模型与注册名称**

继承 ``BaseModel``，实现 ``fit``, ``predict``, ``predict_proba``。

.. code-block:: python

   from sklearn.neural_network import MLPClassifier
   from habit.core.machine_learning.models.base import BaseModel
   from habit.core.machine_learning.models.factory import ModelFactory

   @ModelFactory.register("my_mlp")
   class MyMLPModel(BaseModel):
       def __init__(self, config: dict):
           super().__init__(config)
           # 解析 config 字典
           hidden_layer_sizes = config.get('hidden_layer_sizes', (100,))
           random_state = config.get('random_state', 42)
           
           self.model = MLPClassifier(
               hidden_layer_sizes=hidden_layer_sizes,
               random_state=random_state
           )

       def fit(self, X, y, **kwargs):
           self.model.fit(X, y)
           return self

       def predict(self, X, **kwargs):
           return self.model.predict(X)

       def predict_proba(self, X, **kwargs):
           return self.model.predict_proba(X)

**2. 定义参数 Schema 并注册**

.. code-block:: python

   from typing import Tuple
   from pydantic import BaseModel, Field
   from habit.core.schemas.registry import ParamSchemaRegistry

   class MyMLPParams(BaseModel):
       hidden_layer_sizes: Tuple[int, ...] = Field(default=(100,))
       random_state: int = Field(default=42)

   # 注册到 ParamSchemaRegistry (domain="model")
   ParamSchemaRegistry.register("model", "my_mlp", MyMLPParams)

实战四：自定义聚类阶段特征提取器 (FeatureExtractorRegistry + method_param_spec)
-------------------------------------------------------------------------------------------------

聚类阶段特征提取器通过 ``FeatureExtractorRegistry`` 惰性发现 ``*_extractor.py`` 模块。
除实现 ``BaseClusteringExtractor`` 外，请在类上声明 ``method_param_spec``，供函数式
``method`` 表达式的参数绑定校验与默认值注入使用：

.. code-block:: python

   from habit.core.habitat_analysis.clustering_features.base_extractor import (
       BaseClusteringExtractor,
       FeatureExtractorRegistry,
   )
   from habit.core.habitat_analysis.clustering_features.method_param_spec import (
       MethodParamSpec,
   )

   @FeatureExtractorRegistry.register("my_texture")
   class MyTextureExtractor(BaseClusteringExtractor):
       method_param_spec = MethodParamSpec(
           required=(),                         # names that MUST appear in F(...)
           optional={"window_size": 5},         # built-in defaults when omitted
           default_params_file_preset=None,       # or "voxel"/"supervoxel"/"roi"/"habitat"
           combiner=False,
           takes_image=True,
       )

       def extract_features(self, image_data, mask_data, **kwargs):
           ...

YAML 用法（括号声明绑定，``params`` 只赋 value）：

.. code-block:: yaml

   feature_construction:
     voxel_level:
       method: concat(my_texture(T2, window_size))
       params:
         window_size: 7

``params_file`` 可省略；radiomics 类方法使用 ``habit/resources/radiomics/`` 中的 bundled preset。

全部扩展点速查表
----------------

HABIT 共有 **8 个注册表**\ （6 个类式工厂 + 2 个函数式注册表）。下表是完整清单，新增组件时对号入座即可：

.. list-table::
   :header-rows: 1
   :widths: 22 26 28 24

   * - 组件类型
     - 注册装饰器
     - 逻辑契约 (基类/函数签名)
     - Schema 注册 Domain
   * - **预处理步骤**
     - ``@PreprocessorFactory.register("name")``
     - 继承 ``BasePreprocessor``
     - ``preprocessing``
   * - **机器学习模型**
     - ``@ModelFactory.register("name")``
     - 继承 ``BaseModel``
     - ``model``
   * - **聚类算法**
     - ``@ClusteringAlgorithmFactory.register("name")``
     - 继承 ``BaseClustering``
     - (依附生境，无独立 domain)
   * - **聚类阶段特征提取器**
     - ``FeatureExtractorRegistry`` (惰性发现 ``*_extractor.py``)
     - 继承 ``BaseClusteringExtractor``
     - (依附生境)
   * - **特征表预处理方法**
     - ``@PreprocessingMethodFactory.register("name")``
     - 继承 ``BaseFeaturePreprocessing``
     - (依附生境)
   * - **分割后生境特征插件**
     - ``@HabitatFeatureRegistry.register("name")``
     - 继承 ``HabitatFeaturePluginBase``
     - (依附生境)
   * - **特征选择器**
     - ``@SelectorRegistry.register("name")``
     - 函数 ``(SelectorContext) -> List[str]``
     - ``feature_selection``
   * - **评估指标**
     - ``@MetricRegistry.register("name")``
     - 函数 ``(y_true, y_pred, y_prob, cm=None) -> float``
     - (纯函数，通常无参数 Schema)

.. note::

   **两类注册表的差别**：前 6 个是 **类式工厂**\ （继承 ``ClassRegistry``，产物是"类"，用 ``create()`` 实例化）；
   后 2 个是 **函数式注册表**\ （继承 ``CallableRegistry``，产物是"函数"）。二者都遵守统一注册表接口，
   由架构契约测试守护，详见 :doc:`invariants`。

.. seealso::

   完整可复制的组件模板见 :doc:`../customization/index`；新增后如何登记进契约测试见 :doc:`dev_workflow`。