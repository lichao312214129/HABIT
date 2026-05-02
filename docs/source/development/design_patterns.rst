设计模式
========

HABIT 使用了多种设计模式来提高代码的可维护性和可扩展性。

策略模式 (Strategy Pattern)
^^^^^^^^^^^^^^^^^^^^^^^^^^

用于实现不同的聚类策略。

**问题**：不同的研究场景需要不同的聚类策略（一步法、二步法、直接拼接法）。

**解决方案**：定义策略接口，每个策略实现自己的逻辑。

.. code-block:: python

   from abc import ABC, abstractmethod
   
   class BaseClusteringStrategy(ABC):
       @abstractmethod
       def run(self, analysis, subjects=None):
           pass
   
   class TwoStepStrategy(BaseClusteringStrategy):
       def run(self, analysis, subjects=None):
           # 实现二步法逻辑
           pass
   
   class OneStepStrategy(BaseClusteringStrategy):
       def run(self, analysis, subjects=None):
           # 实现一步法逻辑
           pass

工厂模式 (Factory Pattern)
^^^^^^^^^^^^^^^^^^^^^^

用于创建不同的特征提取器和聚类算法。

**问题**：需要根据配置动态创建不同的对象。

**解决方案**：使用工厂函数根据名称创建对象。

.. code-block:: python

   def create_feature_extractor(method, params):
       if method == 'mean_voxel_features':
           return MeanVoxelFeaturesExtractor(params)
       elif method == 'kinetic_features':
           return KineticFeatureExtractor(params)
       # ... 更多提取器
   
   def create_clustering_algorithm(algorithm, n_clusters):
       if algorithm == 'kmeans':
           return KMeansClustering(n_clusters)
       elif algorithm == 'gmm':
           return GMMClustering(n_clusters)
       # ... 更多算法

依赖注入 (Dependency Injection)
^^^^^^^^^^^^^^^^^^^^^^^^^^

通过域专用 configurator (``HabitatConfigurator`` /
``MLConfigurator`` / ``PreprocessingConfigurator``) 管理依赖关系，
提高可测试性。

**问题**：类之间有复杂的依赖关系，难以测试。

**解决方案**：通过构造函数注入依赖，而不是在内部创建。

.. code-block:: python

   # 之前：在内部创建依赖
   class HabitatAnalysis:
       def __init__(self, config):
           self.feature_service = FeatureService(config)
           self.clustering_service = ClusteringService(config)

   # 之后：通过构造函数注入
   class HabitatAnalysis:
       def __init__(self, config, feature_service, clustering_service):
           self.feature_service = feature_service
           self.clustering_service = clustering_service

   # 使用 HabitatConfigurator 装配
   from habit.core.common.configurators import HabitatConfigurator
   configurator = HabitatConfigurator(config=config)
   analysis = configurator.create_habitat_analysis()

观察者模式 (Observer Pattern)
^^^^^^^^^^^^^^^^^^^^^^^^^^

用于日志和进度通知。

**问题**：需要在不同地方记录日志和显示进度。

**解决方案**：使用 logger 和回调函数。

.. code-block:: python

   class FeatureService:
       def __init__(self, config, logger):
           self.logger = logger
       
       def extract_features(self, subject):
           self.logger.info(f"Extracting features for {subject}")
           # ... 提取逻辑
           self.logger.info(f"Completed {subject}")

模板方法模式 (Template Method Pattern)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

用于定义算法骨架，子类实现具体步骤。

**问题**：不同算法有相似的结构，但某些步骤不同。

**解决方案**：在基类中定义骨架，子类重写特定步骤。

.. code-block:: python

   class BaseClustering(ABC):
       def fit(self, X):
           self._validate_input(X)
           self._initialize(X)
           self._iterate(X)
           self._finalize()
       
       @abstractmethod
       def _initialize(self, X):
           pass
       
       @abstractmethod
       def _iterate(self, X):
           pass

适配器模式 (Adapter Pattern)
^^^^^^^^^^^^^^^^^^^^^^

用于适配不同的数据格式和接口。

**问题**：需要支持多种输入格式（DICOM、NIfTI 等）。

**解决方案**：创建适配器统一接口。

.. code-block:: python

   class ImageAdapter:
       @staticmethod
       def load(filepath):
           if filepath.endswith('.dcm'):
               return DICOMLoader.load(filepath)
           elif filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
               return NIfTILoader.load(filepath)
           else:
               raise ValueError(f"Unsupported format: {filepath}")
