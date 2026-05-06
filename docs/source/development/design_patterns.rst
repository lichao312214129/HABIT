设计模式
========

HABIT 使用了多种设计模式来提高代码的可维护性和可扩展性。

聚类模式派发 (clustering_mode recipes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

V1 生境流水线的 **一步法 / 两步法 / 直接池化** 不再由独立的 ``*Strategy`` 类实现；
``config.HabitatsSegmention.clustering_mode`` 在
:class:`~habit.core.habitat_analysis.habitat_analysis.HabitatAnalysis` 内映射到
一套 step 构造函数（recipe），由 ``_build_pipeline()`` 统一构建 ``HabitatPipeline``。

**问题**：不同研究需要不同的生境发现流程，但又要避免多层 strategy / pipeline_builder
重复与隐式反射。

**解决方案**：单一深模块持有 recipe 注册表；新增模式时登记 recipe 与 step 列表即可。
详细 step 顺序见 :doc:`module_architecture`。

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

   # Collaborators are constructed inside HabitatConfigurator and injected.
   class HabitatAnalysis:
       def __init__(self, config, feature_service, clustering_service, result_writer, logger):
           self.config = config
           self.feature_service = feature_service
           self.clustering_service = clustering_service
           self.result_writer = result_writer
           self.logger = logger

   from habit.core.habitat_analysis.configurator import HabitatConfigurator
   configurator = HabitatConfigurator(config=config, logger=logger, output_dir=str(out_dir))
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
