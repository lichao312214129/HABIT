扩展点总览
==========

HABIT 的可扩展组件都通过 **工厂（Factory）** 或 **注册表（Registry）** 管理：定义一个继承基类的类，加上注册装饰器，即可在 YAML 里按名字使用（"注册即可用"）。

本页是一张 **索引表**：列出所有扩展点、基类与注册方式。**每类组件的完整代码模板与示例见** :doc:`../customization/index`，本页不重复代码。

扩展机制的统一模式
------------------

.. mermaid::

   flowchart LR
     DEF["1. Subclass base class"] --> DEC["2. Add register decorator"]
     DEC --> IMP["3. Ensure module imported"]
     IMP --> YAML["4. Reference by name in YAML"]
     YAML --> RUN["5. Factory instantiates at runtime"]

.. important::

   第 3 步 "确保模块被导入" 很关键：注册装饰器只有在其所在模块被 import 时才会执行。
   内置组件在包 ``__init__`` 中集中导入；自定义组件需保证你的模块在运行前被加载
   （放进对应子包、或在配置/入口处显式 import）。

扩展点清单
----------

.. list-table::
   :header-rows: 1
   :widths: 22 26 30 22

   * - 组件
     - 注册方式
     - 基类 / 注册表定义位置
     - 用于命令
   * - **预处理步骤**
     - ``@PreprocessorFactory.register("name")``
     - ``preprocessing/preprocessor_factory.py`` + ``base_preprocessor.py``
     - ``preprocess``
   * - **聚类算法**
     - ``@register_clustering("name")``
     - ``habitat_analysis/clustering/base_clustering.py``
     - ``get-habitat``
   * - **聚类特征提取器**
     - ``@register_feature_extractor("name")``
     - ``habitat_analysis/clustering_features/base_extractor.py``
     - ``get-habitat``
   * - **组级特征预处理**
     - ``@register_preprocessing("name")``
     - ``habitat_analysis/feature_preprocessing/base_preprocessing.py``
     - ``get-habitat``
   * - **生境后特征插件**
     - ``@register_habitat_feature("name")``
     - ``habitat_analysis/feature_registry.py``
     - ``extract``
   * - **机器学习模型**
     - ``@ModelFactory.register("Name")``
     - ``machine_learning/models/factory.py`` + ``models/base.py``
     - ``model`` / ``cv``
   * - **特征选择方法**
     - ``@register_selector("name")``
     - ``machine_learning/feature_selectors/selector_registry.py``
     - ``model`` / ``cv``
   * - **评估指标**
     - ``@register_metric("name", "Display")``
     - ``machine_learning/evaluation/metrics.py``
     - ``model`` / ``cv`` / ``compare``
   * - **步骤参数 Schema**
     - ``ParamSchemaRegistry.register(domain, step, Model)``
     - ``schemas/registry.py`` + ``schemas/steps/``
     - 全部（校验 + GUI）

各扩展点要点
------------

预处理步骤
~~~~~~~~~~

继承 ``BasePreprocessor``，实现 ``__call__(self, data)``。内置示例：``resample`` / ``registration`` /
``n4_correction`` / ``zscore_normalization`` 等，均在 ``habit/core/preprocessing/`` 下。配准另有后端机制
（``registration/`` 下的 ants / elastix / sitk backend）。

聚类算法
~~~~~~~~

继承 ``BaseClustering``，实现 ``fit_predict(self, X)``。内置：``kmeans`` / ``gmm`` / ``slic`` 等
（``habitat_analysis/clustering/`` 下）。同目录 ``cluster_validation_methods.py`` 提供最优簇数搜索
（silhouette、BIC/AIC、elbow 等）。

聚类特征提取器
~~~~~~~~~~~~~~

继承 ``BaseClusteringExtractor``，实现 ``extract_features(...)``。内置：``raw`` / ``kinetic`` /
``local_entropy`` / ``supervoxel_radiomics`` 等。在 YAML 里通过 ``method: extractor(raw(seq1), raw(seq2))``
这类表达式组合调用。

组级特征预处理
~~~~~~~~~~~~~~

继承 ``BaseFeaturePreprocessing``。作用于跨受试者合并后的特征表（DataFrame）。内置：``minmax`` /
``zscore`` / ``robust`` / ``winsorize`` / ``variance_filter`` / ``correlation_filter`` 等
（``feature_preprocessing/builtin_methods.py``）。

生境后特征插件
~~~~~~~~~~~~~~

在生境图生成后计算特征，通过 ``@register_habitat_feature`` 注册。内置特征类型包括
``traditional`` / ``non_radiomics`` / ``whole_habitat`` / ``each_habitat`` / ``msi`` / ``ith_score``。

机器学习模型
~~~~~~~~~~~~

继承 ``BaseModel``，实现 ``fit`` / ``predict`` / ``predict_proba``，用 ``@ModelFactory.register`` 注册。
内置注册集中在 ``models/utils.py`` 的 ``register_all_models()``。

特征选择方法
~~~~~~~~~~~~

遵循 sklearn 的 ``BaseEstimator`` + ``TransformerMixin``\ （``fit`` / ``transform``），用
``@register_selector`` 注册。内置：``vif`` / ``icc`` / 相关性 / 方差 / 逐步回归 等
（``feature_selectors/`` 下）。ICC / test-retest 相关逻辑在 ``feature_selectors/icc/``。

评估指标
~~~~~~~~

用 ``@register_metric(name, display_name, category=...)`` 注册一个打分函数。内置：``accuracy`` /
``sensitivity`` / ``specificity`` / ``auc`` / ``f1_score`` 及统计类指标（H-L、Spiegelhalter 等）。

步骤参数 Schema
~~~~~~~~~~~~~~~

为新组件的 ``params`` 定义一个 Pydantic 模型（放 ``schemas/steps/``）并注册到 ``ParamSchemaRegistry``。
这让新参数获得 **类型校验** 与 **GUI 表单自动渲染**。机制详见 :doc:`configuration_system`。

.. seealso::

   - 手把手代码模板（含完整示例）：:doc:`../customization/index`。
   - 注册与校验如何联动 GUI：:doc:`configuration_system` 与 :doc:`dev_workflow`。
