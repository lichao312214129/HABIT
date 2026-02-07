验证方法
========

聚类验证指标
------------

Silhouette Score
^^^^^^^^^^^^^^^^

.. autofunction:: habit.core.habitat_analysis.algorithms.cluster_validation_methods.silhouette_score

Calinski-Harabasz Score
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: habit.core.habitat_analysis.algorithms.cluster_validation_methods.calinski_harabasz_score

Davies-Bouldin Score
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: habit.core.habitat_analysis.algorithms.cluster_validation_methods.davies_bouldin_score

Kneedle
^^^^^^^

Kneedle 用于在单调曲线中自动识别“拐点”。在 HABIT 中，Kneedle 使用
``kneed`` 包实现，主要应用于 KMeans 的 Inertia 曲线，以检测最明显的拐点。

验证方法选择
------------

不同的聚类算法支持不同的验证方法。请参考各算法的文档了解支持的验证方法。
