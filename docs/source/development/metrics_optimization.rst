Metrics Module Optimization
===========================

优化概述
--------

本次优化对 ``habit/core/machine_learning/evaluation/metrics.py`` 模块进行了全面改进，在保持向后兼容的前提下，实现了性能提升和功能增强。

主要优化
--------

1. 性能优化：混淆矩阵缓存 🚀
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**问题**：之前每个指标函数都独立计算混淆矩阵，导致重复计算8次。

**解决方案**：引入 ``MetricsCache`` 类

.. code-block:: python

    class MetricsCache:
        """Cache confusion matrix to avoid repeated calculations."""
        @property
        def confusion_matrix(self):
            if self._cm is None:
                self._cm = metrics.confusion_matrix(self.y_true, self.y_pred)
            return self._cm

**性能提升**：约 **8倍** 速度提升（从8次混淆矩阵计算降至1次）

**使用方法**：

.. code-block:: python

    # Enable cache (default)
    metrics = calculate_metrics(y_true, y_pred, y_prob, use_cache=True)

    # Disable cache
    metrics = calculate_metrics(y_true, y_pred, y_prob, use_cache=False)

2. 扩展Target Metrics支持 💡
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**新增支持的指标**：

* ✅ Sensitivity（已支持）
* ✅ Specificity（已支持）
* ✅ **PPV (Precision)** - 新增
* ✅ **NPV** - 新增
* ✅ **F1-score** - 新增
* ✅ **Accuracy** - 新增

**使用示例**：

.. code-block:: python

    targets = {
        'sensitivity': 0.91,
        'specificity': 0.91,
        'ppv': 0.85,           # New
        'npv': 0.90,           # New
        'f1_score': 0.88       # New
    }

    result = calculate_metrics_at_target(y_true, y_pred_proba, targets)

.. note::
   PPV/NPV/F1等指标需要遍历所有阈值计算，性能开销较大，但必要。

3. Fallback机制：最接近阈值 🎯
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**问题**：当没有阈值能同时满足所有目标时（如目标过高），之前会直接返回空结果。

**解决方案**：自动寻找"最接近"的阈值

.. code-block:: python

    result = calculate_metrics_at_target(
        y_true, y_pred_proba,
        {'sensitivity': 0.99, 'specificity': 0.99},  # Impossible targets
        fallback_to_closest=True,
        distance_metric='euclidean'  # 'euclidean', 'manhattan', or 'max'
    )

    # Return structure
    {
        'closest_threshold': {
            'threshold': 0.5234,
            'metrics': {...},
            'distance_to_target': 0.0523,
            'satisfied_targets': ['sensitivity'],
            'unsatisfied_targets': ['specificity'],
            'warning': 'No threshold satisfies all targets. This is the closest match.'
        }
    }

**距离度量**：

* ``euclidean``: √Σ(actual - target)²（默认）
* ``manhattan``: Σ|actual - target|
* ``max``: max(|actual - target|)

.. warning::
   **严格避免数据泄露**：
   
   * 训练集：找到最接近阈值
   * 测试集：应用训练集的阈值（不重新搜索）

4. 智能阈值选择策略 🧠
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**问题**：当多个阈值都满足条件时，如何选择最优？

**解决方案**：三种策略

策略1：First（快速）
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    result = calculate_metrics_at_target(..., threshold_selection='first')
    # Return first threshold that meets criteria

策略2：Youden（经典）
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    result = calculate_metrics_at_target(..., threshold_selection='youden')
    # Return threshold with maximum Youden index
    # Youden = Sensitivity + Specificity - 1

策略3：Pareto+Youden（推荐）⭐
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    result = calculate_metrics_at_target(..., threshold_selection='pareto+youden')
    # 1. Find all Pareto-optimal thresholds (not dominated by others)
    # 2. Select the one with maximum Youden index among Pareto-optimal

**返回结构**：

.. code-block:: python

    {
        'best_threshold': {
            'threshold': 0.5222,
            'metrics': {...},
            'strategy': 'pareto+youden',
            'youden_index': 0.8792,
            'pareto_optimal_count': 3  # Number of Pareto-optimal thresholds
        }
    }

5. 类别筛选功能 📋
~~~~~~~~~~~~~~~~~~~~~

**功能**：按类别筛选要计算的指标

.. code-block:: python

    # Calculate only basic metrics (fast)
    basic_metrics = calculate_metrics(
        y_true, y_pred, y_prob,
        categories=['basic']
    )
    # Returns: accuracy, sensitivity, specificity, ppv, npv, f1_score, auc

    # Calculate only statistical metrics
    stat_metrics = calculate_metrics(
        y_true, y_pred, y_prob,
        categories=['statistical']
    )
    # Returns: hosmer_lemeshow_p_value, spiegelhalter_z_p_value

    # Calculate multiple categories
    metrics = calculate_metrics(
        y_true, y_pred, y_prob,
        categories=['basic', 'statistical']
    )

    # Calculate all metrics (default)
    all_metrics = calculate_metrics(y_true, y_pred, y_prob)

6. 多分类支持准备 🔮
~~~~~~~~~~~~~~~~~~~~~~~

**当前状态**：

* ✅ AUC：已支持多分类（One-vs-Rest）
* ✅ 基础指标：新增多分类支持（macro average）

**多分类示例**：

.. code-block:: python

    # Binary classification (current behavior)
    cm = [[TN, FP],
          [FN, TP]]
    sensitivity = TP / (TP + FN)

    # Multi-class classification (new support)
    cm = [[c00, c01, c02],
          [c10, c11, c12],
          [c20, c21, c22]]
    # Per-class recall, then macro average

向后兼容性 ✅
--------------

所有现有代码 **无需修改**，默认行为保持不变：

.. code-block:: python

    # Old code still works
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    result = calculate_metrics_at_target(y_true, y_prob, {'sensitivity': 0.9})

    # New features enabled via optional parameters
    metrics = calculate_metrics(y_true, y_pred, y_prob, use_cache=True, categories=['basic'])
    result = calculate_metrics_at_target(
        y_true, y_prob, targets,
        threshold_selection='pareto+youden',
        fallback_to_closest=True
    )

使用建议
--------

推荐配置（最佳实践）
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # 1. Training set: find optimal threshold
    train_result = calculate_metrics_at_target(
        y_train_true,
        y_train_prob,
        targets={'sensitivity': 0.91, 'specificity': 0.91, 'ppv': 0.85},
        threshold_selection='pareto+youden',  # Intelligent selection
        fallback_to_closest=True,             # Enable fallback
        distance_metric='euclidean'
    )

    # 2. Extract threshold
    if train_result['best_threshold']:
        threshold = train_result['best_threshold']['threshold']
        logger.info(f"Best threshold found: {threshold}")
    elif train_result['closest_threshold']:
        threshold = train_result['closest_threshold']['threshold']
        logger.warning(f"Using closest threshold: {threshold}")
    else:
        threshold = 0.5  # Default
        logger.error("No suitable threshold found, using default")

    # 3. Test set: apply training threshold
    test_metrics = apply_threshold(y_test_true, y_test_prob, threshold)

性能对比
--------

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 15

   * - 场景
     - 优化前
     - 优化后
     - 提升
   * - 计算8个基础指标
     - 8次CM计算
     - 1次CM计算
     - **8x**
   * - Target metrics (sens+spec)
     - 较快
     - 较快
     - ~1x
   * - Target metrics (+ppv+npv+f1)
     - N/A
     - 中等
     - 新功能
   * - Pareto最优选择
     - N/A
     - 中等
     - 新功能

**推荐**：

* 日常使用：启用缓存
* 简单场景：只用sensitivity+specificity
* 复杂场景：可增加ppv/npv/f1（性能换精度）

测试验证
--------

运行测试套件：

.. code-block:: bash

    pytest tests/test_metrics_optimization.py -v

测试覆盖：

* ✅ 混淆矩阵缓存性能
* ✅ PPV/NPV/F1目标支持
* ✅ Fallback机制
* ✅ Pareto+Youden选择
* ✅ 类别筛选
* ✅ 不同距离度量

已知限制
--------

1. **PPV/NPV计算慢**：需要遍历所有阈值，O(n)复杂度
   
   * 建议：优先用sensitivity+specificity，必要时才加ppv/npv

2. **Pareto算法复杂度**：O(n²) worst case
   
   * 实际影响小（阈值数量通常<1000）

3. **多分类完全支持**：需要更多测试和验证
   
   * 当前：基础支持（macro average）
   * 未来：weighted, per-class等策略

未来改进方向
------------

1. **GPU加速**：混淆矩阵计算（大规模数据）
2. **并行化**：Pareto最优搜索（多线程）
3. **自适应策略**：根据数据自动选择threshold_selection
4. **可视化**：Pareto前沿曲线绘制
5. **多分类全面支持**：weighted, per-class策略

技术债务清理
------------

已解决的技术债：

* ✅ 重复计算混淆矩阵（8x性能损失）
* ✅ 硬编码只支持sens/spec
* ✅ 无fallback机制
* ✅ category参数未使用
* ✅ F1-score低效计算（3次CM）

示例代码
--------

完整工作流示例
~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from habit.core.machine_learning.evaluation.metrics import (
        calculate_metrics_at_target,
        calculate_metrics
    )

    # Simulated data
    np.random.seed(42)
    y_train_true = np.random.randint(0, 2, 300)
    y_train_prob = np.random.rand(300)
    y_test_true = np.random.randint(0, 2, 100)
    y_test_prob = np.random.rand(100)

    # Step 1: Find optimal threshold in training set
    train_result = calculate_metrics_at_target(
        y_train_true,
        y_train_prob,
        targets={
            'sensitivity': 0.85,
            'specificity': 0.85,
            'ppv': 0.80
        },
        threshold_selection='pareto+youden',
        fallback_to_closest=True
    )

    # Step 2: Extract and log threshold
    if 'best_threshold' in train_result and train_result['best_threshold']:
        threshold = train_result['best_threshold']['threshold']
        print(f"Best threshold: {threshold}")
        print(f"Strategy: {train_result['best_threshold']['strategy']}")
        print(f"Youden: {train_result['best_threshold']['youden_index']}")
    elif 'closest_threshold' in train_result and train_result['closest_threshold']:
        threshold = train_result['closest_threshold']['threshold']
        print(f"Closest threshold: {threshold}")
        print(f"Distance: {train_result['closest_threshold']['distance_to_target']}")
    else:
        threshold = 0.5
        print("Using default threshold: 0.5")

    # Step 3: Apply to test set
    y_test_pred = (y_test_prob >= threshold).astype(int)
    test_metrics = calculate_metrics(
        y_test_true, 
        y_test_pred, 
        y_test_prob,
        use_cache=True,
        categories=['basic']
    )

    print(f"Test metrics: {test_metrics}")

类别筛选示例
~~~~~~~~~~~~

.. code-block:: python

    from habit.core.machine_learning.evaluation.metrics import calculate_metrics

    # Only basic metrics (fast)
    basic = calculate_metrics(
        y_true, y_pred, y_prob,
        categories=['basic']
    )
    print("Basic metrics:", basic.keys())
    # Output: accuracy, sensitivity, specificity, ppv, npv, f1_score, auc

    # Only statistical metrics
    stats = calculate_metrics(
        y_true, y_pred, y_prob,
        categories=['statistical']
    )
    print("Statistical metrics:", stats.keys())
    # Output: hosmer_lemeshow_p_value, spiegelhalter_z_p_value

    # All metrics
    all_metrics = calculate_metrics(y_true, y_pred, y_prob)
    print("All metrics:", all_metrics.keys())

API参考
-------

calculate_metrics_at_target
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def calculate_metrics_at_target(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        targets: Dict[str, float],
        threshold_selection: str = 'first',
        fallback_to_closest: bool = False,
        distance_metric: str = 'euclidean'
    ) -> Dict[str, Any]

**参数**：

* ``y_true`` (*np.ndarray*): True labels
* ``y_pred_proba`` (*np.ndarray*): Predicted probabilities
* ``targets`` (*Dict[str, float]*): Target metric values
* ``threshold_selection`` (*str*): 'first', 'youden', or 'pareto+youden' (default: 'first')
* ``fallback_to_closest`` (*bool*): Enable fallback mechanism (default: False)
* ``distance_metric`` (*str*): 'euclidean', 'manhattan', or 'max' (default: 'euclidean')

**返回**：

Dictionary with keys: ``best_threshold``, ``closest_threshold``, ``combined_results``, ``all_thresholds``

calculate_metrics
~~~~~~~~~~~~~~~~~

.. code-block:: python

    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        use_cache: bool = True,
        categories: Optional[List[str]] = None
    ) -> Dict[str, float]

**参数**：

* ``y_true`` (*np.ndarray*): True labels
* ``y_pred`` (*np.ndarray*): Predicted labels
* ``y_prob`` (*np.ndarray*): Predicted probabilities
* ``use_cache`` (*bool*): Enable confusion matrix caching (default: True)
* ``categories`` (*Optional[List[str]]*): List of metric categories to compute (default: None = all)

**返回**：

Dictionary of metric name to value

联系与反馈
----------

如有问题或建议，请提交Issue或Pull Request。

| **作者**：HABIT开发团队
| **日期**：2026-01-25
| **版本**：v2.0
