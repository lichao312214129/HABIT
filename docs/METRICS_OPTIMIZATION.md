# Metrics Module Optimization Summary

## 优化概述

本次优化对 `habit/core/machine_learning/evaluation/metrics.py` 模块进行了全面改进，在保持向后兼容的前提下，实现了性能提升和功能增强。

## 主要优化

### 1. 性能优化：混淆矩阵缓存 🚀

**问题**：之前每个指标函数都独立计算混淆矩阵，导致重复计算8次。

**解决方案**：引入 `MetricsCache` 类

```python
class MetricsCache:
    """Cache confusion matrix to avoid repeated calculations."""
    @property
    def confusion_matrix(self):
        if self._cm is None:
            self._cm = metrics.confusion_matrix(self.y_true, self.y_pred)
        return self._cm
```

**性能提升**：约 **8倍** 速度提升（从8次混淆矩阵计算降至1次）

**使用方法**：
```python
# 启用缓存（默认）
metrics = calculate_metrics(y_true, y_pred, y_prob, use_cache=True)

# 禁用缓存
metrics = calculate_metrics(y_true, y_pred, y_prob, use_cache=False)
```

---

### 2. 扩展Target Metrics支持 💡

**新增支持的指标**：
- ✅ Sensitivity（已支持）
- ✅ Specificity（已支持）
- ✅ **PPV (Precision)** - 新增
- ✅ **NPV** - 新增
- ✅ **F1-score** - 新增
- ✅ **Accuracy** - 新增

**使用示例**：
```python
targets = {
    'sensitivity': 0.91,
    'specificity': 0.91,
    'ppv': 0.85,           # 新增
    'npv': 0.90,           # 新增
    'f1_score': 0.88       # 新增
}

result = calculate_metrics_at_target(y_true, y_pred_proba, targets)
```

**注意**：PPV/NPV/F1等指标需要遍历所有阈值计算，性能开销较大，但必要。

---

### 3. Fallback机制：最接近阈值 🎯

**问题**：当没有阈值能同时满足所有目标时（如目标过高），之前会直接返回空结果。

**解决方案**：自动寻找"最接近"的阈值

```python
result = calculate_metrics_at_target(
    y_true, y_pred_proba,
    {'sensitivity': 0.99, 'specificity': 0.99},  # 不可能同时满足
    fallback_to_closest=True,
    distance_metric='euclidean'  # 'euclidean', 'manhattan', or 'max'
)

# 返回结构
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
```

**距离度量**：
- `euclidean`: √Σ(actual - target)²（默认）
- `manhattan`: Σ|actual - target|
- `max`: max(|actual - target|)

**严格避免数据泄露**：
- 训练集：找到最接近阈值
- 测试集：应用训练集的阈值（不重新搜索）

---

### 4. 智能阈值选择策略 🧠

**问题**：当多个阈值都满足条件时，如何选择最优？

**解决方案**：三种策略

#### 策略1：First（快速）
```python
result = calculate_metrics_at_target(..., threshold_selection='first')
# 返回第一个满足条件的阈值
```

#### 策略2：Youden（经典）
```python
result = calculate_metrics_at_target(..., threshold_selection='youden')
# 返回Youden指数最大的阈值
# Youden = Sensitivity + Specificity - 1
```

#### 策略3：Pareto+Youden（推荐）⭐
```python
result = calculate_metrics_at_target(..., threshold_selection='pareto+youden')
# 1. 找出所有Pareto最优阈值（无法被其他阈值"完全支配"）
# 2. 在Pareto最优中选择Youden最大的
```

**返回结构**：
```python
{
    'best_threshold': {
        'threshold': 0.5222,
        'metrics': {...},
        'strategy': 'pareto+youden',
        'youden_index': 0.8792,
        'pareto_optimal_count': 3  # Pareto最优阈值数量
    }
}
```

---

### 5. 类别筛选功能 📋

**功能**：按类别筛选要计算的指标

```python
# 只计算基础指标（快速）
basic_metrics = calculate_metrics(
    y_true, y_pred, y_prob,
    categories=['basic']
)
# 返回：accuracy, sensitivity, specificity, ppv, npv, f1_score, auc

# 只计算统计指标
stat_metrics = calculate_metrics(
    y_true, y_pred, y_prob,
    categories=['statistical']
)
# 返回：hosmer_lemeshow_p_value, spiegelhalter_z_p_value

# 计算多个类别
metrics = calculate_metrics(
    y_true, y_pred, y_prob,
    categories=['basic', 'statistical']
)

# 计算所有指标（默认）
all_metrics = calculate_metrics(y_true, y_pred, y_prob)
```

---

### 6. 多分类支持准备 🔮

**当前状态**：
- ✅ AUC：已支持多分类（One-vs-Rest）
- ✅ 基础指标：新增多分类支持（macro average）

**多分类示例**：
```python
# 二分类（现有行为）
cm = [[TN, FP],
      [FN, TP]]
sensitivity = TP / (TP + FN)

# 多分类（新增）
cm = [[c00, c01, c02],
      [c10, c11, c12],
      [c20, c21, c22]]
# Per-class recall, then macro average
```

---

## 向后兼容性 ✅

所有现有代码**无需修改**，默认行为保持不变：

```python
# 旧代码仍然有效
metrics = calculate_metrics(y_true, y_pred, y_prob)
result = calculate_metrics_at_target(y_true, y_prob, {'sensitivity': 0.9})

# 新功能通过可选参数启用
metrics = calculate_metrics(y_true, y_pred, y_prob, use_cache=True, categories=['basic'])
result = calculate_metrics_at_target(
    y_true, y_prob, targets,
    threshold_selection='pareto+youden',
    fallback_to_closest=True
)
```

---

## 使用建议

### 推荐配置（最佳实践）

```python
# 1. 训练集：找最优阈值
train_result = calculate_metrics_at_target(
    y_train_true,
    y_train_prob,
    targets={'sensitivity': 0.91, 'specificity': 0.91, 'ppv': 0.85},
    threshold_selection='pareto+youden',  # 智能选择
    fallback_to_closest=True,            # 启用fallback
    distance_metric='euclidean'
)

# 2. 提取阈值
if train_result['best_threshold']:
    threshold = train_result['best_threshold']['threshold']
    logger.info(f"找到最优阈值: {threshold}")
elif train_result['closest_threshold']:
    threshold = train_result['closest_threshold']['threshold']
    logger.warning(f"使用最接近阈值: {threshold}")
else:
    threshold = 0.5  # 默认
    logger.error("未找到合适阈值，使用默认值")

# 3. 测试集：应用训练集阈值
test_metrics = apply_threshold(y_test_true, y_test_prob, threshold)
```

---

## 性能对比

| 场景 | 优化前 | 优化后 | 提升 |
|-----|--------|--------|------|
| 计算8个基础指标 | 8次CM计算 | 1次CM计算 | **8x** |
| Target metrics (sens+spec) | 较快 | 较快 | ~1x |
| Target metrics (+ppv+npv+f1) | N/A | 中等 | 新功能 |
| Pareto最优选择 | N/A | 中等 | 新功能 |

**推荐**：
- 日常使用：启用缓存
- 简单场景：只用sensitivity+specificity
- 复杂场景：可增加ppv/npv/f1（性能换精度）

---

## 测试验证

运行测试套件：
```bash
pytest tests/test_metrics_optimization.py -v
```

测试覆盖：
- ✅ 混淆矩阵缓存性能
- ✅ PPV/NPV/F1目标支持
- ✅ Fallback机制
- ✅ Pareto+Youden选择
- ✅ 类别筛选
- ✅ 不同距离度量

---

## 已知限制

1. **PPV/NPV计算慢**：需要遍历所有阈值，O(n)复杂度
   - 建议：优先用sensitivity+specificity，必要时才加ppv/npv

2. **Pareto算法复杂度**：O(n²) worst case
   - 实际影响小（阈值数量通常<1000）

3. **多分类完全支持**：需要更多测试和验证
   - 当前：基础支持（macro average）
   - 未来：weighted, per-class等策略

---

## 未来改进方向

1. **GPU加速**：混淆矩阵计算（大规模数据）
2. **并行化**：Pareto最优搜索（多线程）
3. **自适应策略**：根据数据自动选择threshold_selection
4. **可视化**：Pareto前沿曲线绘制
5. **多分类全面支持**：weighted, per-class策略

---

## 技术债务清理

已解决的技术债：
- ✅ 重复计算混淆矩阵（8x性能损失）
- ✅ 硬编码只支持sens/spec
- ✅ 无fallback机制
- ✅ category参数未使用
- ✅ F1-score低效计算（3次CM）

---

## 联系与反馈

如有问题或建议，请提交Issue或Pull Request。

**作者**：HABIT开发团队  
**日期**：2026-01-25  
**版本**：v2.0
