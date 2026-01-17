# Habitat Analysis 预处理状态管理优化

## 目标

解决无监督聚类中训练/测试数据一致性问题，确保测试数据使用训练集的统计参数进行预处理，防止数据泄露。

## 核心改进

### 1. 创建 `PreprocessingState` 类

**位置：** `habit/core/habitat_analysis/utils/preprocessing_state.py`

**功能：**
- **训练阶段**：计算并保存群体级统计参数（均值、标准差、最大最小值等）
- **测试阶段**：加载保存的参数，应用于测试数据
- **支持的预处理方法**：
  - `mean_imputation`：使用均值填充缺失值
  - `z_score` / `standardization`：Z-score 标准化
  - `min_max` / `normalize`：Min-Max 归一化

**关键方法：**
```python
# Training: Fit and Transform
state.fit(df, methods)           # Calculate parameters from training data
transformed = state.transform(df)  # Apply parameters

# Testing: Transform only
state = PreprocessingState.load(output_dir)  # Load saved parameters
transformed = state.transform(test_df)        # Apply to test data
```

### 2. 统一存储：模型 + 预处理状态

**改进前：**
- 聚类模型：`supervoxel2habitat_clustering_strategy.pkl`
- 均值参数：`mean_values_of_all_supervoxels_features.csv`
- 分散管理，容易遗漏

**改进后：**
- **单一文件**：`supervoxel2habitat_clustering_strategy_bundle.pkl`
- **包含内容**：
  ```python
  {
      'clustering_model': <KMeans/GMM/etc>,
      'preprocessing_state': <PreprocessingState>,
      'model_name': 'supervoxel2habitat_clustering_strategy'
  }
  ```

### 3. Mode 类增强

#### `BaseMode` (抽象基类)
新增抽象方法：
```python
@abstractmethod
def process_features(self, features: pd.DataFrame, methods: List[Dict]) -> pd.DataFrame:
    """Process features using group-level statistics (stateful)"""
    pass
```

#### `TrainingMode`
- **`process_features`**：调用 `state.fit()` + `state.transform()`
- **`save_model`**：保存 bundle（模型 + 状态）

#### `TestingMode`
- **`process_features`**：调用 `state.transform()`（仅应用，不重新计算）
- **`load_model`**：加载 bundle，自动恢复预处理状态

### 4. FeatureManager 适配

**修改位置：** `habit/core/habitat_analysis/managers/feature_manager.py`

**核心改动：**
```python
def apply_preprocessing(
    self, 
    feature_df: pd.DataFrame, 
    level: str,
    mode_handler: Any = None  # ← 新增参数
) -> pd.DataFrame:
    """
    Apply preprocessing based on level.
    
    For 'group' level: delegates to mode_handler.process_features()
    For 'subject' level: applies standard preprocessing
    """
    if level == 'group':
        return mode_handler.process_features(feature_df, methods)
    elif level == 'subject':
        return self._apply_preprocessing(feature_df, 'preprocessing_for_subject_level')
```

### 5. Strategy 类更新

**修改文件：**
- `strategies/two_step_strategy.py`
- `strategies/direct_pooling_strategy.py`

**关键改动：**
```python
# Before
features = self.analysis.feature_manager.apply_preprocessing(features, level='group')
self.analysis.feature_manager.handle_mean_values(features, mode_handler)

# After
features = self.analysis.feature_manager.apply_preprocessing(
    features, level='group', mode_handler=mode_handler
)
```

## 数据流对比

### Training Mode
```
Raw Features
    ↓
Clean (remove inf/nan types)
    ↓
PreprocessingState.fit(features, methods)    ← Compute mean, std, min, max
    ↓
PreprocessingState.transform(features)       ← Apply computed stats
    ↓
Clustering
    ↓
Save Bundle: {model, preprocessing_state}    ← Single file
```

### Testing Mode
```
Raw Features
    ↓
Clean (remove inf/nan types)
    ↓
Load Bundle: {model, preprocessing_state}
    ↓
PreprocessingState.transform(features)       ← Apply TRAINING stats (no fit!)
    ↓
Model.predict()
```

## 防止数据泄露的关键点

1. **训练阶段**：所有统计量（均值、方差等）仅从训练数据计算
2. **测试阶段**：
   - ✅ 使用训练集的统计量
   - ❌ 不计算测试集自己的统计量
   - ❌ 不混合训练/测试数据
3. **持久化**：统计量与模型一起保存，确保一致性

## 配置示例

```yaml
feature_config:
  preprocessing_for_group_level:
    methods:
      - method: z_score          # Z-score normalization
      - method: min_max          # Min-Max scaling
        # mean_imputation is always applied first by default
```

## 文件说明

### 新增文件
- `habit/core/habitat_analysis/utils/preprocessing_state.py` - 预处理状态管理核心类

### 修改文件
- `modes/base_mode.py` - 添加 `process_features` 抽象方法
- `modes/training_mode.py` - 实现状态计算与保存
- `modes/testing_mode.py` - 实现状态加载与应用
- `managers/feature_manager.py` - 增强 `apply_preprocessing` 方法
- `strategies/two_step_strategy.py` - 更新预处理调用
- `strategies/direct_pooling_strategy.py` - 更新预处理调用

## 测试结果

运行 `tests/test_habitat.py` 成功输出：
```
2026-01-17 11:03:13 - INFO - Computing and applying group-level preprocessing...
2026-01-17 11:03:22 - INFO - Training bundle (model + preprocessing state) saved to: 
    F:\work\habit_project\demo_data\results\habitat\supervoxel2habitat_clustering_strategy_bundle.pkl
2026-01-17 11:03:32 - INFO - Habitat analysis completed successfully
```

## 优势总结

1. ✅ **数据安全**：严格防止测试数据泄露到训练参数中
2. ✅ **统一管理**：单文件包含所有训练产物，易于部署和版本控制
3. ✅ **可扩展性**：轻松添加新的预处理方法（如 RobustScaler, PCA 等）
4. ✅ **可维护性**：清晰的职责分离（State 管理参数，Mode 管理流程，Manager 管理操作）
5. ✅ **向后兼容**：旧的 `preprocessing_for_group_level` 配置仍然有效

## 后续可能的扩展

1. 支持更多预处理方法（如 `robust_scaling`, `quantile_transform`）
2. 添加 `PreprocessingState` 的可视化（显示保存的统计量）
3. 支持增量更新（Online Learning）场景
4. 添加状态验证（检测训练/测试数据分布差异）
