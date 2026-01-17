# PreprocessingState 与 feature_preprocessing 融合方案

## 📊 融合分析

### 原有模块对比

| 模块 | 位置 | 特点 | 问题 |
|------|------|------|------|
| `feature_preprocessing.py` | `extractors/` -> `utils/` | 丰富的预处理方法（无状态）| ❌ 与 PreprocessingState 功能重叠<br>❌ 导致文件碎片化 |
| `preprocessing_state.py` | `utils/` | 状态持久化，支持训练/测试分离 | ❌ 依赖外部工具函数 |

## ✅ 最终融合方案（Phase 2）

### 架构设计：完全合并

为了简化结构并减少文件碎片，我们将 `feature_preprocessing.py` 的所有功能（工具函数）直接整合进 `preprocessing_state.py`。

### 模块职责

#### `preprocessing_state.py` (统一管理模块)
**位置**: `habit/core/habitat_analysis/utils/preprocessing_state.py`

**包含内容**:
1. **Utility Functions (Stateless)**:
   - `handle_extreme_values()`: 处理极值 (inf/nan)
   - `create_discretizer()`: 创建离散化器
   - `preprocess_features()`: 无状态预处理入口（供 `FeatureManager` 的 subject-level 处理使用）
   - `process_features_pipeline()`: 管道处理

2. **State Management Class (Stateful)**:
   - `PreprocessingState` 类: 负责 group-level 的状态管理、训练/测试分离和持久化。

**优势**:
- ✅ **单一事实来源**: 所有预处理逻辑（无论有状态还是无状态）都在一个文件中。
- ✅ **简化引用**: 只需要导入 `preprocessing_state` 即可。
- ✅ **代码复用**: `PreprocessingState` 类内部直接调用同文件的工具函数。

## 🔄 数据流

### Subject Level (Stateless)
```
FeatureManager
    ↓ calls
preprocess_features() (in preprocessing_state.py)
    ↓
Calculation & Transformation (Immediate)
```

### Group Level (Stateful)
```
Mode Handler (Training/Testing)
    ↓ uses
PreprocessingState (Class)
    ↓
fit() / transform()
    ↓ calls
handle_extreme_values() (Utility in same file)
```

## 📂 文件清单

### 修改的文件
```
habit/core/habitat_analysis/
├── utils/
│   └── preprocessing_state.py       ← 包含：State Class + Utility Functions
│   └── feature_preprocessing.py     ← 已删除 (Merged)
├── managers/
│   └── feature_manager.py           ← 更新导入：from ..utils.preprocessing_state import preprocess_features
```

## ✅ 验证结果

### Training 模式
```
2026-01-17 11:34:40 - INFO - Computing and applying group-level preprocessing...
2026-01-17 11:34:48 - INFO - Training bundle (model + preprocessing state) saved
2026-01-17 11:34:57 - INFO - Habitat analysis completed successfully
```

### Testing 模式
```
2026-01-17 11:35:43 - INFO - Preprocessing state not loaded yet, loading from training bundle...
2026-01-17 11:35:43 - INFO - Applying group-level preprocessing from training state...
2026-01-17 11:35:55 - INFO - Habitat analysis completed successfully
```

## 📊 总结

通过将工具函数合并到状态管理模块中，我们实现了一个高内聚的预处理子系统。
- `subject_level` 预处理使用模块级函数 `preprocess_features`。
- `group_level` 预处理使用 `PreprocessingState` 类。
两者共享底层逻辑，代码结构更加紧凑和清晰。
