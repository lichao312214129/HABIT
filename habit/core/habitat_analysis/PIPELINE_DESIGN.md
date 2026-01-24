# Habitat Analysis Pipeline 设计方案

## 一、需求分析

### 1.1 当前问题
- `habitat_analysis` 模块中的 `get_habitats` 子模块负责获取生境图谱
- 在研究中，需要在训练集内部做训练（包括特征处理和聚类模型 fit 等操作）
- 然后把训练集参数和模型应用到测试集
- 这个过程和机器学习模块很像，但目前缺乏统一的 pipeline 接口

### 1.2 设计目标
设计一个类似 sklearn 的 pipeline 机制，实现：
- **训练阶段**：在训练集上调用 `fit()`，学习参数并训练模型
- **测试阶段**：在测试集上调用 `transform()`，应用训练好的参数和模型
- **统一接口**：提供类似 sklearn Pipeline 的 `fit()` 和 `transform()` 方法
- **可组合性**：各个步骤可以独立实现，然后组合成完整的 pipeline
- **无 mode 参数**：遵循 sklearn 标准，通过 `fit()` 和 `transform()` 区分训练和测试，不需要额外的 mode 参数

## 二、架构设计

### 2.1 Pipeline 结构

```
┌─────────────────────────────────────────────────────────────┐
│                    HabitatPipeline                           │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 1: Voxel Feature Extraction                    │   │
│  │  - Extract voxel-level features from images           │   │
│  │  - May include feature selection (stateful)           │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 2: Subject-Level Preprocessing                  │   │
│  │  - Clean features (handle inf/nan)                   │   │
│  │  - Subject-level normalization (stateless)           │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 3: Individual Clustering (Voxel → Supervoxel)  │   │
│  │  - Cluster voxels to supervoxels per subject          │   │
│  │  - Generate supervoxel labels and save supervoxel map│   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 4: Supervoxel Feature Extraction (条件执行)    │   │
│  │  - Extract features for each supervoxel              │   │
│  │  - Based on supervoxel map (texture, shape, etc.)   │   │
│  │  - Per subject, per supervoxel                        │   │
│  │  - 如果只使用 mean_voxel_features，则跳过此步骤      │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 5: Supervoxel Feature Aggregation              │   │
│  │  - Aggregate features across subjects               │   │
│  │  - Combine mean features and advanced features       │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 6: Group-Level Preprocessing                    │   │
│  │  - Stateful: fit() learns params, transform() applies│   │
│  │  - Uses PreprocessingState (already exists)          │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Step 7: Population Clustering (Supervoxel → Habitat)│   │
│  │  - Stateful: fit() trains model, transform() predicts│   │
│  │  - Uses ClusteringManager                             │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件设计

#### 2.2.1 BasePipelineStep (抽象基类)

```python
from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABC, abstractmethod
from typing import Any, Optional

class BasePipelineStep(BaseEstimator, TransformerMixin, ABC):
    """
    Base class for all pipeline steps.
    Follows sklearn interface: fit() and transform()
    """
    
    @abstractmethod
    def fit(self, X: Any, y: Optional[Any] = None, **fit_params) -> 'BasePipelineStep':
        """
        Fit the step on training data.
        
        Args:
            X: Input data (can be DataFrame, dict, or other types)
            y: Optional target data
            **fit_params: Additional fitting parameters
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def transform(self, X: Any) -> Any:
        """
        Transform data using fitted parameters.
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        pass
    
    def fit_transform(self, X: Any, y: Optional[Any] = None, **fit_params) -> Any:
        """Fit and transform in one call."""
        return self.fit(X, y, **fit_params).transform(X)
```

#### 2.2.2 HabitatPipeline (主 Pipeline 类)

```python
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import joblib

class HabitatPipeline:
    """
    Main pipeline for habitat analysis.
    Similar to sklearn Pipeline but adapted for habitat-specific workflow.
    
    Follows sklearn design philosophy:
    - fit() for training: learn parameters and save state
    - transform() for testing: use saved state to transform data
    - No mode parameter needed: state is managed via fitted_ attribute
    """
    
    def __init__(
        self,
        steps: List[Tuple[str, BasePipelineStep]],
        config: Optional[Any] = None,
        load_from: Optional[str] = None
    ):
        """
        Initialize pipeline with steps.
        
        Args:
            steps: List of (name, step) tuples (ignored if load_from is provided)
            config: Configuration object
            load_from: Optional path to load saved pipeline state
        """
        if load_from:
            # Load saved pipeline
            loaded = self.load(load_from)
            self.steps = loaded.steps
            self.config = loaded.config
            self.fitted_ = loaded.fitted_
        else:
            self.steps = steps
            self.config = config
            self.fitted_ = False
        
    def fit(
        self,
        X_train: Dict[str, Any],  # Dict of subject_id -> data
        y: Optional[Any] = None,
        **fit_params
    ) -> 'HabitatPipeline':
        """
        Fit pipeline on training data.
        
        Args:
            X_train: Training data (dict of subject_id -> image/mask paths or features)
            y: Optional target data
            **fit_params: Additional fitting parameters
            
        Returns:
            self
        """
        if self.fitted_:
            raise ValueError(
                "Pipeline already fitted. Use transform() for new data, "
                "or create a new pipeline instance for training."
            )
        
        # Fit each step sequentially
        X = X_train
        for name, step in self.steps:
            X = step.fit_transform(X, y, **fit_params)
        
        self.fitted_ = True
        return self
    
    def transform(
        self,
        X_test: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Transform test data using fitted pipeline.
        
        Args:
            X_test: Test data (dict of subject_id -> image/mask paths or features)
            
        Returns:
            Results DataFrame with habitat labels
        """
        if not self.fitted_:
            raise ValueError(
                "Pipeline must be fitted before transform(). "
                "Either call fit() first, or load a saved pipeline using "
                "HabitatPipeline.load(path) or HabitatPipeline(load_from=path)"
            )
        
        # Transform each step sequentially
        X = X_test
        for name, step in self.steps:
            X = step.transform(X)
        
        return X  # Final output should be DataFrame with habitat labels
    
    def fit_transform(
        self,
        X: Dict[str, Any],
        y: Optional[Any] = None,
        **fit_params
    ) -> pd.DataFrame:
        """Fit and transform in one call."""
        return self.fit(X, y, **fit_params).transform(X)
    
    def save(self, filepath: str) -> None:
        """
        Save fitted pipeline to disk.
        
        Args:
            filepath: Path to save pipeline
        """
        if not self.fitted_:
            raise ValueError("Cannot save unfitted pipeline. Call fit() first.")
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'HabitatPipeline':
        """
        Load pipeline from disk.
        
        Args:
            filepath: Path to saved pipeline
            
        Returns:
            Loaded HabitatPipeline instance
        """
        return joblib.load(filepath)
```

#### 2.2.3 具体步骤实现示例

**Step 1: VoxelFeatureExtractor**

```python
class VoxelFeatureExtractor(BasePipelineStep):
    """
    Extract voxel-level features from images.
    """
    
    def __init__(self, feature_manager: FeatureManager):
        self.feature_manager = feature_manager
        self.fitted_ = False
    
    def fit(self, X: Dict[str, Any], y=None, **fit_params):
        """
        Fit the step (may perform feature selection in the future).
        For now, just mark as fitted.
        """
        self.fitted_ = True
        return self
    
    def transform(self, X: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Extract voxel features for each subject.
        
        Args:
            X: Dict of subject_id -> {'images': paths, 'masks': paths}
            
        Returns:
            Dict of subject_id -> feature DataFrame
        """
        results = {}
        for subject_id, data in X.items():
            _, feature_df, raw_df, mask_info = self.feature_manager.extract_voxel_features(
                subject_id
            )
            results[subject_id] = {
                'features': feature_df,
                'raw': raw_df,
                'mask_info': mask_info
            }
        return results
```

**Step 5: GroupPreprocessingStep**

```python
class GroupPreprocessingStep(BasePipelineStep):
    """
    Group-level preprocessing using PreprocessingState.
    Stateful: fit() learns statistics from training data, transform() applies to new data.
    
    Note: This step manages PreprocessingState internally, no need for external Mode classes.
    """
    
    def __init__(self, methods: List[Dict], out_dir: str):
        """
        Initialize group preprocessing step.
        
        Args:
            methods: List of preprocessing method configurations
            out_dir: Output directory for saving state (if needed)
        """
        self.preprocessing_state = PreprocessingState()  # Create internally
        self.methods = methods
        self.out_dir = out_dir
        self.fitted_ = False
    
    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        """
        Fit preprocessing state on data (learn statistics).
        
        Args:
            X: Combined supervoxel features DataFrame
        """
        self.preprocessing_state.fit(X, self.methods)
        self.fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform using fitted preprocessing state.
        
        Args:
            X: Supervoxel features DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if not self.fitted_:
            raise ValueError(
                "Must fit before transform. "
                "Either call fit() first, or load a saved pipeline."
            )
        return self.preprocessing_state.transform(X)
```

**Step 6: PopulationClusteringStep**

```python
class PopulationClusteringStep(BasePipelineStep):
    """
    Population-level clustering (supervoxel → habitat).
    Stateful: fit() trains clustering model, transform() applies to new data.
    
    Note: This step manages clustering model internally, no need for external Mode classes.
    """
    
    def __init__(
        self, 
        clustering_manager: ClusteringManager, 
        config: HabitatAnalysisConfig,
        out_dir: str
    ):
        """
        Initialize population clustering step.
        
        Args:
            clustering_manager: ClusteringManager instance
            config: Configuration object
            out_dir: Output directory for saving model (if needed)
        """
        self.clustering_manager = clustering_manager
        self.config = config
        self.out_dir = out_dir
        self.clustering_model = None  # Will be created in fit()
        self.fitted_ = False
        self.habitat_labels_ = None
        self.optimal_n_clusters_ = None
    
    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        """
        Fit clustering model on data (train model and find optimal clusters).
        
        Args:
            X: Preprocessed supervoxel features DataFrame
        """
        # Find optimal number of clusters
        optimal_n, scores = self._find_optimal_clusters(X)
        self.optimal_n_clusters_ = optimal_n
        
        # Train clustering model
        self.clustering_model = self.clustering_manager.supervoxel2habitat_clustering
        self.clustering_model.n_clusters = optimal_n
        self.clustering_model.fit(X)
        
        # Predict on training data
        self.habitat_labels_ = self.clustering_model.predict(X) + 1
        self.fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted clustering model to new data.
        
        Args:
            X: Preprocessed supervoxel features DataFrame
            
        Returns:
            DataFrame with habitat labels added
        """
        if not self.fitted_:
            raise ValueError(
                "Must fit before transform. "
                "Either call fit() first, or load a saved pipeline."
            )
        
        # Apply fitted model
        habitat_labels = self.clustering_model.predict(X) + 1
        X[ResultColumns.HABITATS] = habitat_labels
        return X
    
    def _find_optimal_clusters(self, features: pd.DataFrame) -> Tuple[int, Optional[Dict]]:
        """Find optimal number of clusters using validation methods."""
        # Implementation from TrainingMode._find_optimal_clusters
        # (moved directly into this step)
        ...
```

## 三、实现方案

### 3.1 文件结构

```
habit/core/habitat_analysis/
├── pipelines/
│   ├── __init__.py
│   ├── base_pipeline.py          # BasePipelineStep, HabitatPipeline
│   ├── steps/
│   │   ├── __init__.py
│   │   ├── voxel_feature_extractor.py
│   │   ├── subject_preprocessing.py
│   │   ├── individual_clustering.py
│   │   ├── supervoxel_aggregation.py
│   │   ├── group_preprocessing.py
│   │   └── population_clustering.py
│   └── pipeline_builder.py       # Factory to build pipelines
```

### 3.2 集成到现有代码

#### 3.2.1 修改 Strategy 类

现有的 `TwoStepStrategy` 和 `OneStepStrategy` 可以重构为使用 Pipeline：

```python
class TwoStepStrategy(BaseClusteringStrategy):
    """Two-step strategy using HabitatPipeline."""
    
    def __init__(self, analysis: "HabitatAnalysis"):
        super().__init__(analysis)
        self.pipeline = self._build_pipeline()
    
    def _build_pipeline(self) -> HabitatPipeline:
        """Build pipeline with appropriate steps."""
        from .pipelines.pipeline_builder import build_habitat_pipeline
        
        return build_habitat_pipeline(
            config=self.config,
            feature_manager=self.analysis.feature_manager,
            clustering_manager=self.analysis.clustering_manager
        )
    
    def run(
        self,
        subjects: Optional[List[str]] = None,
        save_results_csv: bool = True,
        fit_pipeline: bool = True  # True for training, False for testing
    ) -> pd.DataFrame:
        """
        Execute pipeline.
        
        Args:
            subjects: List of subjects to process
            save_results_csv: Whether to save results to CSV
            fit_pipeline: If True, fit pipeline on data (training);
                         If False, load saved pipeline and transform (testing)
        """
        # Prepare input data
        X = self._prepare_input_data(subjects)
        
        if fit_pipeline:
            # Training: fit and transform
            results_df = self.pipeline.fit_transform(X)
            
            # Save pipeline after training
            pipeline_path = os.path.join(
                self.config.out_dir, 
                'habitat_pipeline.pkl'
            )
            self.pipeline.save(pipeline_path)
        else:
            # Testing: load saved pipeline and transform
            pipeline_path = os.path.join(
                self.config.out_dir,
                'habitat_pipeline.pkl'
            )
            if not os.path.exists(pipeline_path):
                raise FileNotFoundError(
                    f"Saved pipeline not found at {pipeline_path}. "
                    "Please fit the pipeline first by calling run(fit_pipeline=True)."
                )
            self.pipeline = HabitatPipeline.load(pipeline_path)
            results_df = self.pipeline.transform(X)
        
        # Save results
        if save_results_csv:
            self._save_results(results_df)
        
        return results_df
```

#### 3.2.2 Pipeline Builder

```python
def build_habitat_pipeline(
    config: HabitatAnalysisConfig,
    feature_manager: FeatureManager,
    clustering_manager: ClusteringManager,
    mode_handler: BaseMode,
    load_from: Optional[str] = None
) -> HabitatPipeline:
    """
    Build habitat analysis pipeline based on configuration.
    
    Args:
        config: Configuration object
        feature_manager: FeatureManager instance
        clustering_manager: ClusteringManager instance
        mode_handler: Mode handler (TrainingMode or TestingMode) - used for state management
        load_from: Optional path to load saved pipeline (for transform on new data)
        
    Returns:
        Configured HabitatPipeline
    """
    # If loading from saved pipeline, return loaded instance
    if load_from:
        return HabitatPipeline(load_from=load_from)
    
    # Otherwise, build new pipeline
        from .pipelines.steps import (
        VoxelFeatureExtractor,
        SubjectPreprocessingStep,
        IndividualClusteringStep,
        SupervoxelFeatureExtractionStep,
        SupervoxelAggregationStep,
        GroupPreprocessingStep,
        PopulationClusteringStep
    )
    
    steps = [
        ('voxel_features', VoxelFeatureExtractor(feature_manager)),
        ('subject_preprocessing', SubjectPreprocessingStep(feature_manager)),
        ('individual_clustering', IndividualClusteringStep(
            feature_manager, clustering_manager, config
        )),
        ('supervoxel_feature_extraction', SupervoxelFeatureExtractionStep(
            feature_manager, config
        )),
        ('supervoxel_aggregation', SupervoxelAggregationStep(
            feature_manager, config
        )),
        ('group_preprocessing', GroupPreprocessingStep(
            config.FeatureConstruction.preprocessing_for_group_level.methods,
            config.out_dir
        )),
        ('population_clustering', PopulationClusteringStep(
            clustering_manager, config, config.out_dir
        ))
    ]
    
    return HabitatPipeline(steps=steps, config=config)
```

## 四、使用示例

### 4.1 训练阶段（使用 fit()）

```python
from habit.core.habitat_analysis import HabitatAnalysis
from habit.core.habitat_analysis.pipelines import HabitatPipeline

# Initialize analysis
analysis = HabitatAnalysis(config=config, ...)

# Get strategy (which uses pipeline internally)
strategy = TwoStepStrategy(analysis)

# Prepare training data
train_subjects = ['subject1', 'subject2', ...]

# Fit pipeline on training data
results_df = strategy.run(
    subjects=train_subjects, 
    save_results_csv=True,
    fit_pipeline=True  # Fit pipeline on training data
)

# Pipeline is automatically saved after fitting
```

### 4.2 测试阶段（使用 transform()）

```python
# Initialize analysis (same config, but will load saved pipeline)
analysis = HabitatAnalysis(config=config, ...)

# Get strategy
strategy = TwoStepStrategy(analysis)

# Prepare test data
test_subjects = ['test1', 'test2', ...]

# Transform test data (pipeline loads pre-trained model automatically)
results_df = strategy.run(
    subjects=test_subjects, 
    save_results_csv=True,
    fit_pipeline=False  # Load saved pipeline and transform
)

# Or manually load and use pipeline
pipeline = HabitatPipeline.load('path/to/habitat_pipeline.pkl')
test_data = {...}  # Prepare test data
results = pipeline.transform(test_data)
```

### 4.3 直接使用 Pipeline（高级用法）

```python
from habit.core.habitat_analysis.pipelines import build_habitat_pipeline, HabitatPipeline

# Build pipeline
pipeline = build_habitat_pipeline(
    config=config,
    feature_manager=feature_manager,
    clustering_manager=clustering_manager
)

# Fit on training data
train_data = {...}  # Dict of subject_id -> data
pipeline.fit(train_data)

# Save pipeline
pipeline.save('habitat_pipeline.pkl')

# Load and use on test data - Method 1: Explicit load
pipeline = HabitatPipeline.load('habitat_pipeline.pkl')
test_data = {...}
results = pipeline.transform(test_data)

# Method 2: Load during initialization
pipeline = HabitatPipeline(load_from='habitat_pipeline.pkl')
results = pipeline.transform(test_data)

# Method 3: Use builder with load_from
pipeline = build_habitat_pipeline(
    config=config,
    feature_manager=feature_manager,
    clustering_manager=clustering_manager,
    load_from='habitat_pipeline.pkl'
)
results = pipeline.transform(test_data)
```

## 五、优势

1. **统一接口**：类似 sklearn，易于理解和使用
2. **可组合性**：每个步骤独立，易于扩展和修改
3. **状态管理**：自动处理训练/测试状态分离
4. **可序列化**：Pipeline 可以保存和加载
5. **向后兼容**：可以与现有 Strategy 模式共存
6. **易于测试**：每个步骤可以独立测试

## 六、实施步骤

1. **Phase 1**: 实现基础框架
   - 创建 `BasePipelineStep` 和 `HabitatPipeline` 基类
   - 实现简单的步骤示例

2. **Phase 2**: 实现核心步骤
   - 实现所有 6 个步骤
   - 集成到现有 Strategy 中

3. **Phase 3**: 测试和优化
   - 单元测试每个步骤
   - 集成测试完整 pipeline
   - 性能优化

4. **Phase 4**: 文档和示例
   - 更新文档
   - 添加使用示例

## 七、关键设计问题讨论

### 7.1 关于 Mode 参数的必要性

**问题**：如果使用 sklearn 的 pipeline 构建，还需要区分 mode 是 training 还是 testing 吗？

**讨论**：

在 sklearn 的设计哲学中：
- `fit()` 用于训练，学习参数并保存状态
- `transform()` 用于测试，使用已保存的状态进行转换
- 不需要额外的 `mode` 参数，状态通过 `fitted_` 属性来管理

**方案选择**：

**方案 A：完全遵循 sklearn 设计（推荐）**
- Pipeline 不需要 `mode` 参数
- 训练时：`pipeline.fit(X_train)` → 自动保存状态
- 测试时：`pipeline = HabitatPipeline.load(path)` → `pipeline.transform(X_test)`
- 优点：完全符合 sklearn 标准，易于理解和使用
- 缺点：需要显式保存和加载 pipeline

**方案 B：保留 mode 参数（已废弃）**
- ~~Pipeline 保留 `mode` 参数用于内部逻辑判断~~
- ~~训练时：`pipeline.fit(X_train)`，mode='training'~~
- ~~测试时：`pipeline.transform(X_test)`，mode='testing'，自动从配置路径加载状态~~
- 缺点：不完全符合 sklearn 标准，增加不必要的复杂性
- **决定**：不采用此方案，完全遵循 sklearn 标准

**推荐方案**：采用**方案 A**，完全遵循 sklearn 标准。

**重要说明**：
- Pipeline 层面：**不暴露 mode 参数**，通过 `fit()` 和 `transform()` 区分
- **新设计**：Pipeline Steps 内部直接管理状态，不需要 Mode 类：
  - `GroupPreprocessingStep` 内部管理 `PreprocessingState`
  - `PopulationClusteringStep` 内部管理聚类模型
  - 完全符合 sklearn 的"每个 Step 自己管理状态"的理念
- **说明**：`TrainingMode` 和 `TestingMode` 已移除，Pipeline Steps 直接管理状态

**实现方式**：

```python
class HabitatPipeline:
    def __init__(
        self,
        steps: List[Tuple[str, BasePipelineStep]],
        config: Optional[Any] = None,
        load_from: Optional[str] = None  # Optional: load saved pipeline
    ):
        """
        Initialize pipeline.
        
        Args:
            steps: List of (name, step) tuples
            config: Configuration object
            load_from: Optional path to load saved pipeline state
        """
        if load_from:
            # Load saved pipeline
            loaded = self.load(load_from)
            self.steps = loaded.steps
            self.config = loaded.config
            self.fitted_ = loaded.fitted_
        else:
            self.steps = steps
            self.config = config
            self.fitted_ = False
    
    def fit(self, X_train: Dict[str, Any], y=None, **fit_params):
        """Fit pipeline on training data."""
        if self.fitted_:
            raise ValueError("Pipeline already fitted. Use transform() for new data.")
        
        X = X_train
        for name, step in self.steps:
            X = step.fit_transform(X, y, **fit_params)
        
        self.fitted_ = True
        return self
    
    def transform(self, X_test: Dict[str, Any]) -> pd.DataFrame:
        """Transform test data using fitted pipeline."""
        if not self.fitted_:
            raise ValueError(
                "Pipeline must be fitted before transform(). "
                "Either call fit() first, or load a saved pipeline using "
                "HabitatPipeline.load(path) or HabitatPipeline(load_from=path)"
            )
        
        X = X_test
        for name, step in self.steps:
            X = step.transform(X)
        
        return X
```

**使用方式**：

```python
# 训练阶段：使用 fit()
pipeline = build_habitat_pipeline(config, ...)
pipeline.fit(train_data)
pipeline.save('pipeline.pkl')

# 测试阶段 - 方式1：显式加载
pipeline = HabitatPipeline.load('pipeline.pkl')
results = pipeline.transform(test_data)

# 测试阶段 - 方式2：初始化时加载
pipeline = HabitatPipeline(load_from='pipeline.pkl')
results = pipeline.transform(test_data)
```

### 7.2 二步法中 Supervoxel 特征提取的详细说明

**补充信息**：在二步法中，voxel 到 supervoxel 后，会在超体素图上再提取每个超体素的特征（纹理、均值等）。

**正确的流程**（根据代码分析）：

1. **Step 3: Individual Clustering (Voxel → Supervoxel)**
   - 对每个 subject，将 voxels 聚类成 supervoxels
   - 生成 supervoxel 标签数组（supervoxel_labels）
   - **保存 supervoxel 标签图文件**（supervoxel map），用于后续特征提取
   - 同时计算 supervoxel 的均值特征（mean voxel features）- 这是基于 voxel features 的聚合

2. **Step 4: Supervoxel Feature Extraction（新增独立步骤）**
   - **基于 supervoxel map 文件提取每个 supervoxel 的特征**
   - 对每个 subject、每个 supervoxel，提取：
     - 纹理特征（GLCM, GLRLM, GLSZM 等）
     - 形状特征（体积、表面积、球形度等）
     - 其他 radiomics 特征
   - 使用 `extract_supervoxel_features()` 方法
   - 输入：supervoxel map 文件 + 原始图像
   - 输出：每个 supervoxel 的特征 DataFrame

3. **Step 5: Supervoxel Feature Aggregation**
   - 聚合所有 subjects 的 supervoxel 特征
   - 如果配置使用 `mean_voxel_features`：使用 Step 3 中计算的均值特征
   - 如果配置使用高级特征：使用 Step 4 中提取的特征
   - 或者合并两者（mean features + advanced features）

**Step 4 的详细设计**：

```python
class SupervoxelFeatureExtractionStep(BasePipelineStep):
    """
    Extract advanced features for each supervoxel based on supervoxel maps.
    
    This step extracts advanced features (texture, shape, radiomics) from 
    supervoxel label maps. It runs after supervoxel clustering and requires
    supervoxel map files to be saved.
    
    **Important**: This step is conditionally included in the pipeline based on
    configuration. If only `mean_voxel_features()` is used, this step is skipped
    to save computation time.
    """
    
    def __init__(
        self,
        feature_manager: FeatureManager,
        config: HabitatAnalysisConfig
    ):
        self.feature_manager = feature_manager
        self.config = config
        self.fitted_ = False
    
    def fit(self, X: Dict[str, Dict], y=None, **fit_params):
        """
        Fit step: setup supervoxel file discovery for feature extraction.
        
        Args:
            X: Dict of subject_id -> {
                'supervoxel_labels': supervoxel labels array,
                'mask_info': mask information
            }
        """
        # Setup supervoxel file dictionary for feature extraction
        # This discovers supervoxel map files saved in Step 3
        subjects = list(X.keys())
        self.feature_manager.setup_supervoxel_files(
            subjects, 
            failed_subjects=[],
            out_dir=self.config.out_dir
        )
        
        self.fitted_ = True
        return self
    
    def transform(self, X: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """
        Extract supervoxel-level features for each subject.
        
        Args:
            X: Dict of subject_id -> {
                'supervoxel_labels': supervoxel labels array,
                'mask_info': mask information
            }
            
        Returns:
            Dict of subject_id -> supervoxel features DataFrame
        """
        supervoxel_features = {}
        
        for subject_id, data in X.items():
            # Extract advanced features from supervoxel maps
            # This uses the supervoxel map file saved in Step 3
            # Uses extract_supervoxel_features() method
            _, features_df = self.feature_manager.extract_supervoxel_features(
                subject_id
            )
            supervoxel_features[subject_id] = features_df
        
        return supervoxel_features
```

**Step 5 的详细设计**：

```python
class SupervoxelAggregationStep(BasePipelineStep):
    """
    Aggregate voxel features to supervoxel level.
    
    For two-step strategy:
    1. Calculate mean voxel features per supervoxel (always done in Step 3)
    2. Optionally merge with advanced features from Step 4 (if Step 4 was executed)
    
    **Important**: This step always calculates mean features. If Step 4 was executed,
    it merges the advanced features from Step 4's output.
    """
    
    def __init__(
        self,
        feature_manager: FeatureManager,
        config: HabitatAnalysisConfig
    ):
        self.feature_manager = feature_manager
        self.config = config
        self.fitted_ = False
    
    def fit(self, X: Dict[str, Dict], y=None, **fit_params):
        """
        Fit step: no state needed (stateless step).
        
        Args:
            X: Dict of subject_id -> {
                'features': voxel features DataFrame,
                'raw': raw features DataFrame,
                'mask_info': mask information,
                'supervoxel_labels': supervoxel labels array
            }
        """
        # No state to save
        self.fitted_ = True
        return self
    
    def transform(
        self, 
        X: Dict[str, Dict],
        supervoxel_features: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Aggregate features to supervoxel level.
        
        Args:
            X: Dict of subject_id -> {
                'features': voxel features DataFrame,
                'raw': raw features DataFrame,
                'mask_info': mask information,
                'supervoxel_labels': supervoxel labels array
            }
            supervoxel_features: Optional dict from Step 4 (if Step 4 was executed)
                - If provided: merge advanced features with mean features
                - If None: only use mean features (Step 4 was skipped)
            
        Returns:
            Combined DataFrame with supervoxel-level features
        """
        all_supervoxel_features = []
        
        for subject_id, data in X.items():
            feature_df = data['features']
            raw_df = data['raw']
            supervoxel_labels = data['supervoxel_labels']
            n_clusters = len(np.unique(supervoxel_labels))
            
            # Always calculate mean voxel features per supervoxel
            # (This was also done in Step 3, but we recalculate here for consistency)
            mean_features_df = self.feature_manager.calculate_supervoxel_means(
                subject_id, feature_df, raw_df, supervoxel_labels, n_clusters
            )
            
            # If Step 4 was executed, merge advanced features
            if supervoxel_features is not None and subject_id in supervoxel_features:
                advanced_features_df = supervoxel_features[subject_id]
                
                # Merge mean features with advanced features
                # Use subject and supervoxel columns as keys
                mean_features_df = mean_features_df.merge(
                    advanced_features_df,
                    on=[ResultColumns.SUBJECT, ResultColumns.SUPERVOXEL],
                    how='left'
                )
            
            all_supervoxel_features.append(mean_features_df)
        
        # Combine all subjects' supervoxel features
        combined_df = pd.concat(all_supervoxel_features, ignore_index=True)
        
        return combined_df
```

### 7.3 数据格式统一

**问题**：Pipeline 各步骤之间的数据格式需要统一。

**设计**：

```python
# Step 1-3 的输出格式
Dict[str, Dict] = {
    'subject1': {
        'features': pd.DataFrame,      # Voxel features
        'raw': pd.DataFrame,           # Raw features
        'mask_info': dict,             # Mask metadata
        'supervoxel_labels': np.ndarray  # Supervoxel labels (after Step 3)
    },
    'subject2': {...},
    ...
}

# Step 4 的输出格式（如果执行）
Dict[str, pd.DataFrame] = {
    'subject1': pd.DataFrame,  # Advanced supervoxel features (texture, shape, etc.)
    'subject2': pd.DataFrame,
    ...
}
# 注意：如果 Step 4 被跳过（只使用 mean_voxel_features），则此步骤不执行

# Step 5 的输入格式
# - X: Dict[str, Dict] (from Step 3)
# - supervoxel_features: Optional[Dict[str, pd.DataFrame]] (from Step 4, if executed)
# Step 5 的输出格式
pd.DataFrame  # Combined supervoxel features for all subjects

# Step 4 的输出格式（如果执行）
Dict[str, pd.DataFrame] = {
    'subject1': DataFrame with columns: SupervoxelID, Feature1, Feature2, ...
    'subject2': DataFrame with columns: SupervoxelID, Feature1, Feature2, ...
    ...
}
# 注意：如果 Step 4 被跳过（只使用 mean_voxel_features），则此步骤不执行

# Step 5 的输入格式
# - X: Dict[str, Dict] (from Step 3)
# - supervoxel_features: Optional[Dict[str, pd.DataFrame]] (from Step 4, if executed)
# Step 5 的输出格式
pd.DataFrame = Combined supervoxel features with columns:
    - Subject (str)
    - Supervoxel (int)
    - Feature1, Feature2, ... (float)  # Mean features + advanced features (if Step 4 executed)
    - (metadata columns)

# Step 6-7 的输入输出格式
pd.DataFrame = Same as Step 5 output, with additional columns:
    - Habitats (int)  # Added in Step 7
```

### 7.4 状态管理详细分析

本节详细分析每个步骤是否需要保存状态，并给出明确的理由。

#### 7.4.1 完整的状态管理分析

**问题**：训练集内部，个体水平的体素特征提取、预处理和聚类，哪些是要保存状态，后续用于测试集，哪些是不需要？

**分析**：

1. **体素特征提取（Step 1）**
   - **不需要保存状态**
   - 原因：特征提取逻辑是固定的，基于图像和mask，不依赖训练数据
   - 训练和测试时使用相同的提取逻辑
   - 例如：提取原始像素值、纹理特征等，这些逻辑是配置决定的，不是从数据学习的

2. **个体预处理（Step 2）**
   - **不需要保存状态**
   - 原因：每个subject独立处理，使用该subject自身的数据进行归一化
   - 训练时：对每个subject的特征进行归一化（如z-score，使用该subject的均值和标准差）
   - 测试时：同样对每个subject独立归一化，使用该subject的均值和标准差
   - 注意：这是**subject-level**的预处理，与**group-level**预处理不同

3. **个体聚类（Step 3: Voxel → Supervoxel）**
   - **不需要保存每个subject的聚类模型**
   - 原因：
     - 聚类参数（如n_clusters）是配置固定的，不是从数据学习的
     - 每个subject独立聚类，训练和测试时使用相同的参数
     - 对于one-step策略，每个subject的最优cluster数可能不同，但这是通过验证方法实时计算的，不需要保存
   - 特殊情况：
     - 如果使用固定n_clusters：训练和测试都使用相同值
     - 如果使用最优cluster数（one-step）：训练和测试时都会重新计算，不需要保存

**总结**：个体水平的所有操作都是**无状态的**，因为：
- 特征提取逻辑固定
- 预处理基于每个subject自身数据
- 聚类参数固定或实时计算

#### 7.4.2 每个步骤的详细分析

**Step 1: Voxel Feature Extraction（体素特征提取）**
- **状态需求**：❌ **不需要保存状态**
- **代码位置**：`FeatureManager.extract_voxel_features()`
- **详细理由**：
  1. 特征提取逻辑完全由配置决定（`voxel_level.method` 和 `params`）
  2. 提取过程是确定性的：基于图像和mask，使用固定的算法（如raw、texture等）
  3. 训练和测试时使用完全相同的提取逻辑和参数
  4. 不依赖任何从训练数据学习到的参数
  5. 每个subject的特征提取是独立的，不涉及跨subject的信息
- **结论**：这是**无状态操作**，每次调用都是独立的

**Step 2: Subject Preprocessing（个体预处理）**
- **状态需求**：❌ **不需要保存状态**
- **代码位置**：`FeatureManager.apply_preprocessing(level='subject')`
- **详细理由**：
  1. 每个subject独立进行预处理，使用该subject自身的数据计算统计量
  2. 例如z-score归一化：使用该subject的均值和标准差，不依赖其他subject
  3. 训练时：对subject A的特征归一化，使用subject A的均值和标准差
  4. 测试时：对subject B的特征归一化，使用subject B的均值和标准差
  5. 每个subject的预处理参数是实时计算的，不需要保存
  6. 与group-level预处理不同：group-level需要在所有训练subjects上计算统计量
- **结论**：这是**无状态操作**，每个subject独立处理

**Step 3: Individual Clustering（个体聚类：Voxel → Supervoxel）**
- **状态需求**：❌ **不需要保存状态**
- **代码位置**：`ClusteringManager.cluster_subject_voxels()`
- **详细理由**：
  1. 聚类参数完全由配置决定：
     - `n_clusters`：固定的supervoxel数量（如50）
     - `algorithm`：聚类算法（如kmeans）
     - `random_state`：随机种子
  2. 每个subject独立聚类，不依赖其他subject的数据
  3. 训练时：对subject A的voxels聚类，使用配置的n_clusters
  4. 测试时：对subject B的voxels聚类，使用相同的n_clusters
  5. 每次调用`cluster_subject_voxels`都会创建新的聚类器或使用默认实例，不保存模型
  6. 特殊情况（one-step策略）：
     - 如果使用最优cluster数，通过验证方法实时计算
     - 训练和测试时都会重新计算，不需要保存
- **结论**：这是**无状态操作**，聚类参数固定或实时计算

**Step 4: Supervoxel Feature Extraction（Supervoxel高级特征提取）**
- **状态需求**：❌ **不需要保存状态**
- **执行条件**：⚠️ **条件执行** - 根据配置决定是否执行
  - 如果 `config.FeatureConstruction.supervoxel_level.method` 只包含 `'mean_voxel_features'`，则**跳过此步骤**以节省时间
  - 如果包含其他方法（如 `'supervoxel_radiomics'`），则**执行此步骤**
- **代码位置**：`FeatureManager.extract_supervoxel_features()`
- **详细理由**：
  1. 基于supervoxel标签图提取高级特征（纹理、形状、radiomics等）
  2. 提取算法固定，不依赖训练数据
  3. 训练和测试时使用相同的提取逻辑
  4. 每个subject的特征提取是独立的
  5. **优化**：如果只需要均值特征，跳过此步骤可以显著节省计算时间
- **结论**：这是**无状态操作**，但**条件执行**以优化性能

**Step 5: Supervoxel Aggregation（Supervoxel特征聚合）**
- **状态需求**：❌ **不需要保存状态**
- **代码位置**：`FeatureManager.calculate_supervoxel_means()` 和可选的 Step 4 输出
- **详细理由**：
  1. 总是计算均值特征：对每个supervoxel内的voxels求均值，是纯数学计算
  2. 如果 Step 4 被执行，则合并 Step 4 输出的高级特征
  3. 如果 Step 4 被跳过，则只使用均值特征
  4. 不依赖任何从训练数据学习到的参数
  5. 训练和测试时使用相同的计算逻辑
  6. 每个subject的特征聚合是独立的
- **结论**：这是**无状态操作**，能够处理有/无 Step 4 输出的两种情况

**Step 5: Group Preprocessing（群体预处理）**
- **状态需求**：✅ **需要保存状态**
   - **代码位置**：`GroupPreprocessingStep.fit()` 和 `GroupPreprocessingStep.transform()`
   - **实现方式**：Step 内部直接管理 `PreprocessingState`，不需要外部 Mode 类
   - **状态对象**：`PreprocessingState`（包含均值、标准差、分箱参数、winsorization limits等）
   - **详细理由**：
    1. **fit() 时**：
     - 在**所有训练集subjects的supervoxel特征**上计算统计量
     - 例如z-score：计算所有训练subjects的总体均值和标准差
     - 这些统计量代表训练集的整体分布特征
    2. **transform() 时**：
     - 必须使用 fit() 时保存的统计量对新数据进行归一化
     - 不能使用新数据自身的统计量（会导致数据分布不一致）
  3. **为什么必须保存**：
     - 确保测试集和训练集使用相同的归一化参数
     - 避免数据分布偏移（distribution shift）
     - 保证模型在测试集上的性能
  4. **保存的内容**：
     - 均值、标准差（用于z-score）
     - 最小值、最大值（用于min-max）
     - 分位数（用于robust normalization）
     - 分箱器（discretizer）
     - Winsorization limits
     - 其他预处理参数
- **结论**：这是**有状态操作**，必须在训练时保存，测试时使用

**Step 6: Population Clustering（群体聚类：Supervoxel → Habitat）**
- **状态需求**：✅ **需要保存状态**
   - **代码位置**：`PopulationClusteringStep.fit()` 和 `PopulationClusteringStep.transform()`
   - **实现方式**：Step 内部直接管理聚类模型，不需要外部 Mode 类
   - **状态对象**：聚类模型（`supervoxel2habitat_clustering`）和最优cluster数
   - **详细理由**：
    1. **fit() 时**：
     - 在**所有训练集subjects的supervoxel特征**上训练聚类模型
     - 确定最优的habitat数量（通过验证方法）
     - 训练好的模型定义了habitat的划分标准
    2. **transform() 时**：
     - 必须使用 fit() 时训练好的模型对新数据进行预测
     - 不能重新训练模型（会导致habitat定义不一致）
  3. **为什么必须保存**：
     - 确保测试集和训练集使用相同的habitat定义
     - Habitat的编号和特征必须一致，才能进行跨数据集比较
     - 保证研究结果的可重复性
  4. **保存的内容**：
     - 训练好的聚类模型（包含聚类中心、参数等）
     - 最优的habitat数量（optimal_n_clusters）
     - 聚类算法的其他参数
- **结论**：这是**有状态操作**，必须在训练时保存，测试时使用

#### 7.4.3 状态管理总结表

| 步骤 | 操作类型 | 是否需要保存状态 | 关键理由 | 保存的内容 |
|------|---------|-----------------|---------|-----------|
| **Step 1** | 体素特征提取 | ❌ **否** | 提取逻辑固定，由配置决定，不依赖训练数据 | - |
| **Step 2** | 个体预处理 | ❌ **否** | 每个subject独立处理，使用自身数据计算统计量 | - |
| **Step 3** | 个体聚类 | ❌ **否** | 聚类参数固定（配置）或实时计算，每个subject独立 | - |
| **Step 4** | Supervoxel特征提取 | ❌ **否** | 基于supervoxel map的确定性特征提取，算法固定 | ⚠️ **条件执行**：根据配置决定是否执行 |
| **Step 5** | Supervoxel聚合 | ❌ **否** | 合并和聚合特征，确定性操作 | 能够处理有/无 Step 4 输出的两种情况 |
| **Step 6** | 群体预处理 | ✅ **是** | 需要在训练集上计算统计量，测试时使用相同参数 | PreprocessingState对象（均值、标准差等） |
| **Step 7** | 群体聚类 | ✅ **是** | 需要在训练集上训练模型，测试时使用相同模型 | 聚类模型和最优cluster数 |

#### 7.4.4 个体水平操作的状态管理（补充说明）

#### 7.4.5 关键原则总结

**判断是否需要保存状态的核心原则**：

1. **是否依赖 fit() 时的数据**：
   - ❌ 不依赖：使用配置参数或每个subject自身数据 → 不需要保存状态
   - ✅ 依赖：在 fit() 时计算统计量或训练模型 → 需要保存状态

2. **是否跨subject共享**：
   - ❌ 不共享：每个subject独立处理 → 不需要保存状态
   - ✅ 共享：所有subjects使用相同的参数或模型 → 需要保存状态

3. **fit() 和 transform() 是否必须一致**：
   - ❌ 可以不一致：每个subject独立处理，结果可以不同 → 不需要保存状态
   - ✅ 必须一致：transform() 时必须使用 fit() 时的参数/模型 → 需要保存状态

**实际应用**：
- **个体水平操作（Step 1-3）**：每个subject独立，不依赖fit()时的数据 → 全部无状态
- **群体水平操作（Step 6-7）**：跨subject共享，依赖fit()时的数据 → 全部有状态
- **特征提取和聚合（Step 4-5）**：确定性计算，不依赖fit()时的数据 → 无状态

## 八、设计决策总结

### 8.1 关键设计决策

1. **不采用 mode 参数（遵循 sklearn 标准）**
   - **Pipeline 层面**：不需要 `mode` 参数
   - **使用方式**：通过 `fit()` 和 `transform()` 区分训练和测试
     - `fit(X_train)`：在训练数据上学习参数并保存状态
     - `transform(X_test)`：在新数据上应用保存的状态
   - **加载方式**：测试时通过 `load()` 或 `load_from` 参数加载已保存的 pipeline
   - **底层实现**：不再依赖 `TrainingMode` / `TestingMode`
   - **优点**：完全符合 sklearn 标准，易于理解和维护，用户不需要关心底层 mode 细节

2. **Supervoxel 特征提取的两阶段设计**
   - **阶段 1（必需）**：计算均值特征（mean voxel features）
   - **阶段 2（可选）**：提取高级特征（纹理、形状、radiomics）
   - 通过配置 `supervoxel_level.method` 控制是否提取高级特征

3. **状态管理策略**
   - 有状态的步骤：Step 6（群体预处理）、Step 7（群体聚类）
   - 无状态的步骤：Step 1（特征提取）、Step 2（个体预处理）、Step 3（个体聚类）、Step 4（Supervoxel特征提取，条件执行）、Step 5（Supervoxel聚合）
   - **注意**：Step 3（个体聚类）不需要保存状态，因为每个subject独立聚类，聚类参数是配置固定的
   - 所有状态在 `fit()` 时保存，在 `transform()` 时使用

4. **数据格式统一**
   - Step 1-3：`Dict[str, Dict]` 格式（按 subject 组织）
   - Step 4：`Dict[str, pd.DataFrame]` 格式（每个subject的supervoxel特征）
   - Step 5-7：`pd.DataFrame` 格式（合并所有 subjects）
   - 每个步骤明确输入输出格式

### 8.2 不同策略的 Pipeline 设计

**问题**：设计的 pipeline 要考虑到不同的策略，比如一步法、二步法以及 pooling 所有人所有体素法之间的区别，以及是否适用一套 pipeline。

**分析**：

三种策略的流程对比：

| 策略 | 流程 | 关键区别 |
|------|------|----------|
| **一步法 (One-Step)** | Voxel → Habitat (per subject) | 每个subject独立聚类，无群体级步骤 |
| **二步法 (Two-Step)** | Voxel → Supervoxel → Habitat | 有群体级聚类步骤 |
| **Pooling法 (Direct Pooling)** | All Voxels → Habitat (single clustering) | 跳过supervoxel，直接对所有voxel聚类 |

**设计方案：策略特定的 Pipeline**

**方案 A：每个策略独立的 Pipeline（推荐）**

每个策略有自己的 Pipeline 步骤组合：

```python
# 二步法 Pipeline
two_step_steps = [
    ('voxel_features', VoxelFeatureExtractor(...)),
    ('subject_preprocessing', SubjectPreprocessingStep(...)),
    ('individual_clustering', IndividualClusteringStep(...)),  # Voxel → Supervoxel
    ('supervoxel_aggregation', SupervoxelAggregationStep(...)),
    ('group_preprocessing', GroupPreprocessingStep(...)),
    ('population_clustering', PopulationClusteringStep(...))  # Supervoxel → Habitat
]

# 一步法 Pipeline
one_step_steps = [
    ('voxel_features', VoxelFeatureExtractor(...)),
    ('subject_preprocessing', SubjectPreprocessingStep(...)),
    ('individual_clustering', IndividualClusteringStep(...)),  # Voxel → Habitat (per subject)
    # 无群体级步骤
]

# Pooling法 Pipeline
pooling_steps = [
    ('voxel_features', VoxelFeatureExtractor(...)),
    ('subject_preprocessing', SubjectPreprocessingStep(...)),
    ('concatenate_all_voxels', ConcatenateVoxelsStep(...)),  # 合并所有voxel
    ('group_preprocessing', GroupPreprocessingStep(...)),
    ('population_clustering', PopulationClusteringStep(...))  # All Voxels → Habitat
]
```

**Pipeline Builder 设计**：

```python
def build_habitat_pipeline(
    config: HabitatAnalysisConfig,
    feature_manager: FeatureManager,
    clustering_manager: ClusteringManager,
    mode_handler: BaseMode,
    load_from: Optional[str] = None
) -> HabitatPipeline:
    """
    Build habitat analysis pipeline based on strategy.
    
    Args:
        config: Configuration object
        feature_manager: FeatureManager instance
        clustering_manager: ClusteringManager instance
        mode_handler: Mode handler (TrainingMode or TestingMode) - used for state management
        load_from: Optional path to load saved pipeline (for transform on new data)
        
    Returns:
        Configured HabitatPipeline for the specified strategy
    """
    if load_from:
        return HabitatPipeline(load_from=load_from)
    
    strategy = config.HabitatsSegmention.clustering_mode
    
    if strategy == 'two_step':
        return _build_two_step_pipeline(
            config, feature_manager, clustering_manager, mode_handler
        )
    elif strategy == 'one_step':
        return _build_one_step_pipeline(
            config, feature_manager, clustering_manager, mode_handler
        )
    elif strategy == 'direct_pooling':
        return _build_pooling_pipeline(
            config, feature_manager, clustering_manager, mode_handler
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def _build_two_step_pipeline(...) -> HabitatPipeline:
    """Build two-step strategy pipeline."""
    steps = [
        ('voxel_features', VoxelFeatureExtractor(feature_manager)),
        ('subject_preprocessing', SubjectPreprocessingStep(feature_manager)),
        ('individual_clustering', IndividualClusteringStep(
            feature_manager, clustering_manager, config,
            target='supervoxel'  # 聚类到supervoxel
        )),
        ('supervoxel_feature_extraction', SupervoxelFeatureExtractionStep(
            feature_manager, config
        )),
        ('supervoxel_aggregation', SupervoxelAggregationStep(
            feature_manager, config
        )),
        ('group_preprocessing', GroupPreprocessingStep(
            mode_handler.preprocessing_state,
            config.FeatureConstruction.preprocessing_for_group_level.methods
        )),
        ('population_clustering', PopulationClusteringStep(
            clustering_manager, mode_handler
        ))
    ]
    return HabitatPipeline(steps=steps, config=config)

def _build_one_step_pipeline(
    config: HabitatAnalysisConfig,
    feature_manager: FeatureManager,
    clustering_manager: ClusteringManager
) -> HabitatPipeline:
    """Build one-step strategy pipeline."""
    steps = [
        ('voxel_features', VoxelFeatureExtractor(feature_manager)),
        ('subject_preprocessing', SubjectPreprocessingStep(feature_manager)),
        ('individual_clustering', IndividualClusteringStep(
            feature_manager, clustering_manager, config,
            target='habitat',  # 直接聚类到habitat
            find_optimal=True  # 每个subject找最优cluster数
        )),
        # 无群体级步骤，直接输出结果
    ]
    return HabitatPipeline(steps=steps, config=config)

def _build_pooling_pipeline(
    config: HabitatAnalysisConfig,
    feature_manager: FeatureManager,
    clustering_manager: ClusteringManager
) -> HabitatPipeline:
    """Build direct pooling strategy pipeline."""
    steps = [
        ('voxel_features', VoxelFeatureExtractor(feature_manager)),
        ('subject_preprocessing', SubjectPreprocessingStep(feature_manager)),
        ('concatenate_voxels', ConcatenateVoxelsStep()),  # 合并所有voxel
        ('group_preprocessing', GroupPreprocessingStep(
            config.FeatureConstruction.preprocessing_for_group_level.methods,
            config.out_dir
        )),
        ('population_clustering', PopulationClusteringStep(
            clustering_manager, config, config.out_dir,
            input_type='voxel'  # 对voxel聚类，不是supervoxel
        ))
    ]
    return HabitatPipeline(steps=steps, config=config)
```

**关键设计点**：

1. **共享的 Steps**：
   - `VoxelFeatureExtractor`：所有策略都需要
   - `SubjectPreprocessingStep`：所有策略都需要（个体级预处理）

2. **策略特定的 Steps**：
   - **二步法**：需要 `SupervoxelAggregationStep` 和群体级步骤
   - **一步法**：不需要群体级步骤，`IndividualClusteringStep` 直接输出habitat
   - **Pooling法**：需要 `ConcatenateVoxelsStep`，跳过supervoxel步骤

3. **Step 的可配置性**：
   - `IndividualClusteringStep` 可以通过参数控制：
     - `target`: 'supervoxel' 或 'habitat'
     - `find_optimal`: 是否找最优cluster数
   - `PopulationClusteringStep` 可以通过参数控制：
     - `input_type`: 'supervoxel' 或 'voxel'

**方案 B：统一的 Pipeline（不推荐）**

尝试用一套 Pipeline 适配所有策略，通过条件判断选择步骤。这种方案：
- 优点：代码复用度高
- 缺点：逻辑复杂，难以维护，违反单一职责原则

**推荐采用方案 A**：每个策略有独立的 Pipeline 构建函数，但共享通用的 Step 实现。

**三种策略的 Pipeline 步骤对比表**：

| 步骤 | 二步法 | 一步法 | Pooling法 |
|------|--------|--------|-----------|
| 1. 体素特征提取 | ✅ | ✅ | ✅ |
| 2. 个体预处理 | ✅ | ✅ | ✅ |
| 3. 个体聚类 (Voxel→Supervoxel) | ✅ | ❌ | ❌ |
| 3'. 个体聚类 (Voxel→Habitat) | ❌ | ✅ | ❌ |
| 4. Supervoxel特征提取 | ✅ | ❌ | ❌ |
| 5. Supervoxel特征聚合 | ✅ | ❌ | ❌ |
| 4'. 合并所有Voxel | ❌ | ❌ | ✅ |
| 6. 群体预处理 | ✅ | ❌ | ✅ |
| 7. 群体聚类 (Supervoxel→Habitat) | ✅ | ❌ | ❌ |
| 6'. 群体聚类 (Voxel→Habitat) | ❌ | ❌ | ✅ |

**关键区别**：
- **二步法**：有supervoxel中间层，需要群体级步骤
  - **重要**：在individual_clustering后，需要先提取每个supervoxel的特征（Step 4），然后再聚合（Step 5）
- **一步法**：无群体级步骤，每个subject独立完成
- **Pooling法**：跳过supervoxel，直接对所有voxel进行群体聚类

**二步法的正确流程确认**：
1. Individual Clustering (Voxel → Supervoxel) - 生成supervoxel labels和map文件
2. **Supervoxel Feature Extraction** - 基于supervoxel map提取每个supervoxel的特征（纹理、形状等）
3. **Supervoxel Aggregation** - 聚合所有subjects的supervoxel特征
4. Group Preprocessing - 群体预处理
5. Population Clustering - 群体聚类

**注意**：Step 4（Supervoxel Feature Extraction）是**条件执行的独立步骤**：
- **执行条件**：根据 `config.FeatureConstruction.supervoxel_level.method` 判断
  - 如果只包含 `'mean_voxel_features'`：**跳过 Step 4**，直接进入 Step 5（节省计算时间）
  - 如果包含其他方法（如 `'supervoxel_radiomics'`）：**执行 Step 4**
- **执行时机**：必须在 individual_clustering 之后、aggregation 之前执行（如果执行）
- **原因**：
  - 它需要supervoxel map文件（在Step 3中保存）
  - 它提取的是基于supervoxel空间分布的特征（不同于Step 3中计算的均值特征）
  - 这些特征需要与均值特征合并，然后才能进行群体级操作
- **优化目的**：如果只需要均值特征，跳过 Step 4 可以显著节省计算时间

### 8.3 TrainingMode 和 TestingMode 的必要性分析

**问题**：既然采用了 sklearn 的设计模式（通过 `fit()` 和 `transform()` 区分），`TrainingMode` 和 `TestingMode` 还有必要单独分开存在吗？

**当前 TrainingMode 和 TestingMode 的功能**：

1. **TrainingMode**：
   - `cluster_habitats()`: 训练聚类模型，找最优 cluster 数
   - `process_features()`: fit PreprocessingState 并 transform
   - `save_model()`: 保存模型和 PreprocessingState bundle
   - 管理 PreprocessingState 的生命周期（创建和保存）

2. **TestingMode**：
   - `cluster_habitats()`: 加载已保存的模型并 predict
   - `process_features()`: 加载 PreprocessingState 并 transform
   - `load_model()`: 加载模型和 PreprocessingState bundle
   - 管理 PreprocessingState 的生命周期（加载）

**分析**：

如果完全遵循 sklearn 标准，这些功能应该直接集成到 Pipeline Steps 中：
- `GroupPreprocessingStep` 内部管理 `PreprocessingState`，`fit()` 时 fit，`transform()` 时 transform
- `PopulationClusteringStep` 内部管理聚类模型，`fit()` 时训练，`transform()` 时加载并 predict

**设计方案选择**：

**方案 A：完全移除 Mode 类（最符合 sklearn 标准，推荐）**

将逻辑直接集成到 Pipeline Steps 中：

```python
class GroupPreprocessingStep(BasePipelineStep):
    def __init__(self, methods: List[Dict], out_dir: str):
        self.preprocessing_state = PreprocessingState()  # 直接管理
        self.methods = methods
        self.out_dir = out_dir
        self.fitted_ = False
    
    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        """Fit preprocessing state."""
        self.preprocessing_state.fit(X, self.methods)
        self.fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted state."""
        if not self.fitted_:
            # Try to load from saved pipeline
            raise ValueError("Must fit before transform, or load saved pipeline")
        return self.preprocessing_state.transform(X)

class PopulationClusteringStep(BasePipelineStep):
    def __init__(self, clustering_manager: ClusteringManager, config, out_dir: str):
        self.clustering_manager = clustering_manager
        self.config = config
        self.out_dir = out_dir
        self.clustering_model = None
        self.fitted_ = False
    
    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        """Train clustering model."""
        # 直接在这里实现训练逻辑（原TrainingMode.cluster_habitats的逻辑）
        optimal_n, scores = self._find_optimal_clusters(X)
        self.clustering_model = get_clustering_algorithm(...)
        self.clustering_model.n_clusters = optimal_n
        self.clustering_model.fit(X)
        self.fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted model."""
        if not self.fitted_:
            # Try to load from saved pipeline
            raise ValueError("Must fit before transform, or load saved pipeline")
        labels = self.clustering_model.predict(X) + 1
        X[ResultColumns.HABITATS] = labels
        return X
```

**优点**：
- 完全符合 sklearn 标准
- 逻辑更清晰，每个 Step 自己管理状态
- 减少抽象层，代码更直接

**缺点**：
- 需要重构现有代码
- 模型保存/加载逻辑需要在 Steps 中实现

**方案 B：合并为一个 Mode 类（折中方案）**

合并 `TrainingMode` 和 `TestingMode` 为一个 `StateManager` 类，通过 `fitted_` 状态区分行为：

```python
class StateManager:
    """Manages state for group-level operations."""
    
    def __init__(self, config, logger, out_dir: str):
        self.config = config
        self.logger = logger
        self.out_dir = out_dir
        self.preprocessing_state = PreprocessingState()
        self.clustering_model = None
        self.fitted_ = False
    
    def fit_preprocessing(self, features: pd.DataFrame, methods: List[Dict]):
        """Fit preprocessing state."""
        self.preprocessing_state.fit(features, methods)
    
    def transform_preprocessing(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted state."""
        if not self.fitted_:
            self._load_state()  # Auto-load if not fitted
        return self.preprocessing_state.transform(features)
    
    def fit_clustering(self, features: pd.DataFrame, algorithm):
        """Train clustering model."""
        # Training logic
        ...
        self.clustering_model = trained_model
        self.fitted_ = True
    
    def transform_clustering(self, features: pd.DataFrame):
        """Apply fitted model."""
        if not self.fitted_:
            self._load_state()  # Auto-load if not fitted
        return self.clustering_model.predict(features)
    
    def save_state(self, model_name: str):
        """Save all state."""
        bundle = {
            'preprocessing_state': self.preprocessing_state,
            'clustering_model': self.clustering_model
        }
        # Save bundle
    
    def _load_state(self):
        """Load saved state."""
        # Load bundle
```

**优点**：
- 保留状态管理的封装
- 减少代码重复
- 向后兼容性更好

**缺点**：
- 仍然有一个额外的抽象层
- 不完全符合 sklearn 的"每个 Step 自己管理状态"的理念

**方案 C：保留但简化（向后兼容方案）**

保留 `TrainingMode` 和 `TestingMode`，但作为 Pipeline Steps 的内部辅助类，不暴露给用户：

```python
class GroupPreprocessingStep(BasePipelineStep):
    def __init__(self, methods: List[Dict], config, out_dir: str):
        # 内部创建 mode handler，但不暴露
        self._state_manager = self._create_state_manager(config, out_dir)
        self.methods = methods
        self.fitted_ = False
    
    def _create_state_manager(self, config, out_dir):
        """Create state manager based on fitted status."""
        # 根据是否已加载pipeline来决定创建TrainingMode还是TestingMode
        # 或者统一使用一个StateManager
        ...
    
    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        """Fit preprocessing state."""
        self._state_manager.preprocessing_state.fit(X, self.methods)
        self.fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted state."""
        if not self.fitted_:
            self._state_manager._load_state()  # Auto-load
        return self._state_manager.preprocessing_state.transform(X)
```

**推荐方案**：**方案 A（完全移除 Mode 类）**

**理由**：
1. **完全符合 sklearn 标准**：每个 Step 自己管理状态，不需要额外的 Mode 类
2. **逻辑更清晰**：状态管理逻辑直接在使用它的 Step 中，更容易理解
3. **减少抽象层**：减少不必要的抽象，代码更直接
4. **更好的封装**：每个 Step 是自包含的，可以独立序列化和反序列化

**实施建议**：
1. 将 `PreprocessingState` 的管理直接集成到 `GroupPreprocessingStep`
2. 将聚类模型的训练和加载逻辑直接集成到 `PopulationClusteringStep`
3. 模型保存/加载逻辑在 Pipeline 层面统一处理（通过 `save()` 和 `load()` 方法）
4. 移除 `TrainingMode` 和 `TestingMode`，集中在 Pipeline Steps 内部管理状态

### 8.4 与现有代码的集成

- **Strategy 层**：`TwoStepStrategy`、`OneStepStrategy`、`DirectPoolingStrategy` 内部使用对应的 Pipeline
- **Mode 层**：
  - Pipeline Steps 内部直接管理状态，不需要 Mode 类
- **Manager 层**：`FeatureManager`、`ClusteringManager` 等被 Pipeline Steps 调用
- **向后兼容**：现有的 Strategy API 保持不变，Pipeline 作为内部实现

### 8.4 实施优先级

1. **Phase 1（核心）**：实现基础框架和关键步骤
   - `BasePipelineStep` 和 `HabitatPipeline`
   - Step 6（Group Preprocessing）和 Step 7（Population Clustering）
   - Pipeline Builder 框架
   
2. **Phase 2（完整）**：实现所有步骤
   - Step 1-5 的实现
   - 策略特定的 Pipeline 构建函数
   - 支持二步法、一步法、pooling法
   
3. **Phase 3（集成）**：集成到 Strategy
   - 重构 `TwoStepStrategy` 使用 Pipeline
   - 重构 `OneStepStrategy` 使用 Pipeline
   - 重构 `DirectPoolingStrategy` 使用 Pipeline
   
4. **Phase 4（优化）**：测试和文档
   - 单元测试和集成测试
   - 更新文档和示例
   - 性能优化

## 九、注意事项

1. **数据格式**：Pipeline 的输入输出格式需要统一（见 7.3）
2. **状态管理**：
   - 个体水平操作：**不需要保存状态**（见 7.4.1）
   - 群体水平操作：**必须保存状态**（见 7.4.2）
   - 确保所有状态都在 fit() 时保存，transform() 时加载
3. **并行处理**：某些步骤（如特征提取）可能需要并行处理，需要在 Step 内部处理
4. **错误处理**：需要完善的错误处理和日志记录
5. **向后兼容**：保持与现有 API 的兼容性
6. **Supervoxel 特征提取**：需要明确区分均值特征和高级特征提取（见 7.2）
7. **Pipeline 序列化**：确保所有 Steps 都可以被 joblib 正确序列化和反序列化
8. **策略特定设计**：不同策略使用不同的 Pipeline 步骤组合（见 8.2）
9. **Step 可配置性**：通用 Steps（如 `IndividualClusteringStep`）需要支持不同策略的参数配置
10. **无 mode 参数**：Pipeline 层面不区分 training/testing mode，通过 `fit()` 和 `transform()` 方法区分：
    - `fit()`：学习参数并保存状态
    - `transform()`：应用保存的状态
    - 用户只需调用相应的方法，不需要指定 mode
11. **Mode 类移除（重要）**：`TrainingMode` 和 `TestingMode` 已移除，状态管理逻辑直接集成到 Pipeline Steps 中：
    - `GroupPreprocessingStep` 内部直接管理 `PreprocessingState`
    - `PopulationClusteringStep` 内部直接管理聚类模型
    - 完全符合 sklearn 的"每个 Step 自己管理状态"的理念
    - 不存在过渡兼容层，Pipeline 直接负责状态
12. **Step 4 的条件执行（性能优化）**：Step 4（Supervoxel Feature Extraction）根据配置条件执行：
    - 判断逻辑：检查 `config.FeatureConstruction.supervoxel_level.method` 是否只包含 `'mean_voxel_features'`
    - 如果只使用均值特征：**跳过 Step 4**，直接进入 Step 5（节省计算时间）
    - 如果使用高级特征（如 `supervoxel_radiomics`）：**执行 Step 4**
    - Step 5 需要能够处理两种情况：有/无 Step 4 的输出
    - Pipeline Builder 在构建 Pipeline 时根据配置条件性地添加 Step 4

## 十、TODO LIST

### 10.1 Phase 1: 基础框架实现（核心）

#### 1.1 创建基础类结构
- [x] 创建 `habit/core/habitat_analysis/pipelines/` 目录
- [x] 创建 `pipelines/__init__.py`，导出主要类
- [x] 实现 `BasePipelineStep` 抽象基类
  - [x] 定义 `fit()`, `transform()`, `fit_transform()` 方法
  - [x] 实现 `fitted_` 状态管理
  - [x] 添加类型提示和文档字符串
- [x] 实现 `HabitatPipeline` 主类
  - [x] 实现 `__init__()` 方法（支持从文件加载）
  - [x] 实现 `fit()` 方法（顺序调用各步骤的 fit）
  - [x] 实现 `transform()` 方法（顺序调用各步骤的 transform）
  - [x] 实现 `fit_transform()` 方法
  - [x] 实现 `save()` 方法（使用 joblib）
  - [x] 实现 `load()` 类方法（使用 joblib）
  - [x] 添加错误处理和验证逻辑
  - [ ] 添加日志记录（可选，后续优化）

#### 1.2 实现状态管理步骤（关键）
- [x] 实现 `GroupPreprocessingStep`
  - [x] 内部管理 `PreprocessingState` 实例
  - [x] 实现 `fit()`：调用 `preprocessing_state.fit()`
  - [x] 实现 `transform()`：调用 `preprocessing_state.transform()`
  - [x] 确保可以序列化（joblib）
- [x] 实现 `PopulationClusteringStep`
  - [x] 内部管理聚类模型实例
  - [x] 实现 `fit()`：
    - [x] 实现最优 cluster 数查找逻辑（从 TrainingMode 迁移）
    - [x] 训练聚类模型
    - [x] 保存模型和最优 cluster 数
  - [x] 实现 `transform()`：
    - [x] 应用模型进行预测（通过 fitted_ 状态管理）
    - [x] 添加错误处理
  - [x] 确保可以序列化（joblib）

#### 1.3 Pipeline Builder 框架
- [x] 创建 `pipeline_builder.py`
- [x] 实现 `build_habitat_pipeline()` 主函数
  - [x] 支持 `load_from` 参数（从文件加载）
  - [x] 根据策略类型调用对应的构建函数
- [x] 实现策略特定的构建函数框架
  - [x] `_build_two_step_pipeline()` 框架（仅实现状态管理步骤）
  - [x] `_build_one_step_pipeline()` 框架（占位符）
  - [x] `_build_pooling_pipeline()` 框架（仅实现状态管理步骤）

#### 1.4 单元测试（Phase 1）
- [x] 测试 `BasePipelineStep` 接口
- [x] 测试 `HabitatPipeline` 的基本功能
- [x] 测试 `GroupPreprocessingStep` 的 fit/transform
- [ ] 测试 `PopulationClusteringStep` 的 fit/transform（需要完整的 ClusteringManager 和 Config）
- [ ] 测试 Pipeline 的 save/load 功能（需要完整测试环境）

---

### 10.2 Phase 2: 实现所有步骤（完整功能）

#### 2.1 实现无状态步骤
- [x] 实现 `VoxelFeatureExtractor`
  - [x] 调用 `FeatureManager.extract_voxel_features()`
  - [x] 处理输入格式（Dict[str, Dict]）
  - [x] 返回正确的输出格式
- [x] 实现 `SubjectPreprocessingStep`
  - [x] 调用 `FeatureManager.apply_preprocessing(level='subject')`
  - [x] 处理特征清理（inf, nan）
  - [x] 保持数据格式一致性
- [x] 实现 `IndividualClusteringStep`
  - [x] 支持 `target='supervoxel'` 和 `target='habitat'`
  - [x] 调用 `ClusteringManager.cluster_subject_voxels()`
  - [x] 保存 supervoxel map 文件
  - [x] 处理 one-step 策略的最优 cluster 数查找
- [x] 实现 `SupervoxelFeatureExtractionStep`（条件执行）
  - [x] 调用 `FeatureManager.extract_supervoxel_features()`
  - [x] 处理 supervoxel map 文件发现
  - [x] 返回格式：将 supervoxel_features 添加到 X 中（通过 X 传递）
- [x] 实现 `SupervoxelAggregationStep`
  - [x] 调用 `FeatureManager.calculate_supervoxel_means()`
  - [x] 支持从 X 中提取 Step 4 的输出（`supervoxel_features` 键）
  - [x] 合并均值特征和高级特征（如果提供）
  - [x] 返回合并后的 DataFrame
- [x] 实现 `ConcatenateVoxelsStep`（用于 pooling 策略）
  - [x] 合并所有 subjects 的 voxel 特征
  - [x] 添加 Subject 列

#### 2.2 完善 Pipeline Builder
- [x] 完善 `_build_two_step_pipeline()`
  - [x] 实现 Step 4 的条件添加逻辑
  - [x] 判断 `config.FeatureConstruction.supervoxel_level.method`
  - [x] 如果只使用 `mean_voxel_features`，跳过 Step 4
  - [x] 正确连接所有步骤（Step 1-7）
- [x] 完善 `_build_one_step_pipeline()`
  - [x] 构建一步法 Pipeline（无群体级步骤）
  - [x] 支持 `target='habitat'` 的 IndividualClusteringStep
  - [x] 支持 `find_optimal=True` 的最优 cluster 数查找
- [x] 完善 `_build_pooling_pipeline()`
  - [x] 实现 `ConcatenateVoxelsStep`
  - [x] 构建 pooling 策略 Pipeline（Step 1-5）

#### 2.3 数据格式处理
- [x] 统一所有步骤的输入输出格式
  - [x] Step 1-3: Dict[str, Dict] 格式（按 subject 组织）
  - [x] Step 4: 将 supervoxel_features 添加到 X 中（通过 X 传递）
  - [x] Step 5: Dict[str, Dict] → pd.DataFrame（合并所有 subjects）
  - [x] Step 6-7: pd.DataFrame 格式（群体级操作）
- [x] 实现数据格式转换辅助函数（如果需要）
  - [x] `calculate_supervoxel_means()` 返回包含 Subject, Supervoxel 列的 DataFrame
  - [x] `ConcatenateVoxelsStep` 添加 Subject 列
  - [x] `SupervoxelAggregationStep` 合并并转换格式
- [x] 确保 Dict[str, Dict] 到 pd.DataFrame 的转换正确
  - [x] Step 5 正确执行转换（通过 `calculate_supervoxel_means` 和 `pd.concat`）
  - [x] Step 3 (ConcatenateVoxelsStep) 也执行转换（用于 pooling 策略）
- [x] 处理 metadata 列（Subject, Supervoxel, Habitats）
  - [x] 使用 `ResultColumns.SUBJECT`, `ResultColumns.SUPERVOXEL`, `ResultColumns.HABITATS`
  - [x] `calculate_supervoxel_means()` 自动添加 Subject 和 Supervoxel 列
  - [x] `PopulationClusteringStep` 添加 Habitats 列
  - [x] `ConcatenateVoxelsStep` 添加 Subject 列

#### 2.4 单元测试（Phase 2）
- [x] 测试所有步骤的 fit/transform
  - [x] 创建 `test_pipeline_phase2.py` 测试文件
  - [x] 测试 VoxelFeatureExtractor, SubjectPreprocessingStep, ConcatenateVoxelsStep
  - [x] 测试 SupervoxelAggregationStep（有/无 Step 4 输出）
  - [ ] 完整集成测试（需要完整的 Manager 和 Config 实例）
- [x] 测试 Step 4 的条件执行逻辑
  - [x] 在 Pipeline Builder 中实现条件判断
  - [ ] 单元测试（需要完整配置）
- [x] 测试 Step 5 处理有/无 Step 4 输出的情况
  - [x] 实现逻辑：从 X 中检查 'supervoxel_features' 键
  - [x] 创建单元测试验证两种情况
- [ ] 测试不同策略的 Pipeline 构建
  - [ ] 需要完整的配置和 Manager 实例
- [x] 测试数据格式转换
  - [x] 创建测试验证 Dict → DataFrame 转换
  - [x] 验证 metadata 列的正确添加

---

### 10.3 Phase 3: 集成到现有代码

#### 3.1 重构 Strategy 类
- [x] 重构 `TwoStepStrategy`
  - [x] `run()` 只表达 sklearn 语义：训练走 `fit_transform`，推理走 `transform`
  - [x] 移除 `mode` 分支，仅由调用路径（训练/推理）决定流程
  - [x] 统一 pipeline 保存/加载策略（显式参数控制）
  - [x] 清理遗留分支与重复逻辑
- [x] 重构 `OneStepStrategy`
  - [x] `run()` 改为 sklearn 风格（fit/transform）
  - [x] 移除 `mode` 与旧流程分支
- [x] 重构 `DirectPoolingStrategy`
  - [x] `run()` 改为 sklearn 风格（fit/transform）
  - [x] 移除 `mode` 与旧流程分支

#### 3.2 更新 HabitatAnalysis 类
- [x] 移除 `mode_handler` 与 `create_mode` 依赖
- [ ] 对外暴露明确的训练/推理入口（fit/transform 风格）
- [x] 统一 Strategy 构建与 Pipeline 交互方式

#### 3.3 旧流程清理
- [x] 删除 `TrainingMode` / `TestingMode` 相关代码与依赖
- [x] 删除 `mode` 字段的读取与分支逻辑
- [x] 移除旧入口或旧配置路径
- [x] 更新对应调用方，确保无遗留依赖

#### 3.4 集成测试
- [ ] 测试完整的训练流程（fit）
- [ ] 测试完整的测试流程（transform）
- [ ] 测试 Pipeline 的保存和加载
- [ ] 测试不同策略的完整流程
- [ ] 测试与现有代码的集成

---

### 10.4 Phase 4: 测试和优化

#### 4.1 单元测试完善
- [ ] 提高测试覆盖率（目标 > 80%）
- [ ] 测试边界情况
- [ ] 测试错误处理
- [ ] 测试序列化/反序列化

#### 4.2 集成测试
- [ ] 端到端测试（从配置到结果）
- [ ] 测试不同配置组合
- [ ] 测试大数据集性能
- [ ] 测试并行处理（如果实现）

#### 4.3 性能优化
- [ ] 分析性能瓶颈
- [ ] 优化特征提取步骤（并行处理）
- [ ] 优化数据格式转换
- [ ] 优化序列化/反序列化性能

#### 4.4 文档更新
- [ ] 更新 API 文档
- [ ] 创建使用示例和教程
- [ ] 更新配置文档
- [ ] 添加迁移指南（从旧代码到 Pipeline）
- [ ] 更新 README

---

### 10.5 Phase 5: 代码清理和重构（可选）

#### 5.1 代码清理
- [ ] 移除未使用的代码
- [ ] 统一代码风格
- [ ] 添加类型提示
- [ ] 完善文档字符串

#### 5.2 重构优化
- [ ] 提取公共逻辑到工具函数
- [ ] 优化类设计
- [ ] 改进错误消息
- [ ] 改进日志记录

#### 5.3 最终验证
- [ ] 代码审查
- [ ] 性能测试
- [ ] 用户验收测试
- [ ] 发布准备

---

### 10.6 实施注意事项

#### 6.1 优先级说明
- **Phase 1** 是核心，必须首先完成
- **Phase 2** 实现完整功能，可以逐步完成各个步骤
- **Phase 3** 集成时严格遵循 sklearn 语义（fit/transform）
- **Phase 4** 确保质量和性能
- **Phase 5** 是可选的优化阶段

#### 6.2 关键检查点
- [ ] Phase 1 完成后：基础框架可用，可以保存和加载 Pipeline
- [ ] Phase 2 完成后：所有步骤实现，可以构建完整 Pipeline
- [ ] Phase 3 完成后：旧流程移除，入口语义清晰（fit/transform）
- [ ] Phase 4 完成后：测试通过，性能满足要求

#### 6.3 依赖关系
- Phase 2 依赖 Phase 1
- Phase 3 依赖 Phase 1 和 Phase 2
- Phase 4 依赖 Phase 1、2、3
- Phase 5 依赖 Phase 1-4

#### 6.4 风险评估
- **技术风险**：序列化/反序列化可能遇到问题（特别是复杂对象）
- **行为变更风险**：旧入口/旧配置移除会影响现有调用路径
- **性能风险**：Pipeline 可能增加一些开销
- **缓解措施**：
  - 充分的单元测试和集成测试
  - 明确迁移窗口与变更清单
  - 性能测试和优化

---

### 10.7 当前状态跟踪

**最后更新日期**：2024-12-XX

**当前阶段**：Phase 2 所有步骤实现完成 ✅

**已完成**：
1. ✅ Phase 1：基础框架实现
   - 创建 `pipelines/` 目录结构
   - 实现 `BasePipelineStep` 和 `HabitatPipeline`
   - 实现 `GroupPreprocessingStep` 和 `PopulationClusteringStep`
   - 创建 `pipeline_builder.py` 框架
2. ✅ Phase 2：所有步骤实现
   - 实现 `VoxelFeatureExtractor`
   - 实现 `SubjectPreprocessingStep`
   - 实现 `IndividualClusteringStep`（支持 supervoxel 和 habitat 两种目标）
   - 实现 `SupervoxelFeatureExtractionStep`（条件执行）
   - 实现 `SupervoxelAggregationStep`（支持有/无 Step 4 输出）
   - 实现 `ConcatenateVoxelsStep`（用于 pooling 策略）
   - 完善 Pipeline Builder 支持完整的二步法、一步法和 pooling 流程
3. ✅ Phase 2.3：数据格式处理
   - 统一所有步骤的输入输出格式
   - 实现数据格式转换（Dict → DataFrame）
   - 处理 metadata 列（Subject, Supervoxel, Habitats）
   - 处理列名标准化（SupervoxelID → Supervoxel）
4. ✅ Phase 2.4：单元测试框架
   - 创建 `test_pipeline_phase2.py` 测试文件
   - 实现基础测试用例
   - 测试数据格式转换逻辑

**下一步行动**：
1. 开始 Phase 3：集成到现有代码（重构 Strategy 类使用 Pipeline）
2. 完善单元测试的完整集成测试（需要完整的 Manager 和 Config 实例）
3. 测试不同策略的 Pipeline 构建和数据流

**阻塞问题**：无

**备注**：
- Phase 1 核心框架已完成，可以保存和加载 Pipeline
- 当前 Pipeline Builder 只实现了状态管理步骤（Step 6 和 Step 7）
- 需要在 Phase 2 中实现前面的步骤（Step 1-5）才能构建完整的 Pipeline

**备注**：
- 设计文档已完成，所有关键设计决策已明确
- 建议按照 Phase 顺序逐步实施
- 每个 Phase 完成后进行代码审查和测试
