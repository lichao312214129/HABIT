# Supervoxel Feature Pipeline Refactoring

## 📋 Overview

This document describes the refactoring of supervoxel feature handling in the two-step habitat analysis pipeline.

## 🎯 Goals

1. **Clear separation of concerns**: Split feature calculation and feature selection into distinct steps
2. **Mutually exclusive feature types**: Users can choose EITHER mean voxel features OR advanced supervoxel features, not both
3. **Better naming**: Use descriptive names that clearly indicate what each step does

## 🔄 Changes

### Old Architecture (Deprecated)

```
Step 3: Individual Clustering (voxel → supervoxel)
Step 4: SupervoxelFeatureExtractionStep (optional, advanced features)
Step 5: SupervoxelAggregationStep (calculate means + merge advanced features)
Step 6: CombineSupervoxelsStep
Step 7: Group Preprocessing
Step 8: Population Clustering
```

**Problems**:
- Step 5 had two responsibilities (calculate means AND merge features)
- Name "SupervoxelAggregationStep" was too vague
- Mixed mean features + advanced features, unclear which is used for clustering

### New Architecture

```
Step 3: Individual Clustering (voxel → supervoxel)
Step 4: CalculateMeanVoxelFeaturesStep (ALWAYS executed)
Step 5: SupervoxelFeatureExtractionStep (OPTIONAL, advanced features)
Step 6: MergeSupervoxelFeaturesStep (SELECT one feature type)
Step 7: CombineSupervoxelsStep
Step 8: Group Preprocessing
Step 9: Population Clustering
```

**Improvements**:
- Clear single responsibility for each step
- Descriptive names: `CalculateMeanVoxelFeaturesStep`, `MergeSupervoxelFeaturesStep`
- Explicit feature selection logic in Step 6

## 📊 New Steps

### 1. CalculateMeanVoxelFeaturesStep

**File**: `calculate_mean_voxel_features.py`

**Purpose**: Calculate mean of voxel features within each supervoxel

**Input**:
```python
{
    'subject_id': {
        'features': DataFrame (voxel-level),
        'raw': DataFrame,
        'supervoxel_labels': ndarray
    }
}
```

**Output**:
```python
{
    'subject_id': {
        ... (original fields),
        'mean_voxel_features': DataFrame  # NEW
    }
}
```

**Always executed**: Yes

### 2. MergeSupervoxelFeaturesStep

**File**: `merge_supervoxel_features.py`

**Purpose**: Select which supervoxel features to use for clustering

**Feature Selection Logic**:

```python
if config.supervoxel_level.method contains 'mean_voxel_features':
    # Mode 1: Use mean voxel features
    supervoxel_df = data['mean_voxel_features']
else:
    # Mode 2: Use advanced features
    supervoxel_df = data['supervoxel_features']
```

**Input**:
```python
{
    'subject_id': {
        'mean_voxel_features': DataFrame (always present),
        'supervoxel_features': DataFrame (optional)
    }
}
```

**Output**:
```python
{
    'subject_id': {
        'supervoxel_df': DataFrame  # Selected features
    }
}
```

**Mutually Exclusive**: Yes - uses EITHER mean OR advanced, never both

## ⚙️ Configuration

### Mode 1: Use Mean Voxel Features (Default)

```yaml
FeatureConstruction:
  voxel_level:
    method: concat(raw(delay2), raw(delay3))
    
  supervoxel_level:
    method: mean_voxel_features()  # Triggers Mode 1
    params: {}
```

**Pipeline**:
```
Step 4: CalculateMeanVoxelFeaturesStep → computes means
Step 5: SupervoxelFeatureExtractionStep → SKIPPED
Step 6: MergeSupervoxelFeaturesStep → selects 'mean_voxel_features'
```

### Mode 2: Use Advanced Supervoxel Features

```yaml
FeatureConstruction:
  voxel_level:
    method: concat(raw(delay2), raw(delay3))
    
  supervoxel_level:
    method: supervoxel_radiomics()  # Triggers Mode 2
    params:
      params_file: ./radiomics_params.yaml
```

**Pipeline**:
```
Step 4: CalculateMeanVoxelFeaturesStep → computes means (but not used)
Step 5: SupervoxelFeatureExtractionStep → extracts shape/texture features
Step 6: MergeSupervoxelFeaturesStep → selects 'supervoxel_features'
```

## 🔄 Backward Compatibility

**SupervoxelAggregationStep** is kept but marked as **DEPRECATED**:

```python
class SupervoxelAggregationStep(IndividualLevelStep):
    """
    DEPRECATED: Use CalculateMeanVoxelFeaturesStep + MergeSupervoxelFeaturesStep instead.
    """
```

A `DeprecationWarning` is emitted when this class is instantiated.

## 🎨 Feature Selection Matrix

| Config Method | Step 4 (Mean) | Step 5 (Advanced) | Step 6 Selection | Clustering Uses |
|---------------|---------------|-------------------|------------------|-----------------|
| `mean_voxel_features()` | ✅ Executed | ❌ Skipped | Mean features | **Mean only** |
| `supervoxel_radiomics()` | ✅ Executed | ✅ Executed | Advanced features | **Advanced only** |

## 📝 Migration Guide

### Old Code (Still Works)

```python
# No changes needed - old pipelines still work with deprecation warning
```

### New Code (Recommended)

The new steps are automatically used when you call `build_habitat_pipeline()`:

```python
from habit.core.habitat_analysis.pipelines import build_habitat_pipeline

# Just use the builder - it handles everything
pipeline = build_habitat_pipeline(config, feature_manager, clustering_manager)
```

## ✅ Benefits

1. **Clearer semantics**: Each step has a single, well-defined purpose
2. **Better naming**: `CalculateMeanVoxelFeaturesStep` vs vague "Aggregation"
3. **Explicit feature selection**: Users know exactly which features are used
4. **No feature mixing**: Prevents confusion about what's being clustered
5. **Easier to extend**: Adding new feature types is straightforward

## 🧪 Testing

Run the two-step tests to verify:

```bash
pytest tests/test_habitat_two_step_train.py -v
```

## 📌 Summary

This refactoring improves the clarity and maintainability of the two-step pipeline by:
- Splitting responsibilities into focused steps
- Using descriptive names
- Making feature selection explicit and mutually exclusive
- Maintaining backward compatibility
