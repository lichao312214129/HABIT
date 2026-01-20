# HABIT Machine Learning Module Refactoring TODO

## Phase 1: Metric System Decoupling (Current)
- [ ] Implement `MetricRegistry` in `metrics.py`
- [ ] Refactor individual metrics into registered functions (Accuracy, Sensitivity, F1, etc.)
- [ ] Update `calculate_metrics` to dynamically invoke registered metrics
- [ ] Standardize display names for reports and tables

## Phase 2: Feature Selector Registry
- [ ] Implement `@register_selector` decorator (already partially exists, but needs standardization)
- [ ] Decouple selector parameters from the main execution loop
- [ ] Support "before_z_score" and "after_z_score" logic more cleanly through metadata

## Phase 3: Pipeline Builder
- [ ] Create `PipelineBuilder` class to centralize `sklearn.Pipeline` construction
- [ ] Ensure consistent preprocessing and selection order across all workflows

## Phase 4: Configuration & Validation
- [x] Implement Pydantic models for ML configuration
- [x] Add fail-fast validation for YAML parameters
- [ ] Extend Pydantic validation to Preprocessing module
- [ ] Extend Pydantic validation to Habitat Analysis module
- [ ] Create a `CommonConfig` base schema for shared parameters (paths, logging)

## Phase 6: Model Comparison Refactoring (Completed)
- [x] Decouple `ModelComparison` from monolithic implementation
- [x] Integrate `PredictionContainer` for data consistency
- [x] Implement `ThresholdManager` for cross-dataset threshold transfer
- [x] Move reporting logic to `ReportExporter`
- [x] Utilize `PlotManager` for unified visualization

### Phase 6 Optimization (2024-01-20)
- **MetricsStore Class**: Introduced unified metrics storage manager
  - Eliminates 50+ instances of direct `all_metrics` dictionary manipulation
  - Provides clean `add_metrics()`, `add_threshold()`, `ensure_group()` methods
  - Reduces nested dictionary access code throughout the codebase
- **Code Reduction**: comparison_workflow.py reduced from ~966 lines to ~772 lines
- **Simplified Methods**:
  - `_calculate_all_basic_metrics`: 27 lines → 16 lines
  - `_calculate_basic_metrics`: 52 lines → 19 lines
  - `_calculate_youden_metrics`: 54 lines → 26 lines
  - `_calculate_target_metrics`: 68 lines → 36 lines
  - `_calculate_target_metrics_by_split`: 100 lines → 50 lines
- **Benefits**:
  - Single source of truth for metrics structure
  - Easier to maintain and extend
  - Better separation of concerns

## Phase 7: Global Robustness & Testing
- [ ] Extend Pydantic validation to Preprocessing module
- [ ] Extend Pydantic validation to Habitat Analysis module
- [ ] Implement full-pipeline automated integration tests (Model -> CV -> Compare)
