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

## Phase 6: Model Comparison Refactoring (Current)
- [ ] Decouple `ModelComparison` from monolithic implementation
- [ ] Integrate `PredictionContainer` for data consistency
- [ ] Implement `ThresholdManager` for cross-dataset threshold transfer
- [ ] Move reporting logic to `ReportExporter`
- [ ] Utilize `PlotManager` for unified visualization

## Phase 7: Global Robustness & Testing
- [ ] Extend Pydantic validation to Preprocessing module
- [ ] Extend Pydantic validation to Habitat Analysis module
- [ ] Implement full-pipeline automated integration tests (Model -> CV -> Compare)
