# habitat_analysis deep module (`HabitatAnalysis`)

`habit/core/habitat_analysis/habitat_analysis.py` is the single orchestrator for habitat clustering pipelines.

Responsibilities:
- Builds a sklearn-style `HabitatPipeline` whose step list is selected by **`config.HabitatsSegmention.clustering_mode`** via an internal **`_PIPELINE_RECIPES`** map (modes: `two_step`, `one_step`, `direct_pooling`). The old separate `strategies/` package is removed in V1.
- **Training**: `fit()` builds pipeline, persists `habitat_pipeline.pkl`, emits NRRD maps + CSV via `HabitatImageWriter`.
- **Inference**: `predict(pipeline_path=...)` loads pipeline, reconciles collaborators through **`_PIPELINE_SERVICE_ATTRS`** (`feature_service`, `clustering_service`, `habitat_image_writer`).
- **`run()`** remains a backwards-compatible dispatcher that reads `run_mode` / `pipeline_path` from config; CLI uses `fit` / `predict` explicitly (`habit/cli_commands/commands/cmd_habitat.py`).

`HabitatConfigurator.create_habitat_analysis()` wires the three services and logger.
