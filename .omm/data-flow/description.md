# Data flow (selected paths)

This perspective highlights how **artefacts** chain domains without Python imports between `preprocessing`, `habitat_analysis`, and `machine_learning`.

**Preprocessing → Habitat**
- Input: raw or semi-structured image trees / YAML path lists.
- Output: standardized images under `out_dir` (layout depends on `PreprocessingConfig`); habitat reads `data_dir` pointing at preprocessed images + masks.

**Habitat train**
- `FeatureService` discovers per-subject images/masks; pipeline steps produce supervoxel/habitat features and cluster labels.
- Outputs: per-subject `*_habitats.nrrd`, `habitats.csv`, validation plots (optional), serialized `habitat_pipeline.pkl`.

**Habitat predict**
- Loads `habitat_pipeline.pkl`, re-injects runtime services, runs `transform` only; writes maps/CSV for new subjects.

**Downstream modelling**
- `habit extract` / radiomics workflows consume habitat maps + raw images paths from config CSVs/YAML—not direct imports from `habitat_analysis` internals.
