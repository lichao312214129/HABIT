# HABIT v1 Recipe Library

Curated, testable YAML recipes for every major CLI workflow. Each recipe is registered in
`demo_data/results_config_test/matrix_manifest.yaml` and validated by
`demo_data/results_config_test/scripts/run_config_matrix.py`.

## Layout

| Directory | CLI command | Description |
|-----------|-------------|-------------|
| `habitat/` | `habit get-habitat` | Habitat train/predict variants not duplicated under `config/habitat/` |
| `machine_learning/` | `habit model` / `habit cv` | Split/resampling/feature-selection recipe overlays |
| `preprocessing/` | `habit preprocess` | (reserved) step-specific recipes |
| `feature_extraction/` | `habit extract` | (reserved) single-type extract recipes |
| `model_comparison/` | `habit compare` | (reserved) comparison recipes |
| `auxiliary/` | `habit icc`, `habit retest`, … | (reserved) auxiliary recipes |
| `radiomics/` | `habit radiomics` | (reserved) standalone radiomics recipes |

Most recipes still live under `config/` (historical templates). New gap-filling recipes
are added here with stable `recipe_id` names in the manifest.

## Run the full matrix

```bash
# Fast cases only (~20 min)
python demo_data/results_config_test/scripts/run_config_matrix.py

# Full coverage including habitat/preprocess (~2–4 h)
python demo_data/results_config_test/scripts/run_config_matrix.py --include-slow

# Regenerate Sphinx recipe catalog from latest results
python demo_data/results_config_test/scripts/generate_recipe_catalog_doc.py
```

## Documentation

See `docs/source/configuration/recipe_catalog.rst` (generated after each matrix run).
