# HABIT V1 — Package layout

This perspective summarizes how the **HABIT** Python toolkit is structured at the coarsest level.

- **Entry layer**: Click CLI in `habit/cli.py` delegates to `habit/cli_commands/commands/cmd_*.py`. The `habit` console script targets `habit.cli:cli` (see `pyproject.toml`). Lazy Python exports live in `habit/__init__.py` (`HabitatAnalysis`, `Modeling`, etc.).
- **Configurator layer**: Shared `BaseConfigurator` in `habit/core/common/configurators/base.py`. Domain factories are co-located with each domain: `HabitatConfigurator`, `MLConfigurator`, `PreprocessingConfigurator`.
- **Domain packages** (no cross-import of business code): `habit/core/preprocessing`, `habit/core/habitat_analysis`, `habit/core/machine_learning`. Domains communicate via **files** (NIfTI/NRRD, CSV, joblib `.pkl`).
- **Utilities**: `habit/utils` (I/O, logging, progress, parallel helpers).

See `docs/source/development/architecture.rst` for the authoritative developer narrative.
