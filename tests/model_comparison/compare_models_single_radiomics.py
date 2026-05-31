"""单模型（影像组学）对比

Config: config/model_comparison/config_model_comparison_single_radiomics.yaml
Run:    python tests/model_comparison/compare_models_single_radiomics.py

Edit the YAML above (#%% path blocks) for your own data. Optional: pass --debug
"""

import os
import sys
from pathlib import Path


def main() -> None:
    """Invoke habit CLI from repository root (Windows spawn-safe)."""
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.argv = ["habit", "compare", "-c", "config/model_comparison/config_model_comparison_single_radiomics.yaml", *sys.argv[1:]]
    from habit.cli import cli

    cli()


if __name__ == "__main__":
    main()
