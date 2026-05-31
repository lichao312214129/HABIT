"""传统影像组学（独立流程）

Config: config/radiomics/config_traditional_radiomics.yaml
Run:    python tests/machine_learning/ml_radiomics_standalone.py

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
    sys.argv = ["habit", "radiomics", "-c", "config/radiomics/config_traditional_radiomics.yaml", *sys.argv[1:]]
    from habit.cli import cli

    cli()


if __name__ == "__main__":
    main()
