"""预处理 — 仅重采样

Config: config/preprocessing/config_preprocessing_resample_only.yaml
Run:    python tests/preprocessing/preprocess_resample_only.py

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
    sys.argv = ["habit", "preprocess", "-c", "config/preprocessing/config_preprocessing_resample_only.yaml", *sys.argv[1:]]
    from habit.cli import cli

    cli()


if __name__ == "__main__":
    main()
