"""特征提取 — 全部类型

Config: config/feature_extraction/config_extract_features_demo.yaml
Run:    python tests/feature_extraction/extract_features.py

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
    sys.argv = ["habit", "extract", "-c", "config/feature_extraction/config_extract_features_demo.yaml", *sys.argv[1:]]
    from habit.cli import cli

    cli()


if __name__ == "__main__":
    main()
