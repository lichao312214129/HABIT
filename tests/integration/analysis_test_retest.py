"""Test-Retest 分析

Config: config/auxiliary/config_test_retest.yaml
Run:    python tests/integration/analysis_test_retest.py

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
    sys.argv = ["habit", "retest", "-c", "config/auxiliary/config_test_retest.yaml", *sys.argv[1:]]
    from habit.cli import cli

    cli()


if __name__ == "__main__":
    main()
