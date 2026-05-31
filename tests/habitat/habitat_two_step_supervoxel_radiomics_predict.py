"""二步法生境 — 超体素纹理（影像组学）特征 — 预测

Config: config/habitat/config_habitat_two_step_supervoxel_radiomics_predict.yaml
Run:    python tests/habitat/habitat_two_step_supervoxel_radiomics_predict.py

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
    sys.argv = ["habit", "get-habitat", "-c", "config/habitat/config_habitat_two_step_supervoxel_radiomics_predict.yaml", *sys.argv[1:]]
    from habit.cli import cli

    cli()


if __name__ == "__main__":
    main()
