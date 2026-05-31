"""临床特征机器学习 — 训练（单表）

Config: config/machine_learning/config_machine_learning_clinical.yaml
Run:    python tests/machine_learning/ml_train_clinical.py

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
    sys.argv = ["habit", "model", "-c", "config/machine_learning/config_machine_learning_clinical.yaml", "-m", "train", *sys.argv[1:]]
    from habit.cli import cli

    cli()


if __name__ == "__main__":
    main()
