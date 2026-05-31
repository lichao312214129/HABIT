"""全流程 — 预处理 → 生境 → 特征 → 建模 → 对比

Run: python tests/integration/workflow_preprocess_to_compare.py

Each step uses its own config under config/ — edit paths in those YAML files.
"""

import os
import sys
from pathlib import Path

STEPS = [['preprocess', '-c', 'config/preprocessing/config_preprocessing_demo.yaml'], ['get-habitat', '-c', 'config/habitat/config_habitat_two_step.yaml'], ['extract', '-c', 'config/feature_extraction/config_extract_features_demo.yaml'], ['model', '-c', 'config/machine_learning/config_machine_learning_clinical.yaml', '-m', 'train'], ['compare', '-c', 'config/model_comparison/config_model_comparison_demo.yaml']]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root))
    from habit.cli import cli

    extra = sys.argv[1:]
    for i, step in enumerate(STEPS, 1):
        print(f"\n--- step {i}/{len(STEPS)}: habit {' '.join(step)} ---\n")
        sys.argv = ["habit", *step, *extra]
        cli()


if __name__ == "__main__":
    main()
