"""工具 — Dice 重叠度

Run: python tests/utils/util_dice_overlap.py
"""

import os
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.argv = ["habit", "dice", "--input1", ".cursor/test/resample_02", "--input2", ".cursor/test/resample_02", "--output", "demo_data/results/dice_resample02_demo.csv", "--mask-keyword", "masks", *sys.argv[1:]]
    from habit.cli import cli

    cli()


if __name__ == "__main__":
    main()
