"""
Manual smoke: ``habit sort-dicom`` with demo DicomSortConfig YAML (heavy I/O).

Not collected by pytest. Run from anywhere::

    python tests/preprocessing/manual_cli_preprocess.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    """Run DICOM sort CLI against ``config/dicom_sort/config_sort_dicom.yaml``."""
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    config = root / "config" / "dicom_sort" / "config_sort_dicom.yaml"
    if not config.is_file():
        print(f"Config not found: {config}", file=sys.stderr)
        sys.exit(2)

    sys.argv = ["habit", "sort-dicom", "-c", str(config)]
    from habit.cli import cli

    cli()


if __name__ == "__main__":
    main()
