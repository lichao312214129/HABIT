"""Two-step K-Means + silhouette — training smoke runner."""

import os
import sys
from pathlib import Path


def main() -> None:
    """Invoke habit CLI from repository root (Windows spawn-safe)."""
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.argv = [
        "habit",
        "get-habitat",
        "-c",
        "config/habitat/config_habitat_two_step_kmeans_silhouette.yaml",
        *sys.argv[1:],
    ]
    from habit.cli import cli

    cli()


if __name__ == "__main__":
    main()
