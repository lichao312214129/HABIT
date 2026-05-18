"""One demo YAML only: ``get-habitat -c .../config_habitat_two_step.yaml`` with cwd at repo root.

Run::

    python tests/habitat/manual_cli_e2e_two_step.py
    python tests/habitat/manual_cli_e2e_two_step.py --debug

Anything after the script name (``sys.argv[1:]`` when ``len(sys.argv) > 1``) is appended to the CLI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from click.testing import CliRunner

from habit.cli import cli

CFG = r'F:\work\habit_project\.cursor\test\config_habitat_two_step.yaml'

if __name__ == "__main__":
    argv = ["get-habitat", "-c", CFG, *sys.argv[1:]]
    r = CliRunner().invoke(cli, argv)
    print(r.output, end="")
    sys.exit(r.exit_code)
