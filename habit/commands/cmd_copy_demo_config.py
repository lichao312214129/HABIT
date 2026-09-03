# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""``habit copy-demo-config``: materialize bundled demo YAML into a work dir."""

from __future__ import annotations

from pathlib import Path

import click

from habit.common import echo_error, echo_success, exit_with_error


def run_copy_demo_config(dest: str, overwrite: bool = False) -> Path:
    """
    Copy packaged demo configs into ``<dest>/config/``.

    Args:
        dest: Work directory that should receive ``config/``.
        overwrite: When True, overwrite existing files under ``config/``.

    Returns:
        Path: Absolute path of the created ``config/`` directory.
    """
    from habit.utils.demo_config_utils import copy_demo_config

    try:
        config_dir: Path = copy_demo_config(
            dest,
            overwrite=overwrite,
            show_progress=True,
        )
    except FileExistsError as exc:
        exit_with_error(f"Error: {exc}")
    except FileNotFoundError as exc:
        exit_with_error(f"Error: {exc}")
    except OSError as exc:
        echo_error(f"Error copying demo config: {exc}")
        raise SystemExit(1) from exc

    echo_success(f"Demo config written to: {config_dir}")
    click.echo(
        "Next: habit fetch-demo --work-dir .   "
        "(downloads the imaging pack once, prints the folder tree), then "
        "habit get-habitat -c config/habitat/config_habitat_two_step.yaml"
    )
    return config_dir
