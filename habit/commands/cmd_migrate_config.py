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
"""``habit migrate-config``: upgrade a v0 YAML config to the v1 layout.

Migration is strictly OPTIONAL -- v0 configs keep running unchanged -- but
the v1 layout separates spec / data / policy / output so the same analysis
is easier to review, diff, and reuse across machines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from habit.common import (
    echo_error,
    echo_success,
    exit_with_error,
    format_config_load_error,
    require_config_path,
)


def run_migrate_config(
    config_path: str,
    output_path: Optional[str] = None,
    dry_run: bool = False,
    workflow: Optional[str] = None,
) -> None:
    """
    Migrate one v0 YAML config into the v1 document layout.

    Args:
        config_path: Source v0 YAML file (required).
        output_path: Destination v1 file; defaults to ``<name>.v1.yaml``
            next to the source.
        dry_run: When True, print the unified diff without writing a file.
        workflow: Workflow alias override; guessed from the path when
            omitted, exactly like ``habit check-config``.
    """
    from habit.spec.legacy import migrate_yaml

    path = Path(require_config_path(config_path))
    if not path.is_file():
        exit_with_error(f"Error: 找不到配置文件 / Config not found: {path}")
    if output_path is not None and Path(output_path).resolve() == path.resolve():
        exit_with_error(
            "Error: 输出路径不能与源文件相同（不会原地覆盖）/\n"
            "Output path must differ from the source (in-place overwrite "
            "is refused)."
        )

    try:
        report = migrate_yaml(
            path, output_path, dry_run=dry_run, workflow=workflow
        )
    except Exception as exc:  # noqa: BLE001
        echo_error(format_config_load_error(exc, str(path)))
        exit_with_error(
            f"迁移失败 / Migration failed: {exc.__class__.__name__}."
        )

    if report.warnings:
        click.echo("迁移提示 / Migration notes:")
        for note in report.warnings:
            click.echo(f"  - {note}")
        click.echo("")

    if dry_run:
        click.echo(report.diff)
        echo_success(
            "预演完成，未写文件 / Dry run only; no file written. "
            "去掉 --dry-run 以写出 v1 配置。"
        )
        return

    click.echo(f"v0 配置 / source : {report.source}")
    click.echo(f"v1 配置 / output : {report.destination}")
    echo_success(
        f"迁移完成 / Migration done (workflow={report.workflow}). "
        "可用 habit check-config -c <v1 文件> 校验。"
    )
