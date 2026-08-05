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
"""
Shared plumbing for the cloud coverage matrix.

Everything in this suite runs on the deterministic synthetic tree from
:mod:`tests.fixtures.synthetic_data`; the real ``demo_data`` dataset is
gitignored and unavailable in CI. Config templates live in ``configs/``
with ``@TOKEN@`` placeholders that :func:`render_config` substitutes with
absolute paths of the session-scoped synthetic tree, so every rendered
YAML works regardless of the pytest working directory (v0.1 configs
resolve relative paths against the YAML location; absolute paths are
unambiguous under both v0 and v1 semantics).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pytest
from click.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.fixtures.synthetic_data import SyntheticTree, make_synthetic_tree

CONFIG_TEMPLATE_DIR = Path(__file__).resolve().parent / "configs"


@dataclass(frozen=True)
class RenderedConfig:
    """One rendered config file and the output directory it targets."""

    path: Path
    out_dir: Path


@pytest.fixture(scope="session")
def synthetic_tree(tmp_path_factory: pytest.TempPathFactory) -> SyntheticTree:
    """
    Build the deterministic synthetic demo-data tree once per session.

    Args:
        tmp_path_factory: pytest session-scoped temporary directory factory.

    Returns:
        The generated :class:`SyntheticTree` under a session temp dir.
    """
    root = tmp_path_factory.mktemp("synthetic") / "data"
    return make_synthetic_tree(root, seed=42)


@pytest.fixture(scope="session")
def results_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Session-scoped root directory for all pipeline outputs.

    Args:
        tmp_path_factory: pytest session-scoped temporary directory factory.

    Returns:
        A writable directory; each test uses one subdirectory below it.
    """
    return tmp_path_factory.mktemp("results")


@pytest.fixture(scope="session")
def render_config(results_root: Path):
    """
    Return a renderer turning a config template into an executable YAML.

    The renderer substitutes ``@DATA_ROOT@`` with the synthetic tree root
    and ``@OUT_DIR@`` with ``<results_root>/<out_name>``, plus any extra
    ``{token: value}`` pairs the caller passes (e.g. a trained pipeline
    path for predict-mode configs). The rendered file is written next to
    the results directory so relative paths inside it (if any) stay sane.
    """

    def _render(
        template_name: str,
        out_name: str,
        synthetic: SyntheticTree,
        extra: Optional[Dict[str, str]] = None,
    ) -> RenderedConfig:
        """
        Render ``configs/<template_name>`` into a runnable YAML file.

        Args:
            template_name: Template file name inside ``configs/``.
            out_name: Subdirectory name for this run's outputs; also used
                as the rendered file stem.
            synthetic: The session synthetic tree.
            extra: Additional ``@KEY@ -> value`` substitutions.

        Returns:
            A :class:`RenderedConfig` with the rendered YAML path and the
            output directory the config writes to.
        """
        text = (CONFIG_TEMPLATE_DIR / template_name).read_text(encoding="utf-8")
        out_dir = results_root / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        replacements = {
            "@DATA_ROOT@": synthetic.root.as_posix(),
            "@OUT_DIR@": out_dir.as_posix(),
        }
        replacements.update(extra or {})
        for token, value in replacements.items():
            text = text.replace(token, value)
        rendered = results_root / f"{out_name}.yaml"
        rendered.write_text(text, encoding="utf-8")
        return RenderedConfig(path=rendered, out_dir=out_dir)

    return _render


@pytest.fixture
def cli_runner() -> CliRunner:
    """Return a Click ``CliRunner`` for in-process CLI invocations."""
    return CliRunner(mix_stderr=False)


def run_cli(cli_runner: CliRunner, args: list) -> "object":
    """
    Invoke the ``habit`` CLI, failing loudly on any exception.

    Args:
        cli_runner: Click test runner.
        args: Argument vector after ``habit`` (e.g. ``["get-habitat", ...]``).

    Returns:
        The Click ``Result``; the test fails on non-zero exit with output.
    """
    from habit.cli import cli

    result = cli_runner.invoke(cli, args, catch_exceptions=False)
    assert result.exit_code == 0, (
        f"habit {' '.join(args)} exited {result.exit_code}:\n{result.output}"
    )
    return result
