#!/usr/bin/env python
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
Build helper: copy repository ``config/`` → ``habit/resources/demo_config/``.

Canonical demo YAML lives **only** at repo-root ``config/``. Developers edit
that tree and never maintain a second copy by hand.

``setup.py`` invokes :func:`sync_demo_config` from the ``build_py`` command
so wheels/sdists always bake a fresh mirror. Editable installs
(``pip install -e .``) do **not** need this script: runtime reads
``config/`` directly via :func:`habit.utils.demo_config_utils.demo_config_root`.

Manual use (optional, e.g. inspecting the wheel layout locally)::

    python scripts/sync_demo_config.py

Generated files under ``habit/resources/demo_config/`` (except ``__init__.py``)
are gitignored. ``demo_data/`` is never packaged.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Set

_ROOT: Path = Path(__file__).resolve().parents[1]
_SRC: Path = _ROOT / "config"
_DST: Path = _ROOT / "habit" / "resources" / "demo_config"

_INIT_TEXT: str = (
    '"""Bundled demo YAML configs shipped inside the HABIT wheel.\n'
    "\n"
    "Populated at build time from repository ``config/`` by\n"
    "``scripts/sync_demo_config.py`` (see setup.py build_py). Edit repo\n"
    "``config/`` only — do not hand-maintain files here.\n"
    '"""\n'
)

# Extensions that belong in the bundled demo-config tree.
_INCLUDE_SUFFIXES: Set[str] = {".yaml", ".yml", ".md"}

# Never ship runtime log noise that may linger under config/.
_EXCLUDE_SUFFIXES: Set[str] = {".log"}


def _iter_source_files(src: Path) -> Iterable[Path]:
    """
    Yield files under ``src`` that should be mirrored into the package.

    Args:
        src: Repository ``config/`` directory.

    Yields:
        Path: Absolute paths of files to copy.
    """
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        suffix: str = path.suffix.lower()
        if suffix in _EXCLUDE_SUFFIXES:
            continue
        if suffix not in _INCLUDE_SUFFIXES:
            continue
        yield path


def sync_demo_config(
    src: Path = _SRC,
    dst: Path = _DST,
    *,
    dry_run: bool = False,
) -> List[Path]:
    """
    Mirror eligible files from ``src`` into ``dst``, replacing prior contents.

    Args:
        src: Source tree (repo ``config/``).
        dst: Destination package resource tree (build artefact).
        dry_run: When True, report planned copies without writing.

    Returns:
        List[Path]: Relative paths (under ``src``) that were / would be copied.

    Raises:
        FileNotFoundError: When ``src`` does not exist.
    """
    if not src.is_dir():
        raise FileNotFoundError(f"Source config directory not found: {src}")

    relative_files: List[Path] = [
        path.relative_to(src) for path in _iter_source_files(src)
    ]
    if dry_run:
        return relative_files

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "__init__.py").write_text(_INIT_TEXT, encoding="utf-8")

    for rel in relative_files:
        target: Path = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, target)

    return relative_files


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the sync script."""
    parser = argparse.ArgumentParser(
        description=(
            "Build helper: sync repo config/ into "
            "habit/resources/demo_config/ (setup.py calls this automatically)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be copied without writing.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Entry point for ``python scripts/sync_demo_config.py``.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        int: Process exit code (0 on success).
    """
    args = _parse_args(argv)
    files = sync_demo_config(dry_run=bool(args.dry_run))
    action = "Would copy" if args.dry_run else "Copied"
    print(f"{action} {len(files)} file(s) into {_DST.relative_to(_ROOT)}")
    for rel in files[:10]:
        print(f"  {rel.as_posix()}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
