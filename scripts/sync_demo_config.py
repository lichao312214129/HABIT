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
Sync repository ``config/`` demo YAMLs into ``habit/resources/demo_config/``.

The installable wheel ships the mirrored tree under the package so users can
run ``habit copy-demo-config`` (or ``habit.copy_demo_config``) without cloning
the repository. Developers who edit files under the repo-root ``config/``
directory should re-run this script before committing packaging changes::

    python scripts/sync_demo_config.py

Only YAML templates and ``README_CONFIG.md`` are copied; accidental log
artefacts under ``config/`` are skipped. ``demo_data/`` is never packaged.
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
        dst: Destination package resource tree.
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
    # Keep demo_config importable as a package (matches radiomics resources).
    (dst / "__init__.py").write_text(
        '"""Bundled demo YAML configs shipped inside the HABIT wheel."""\n',
        encoding="utf-8",
    )

    for rel in relative_files:
        target: Path = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, target)

    return relative_files


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the sync script."""
    parser = argparse.ArgumentParser(
        description="Sync repo config/ into habit/resources/demo_config/.",
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
