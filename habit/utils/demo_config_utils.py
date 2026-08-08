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
Materialize bundled demo YAML configs into a user-chosen work directory.

After ``pip install habitat-analysis``, users do not need a git clone to obtain
the demo ``config/`` tree. The wheel ships a mirror under
``habit/resources/demo_config/`` (kept in sync with the repository ``config/``
via ``scripts/sync_demo_config.py``). Call :func:`copy_demo_config` or the CLI
``habit copy-demo-config`` to write that tree next to a user-owned
``demo_data/`` folder.

``demo_data/`` itself is **not** packaged; users download it separately.
"""

from __future__ import annotations

import importlib.resources as importlib_resources
import shutil
from pathlib import Path
from typing import Iterator, List, Optional, Union

from habit.utils.progress_utils import CustomTqdm

# Package that physically stores the mirrored demo YAML tree.
_RESOURCE_PACKAGE: str = "habit.resources.demo_config"

PathLike = Union[str, Path]


def demo_config_root() -> Path:
    """
    Return the filesystem path of the bundled demo-config resource tree.

    Returns:
        Path: Absolute directory containing the packaged ``config/`` mirror
        (YAML templates under habitat/, machine_learning/, …).

    Raises:
        FileNotFoundError: When the resource package is missing from the
            installation (incomplete wheel / editable install without sync).
    """
    resource = importlib_resources.files(_RESOURCE_PACKAGE)
    root = Path(str(resource))
    if not root.is_dir():
        raise FileNotFoundError(
            f"Bundled demo config package not found at {root}. "
            "The HABIT installation may be incomplete; developers should run "
            "`python scripts/sync_demo_config.py` before packaging."
        )
    return root


def iter_demo_config_files() -> Iterator[Path]:
    """
    Iterate all regular files under the bundled demo-config tree.

    Yields:
        Path: Absolute paths of bundled files (YAML / README), excluding the
        package ``__init__.py`` marker.
    """
    root: Path = demo_config_root()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "__init__.py":
            continue
        yield path


def copy_demo_config(
    dest: PathLike,
    *,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    """
    Copy bundled demo YAML configs into ``<dest>/config/``.

    Typical pip-user layout after this call (``demo_data/`` still downloaded
    separately)::

        <work_dir>/
        ├── config/          # created by this function
        └── demo_data/       # user download; not part of the wheel

    Args:
        dest: Work directory that should receive a ``config/`` subdirectory.
            Relative paths are resolved against the current working directory.
        overwrite: When False (default), refuse to replace an existing
            ``config/`` tree. When True, existing files under ``config/`` may
            be overwritten; extra user files are left alone.
        show_progress: When True, show a :class:`CustomTqdm` bar while copying.

    Returns:
        Path: Absolute path of the materialized ``config/`` directory.

    Raises:
        FileExistsError: When ``<dest>/config`` already exists and
            ``overwrite`` is False.
        FileNotFoundError: When the bundled resource tree is missing.
        OSError: On filesystem errors while creating directories or copying.
    """
    work_dir: Path = Path(dest).expanduser().resolve()
    target_config: Path = work_dir / "config"
    if target_config.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing directory: {target_config}. "
            "Pass overwrite=True (CLI: --overwrite) to replace files, or "
            "choose another --dest work directory."
        )

    source_root: Path = demo_config_root()
    files: List[Path] = list(iter_demo_config_files())
    if not files:
        raise FileNotFoundError(
            f"No demo config files found under {source_root}. "
            "Run `python scripts/sync_demo_config.py` in a source checkout."
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    target_config.mkdir(parents=True, exist_ok=True)

    progress: Optional[CustomTqdm] = None
    if show_progress:
        progress = CustomTqdm(total=len(files), desc="Copy demo config")

    try:
        for src in files:
            rel: Path = src.relative_to(source_root)
            out: Path = target_config / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    return target_config
