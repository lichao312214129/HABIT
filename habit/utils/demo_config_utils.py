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
Materialize demo YAML configs into a user-chosen work directory.

Single source of truth
----------------------
Developers edit **only** the repository-root ``config/`` tree. There is no
hand-maintained duplicate.

Resolution order for :func:`demo_config_root`:

1. **Source / editable checkout** — if ``<repo>/config/`` exists next to the
   ``habit`` package, use it directly (``pip install -e .`` sees edits live).
2. **Installed wheel** — use the copy baked into
   ``habit.resources.demo_config`` at build time (``setup.py`` ``build_py``
   runs ``scripts/sync_demo_config.py`` automatically).

``demo_data/`` is never packaged; users download it beside the copied
``config/``.
"""

from __future__ import annotations

import importlib.resources as importlib_resources
import shutil
from pathlib import Path
from typing import Iterator, List, Optional, Union

from habit.utils.progress_utils import CustomTqdm

# Package that holds the wheel-baked mirror of repo ``config/``.
_RESOURCE_PACKAGE: str = "habit.resources.demo_config"

PathLike = Union[str, Path]


def _repo_config_dir() -> Optional[Path]:
    """
    Return the repository ``config/`` directory when running from a checkout.

    ``habit/utils/demo_config_utils.py`` → parents[2] is the repo root in an
    editable or in-tree install. After a normal wheel install that path is
    ``site-packages/``, which has no ``config/`` sibling — then we fall back
    to the packaged resource tree.

    Returns:
        Optional[Path]: Absolute ``config/`` path, or None when absent.
    """
    # habit/utils/<this file> -> habit/utils -> habit -> <repo or site-packages>
    candidate: Path = Path(__file__).resolve().parents[2] / "config"
    marker: Path = candidate / "habitat" / "config_habitat_two_step.yaml"
    if candidate.is_dir() and marker.is_file():
        return candidate
    return None


def demo_config_root() -> Path:
    """
    Return the directory of demo YAML templates to copy from.

    Prefers the live repository ``config/`` in editable/source checkouts;
    otherwise uses the wheel-bundled ``habit.resources.demo_config`` tree.

    Returns:
        Path: Absolute directory containing habitat/, machine_learning/, …

    Raises:
        FileNotFoundError: When neither the repo ``config/`` nor the bundled
            package resources are available.
    """
    repo_config: Optional[Path] = _repo_config_dir()
    if repo_config is not None:
        return repo_config

    resource = importlib_resources.files(_RESOURCE_PACKAGE)
    root = Path(str(resource))
    marker = root / "habitat" / "config_habitat_two_step.yaml"
    if root.is_dir() and marker.is_file():
        return root

    raise FileNotFoundError(
        "Demo config templates not found. In a source checkout, ensure "
        "repository config/ exists. In a wheel install, the package should "
        "include habit/resources/demo_config (populated automatically by "
        "setup.py build_py via scripts/sync_demo_config.py)."
    )


def iter_demo_config_files() -> Iterator[Path]:
    """
    Iterate demo template files under :func:`demo_config_root`.

    Yields:
        Path: Absolute paths of YAML / markdown templates. Skips package
        ``__init__.py`` markers that may exist in the wheel mirror.
    """
    root: Path = demo_config_root()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "__init__.py":
            continue
        suffix: str = path.suffix.lower()
        if suffix not in {".yaml", ".yml", ".md"}:
            continue
        yield path


def copy_demo_config(
    dest: PathLike,
    *,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    """
    Copy demo YAML configs into ``<dest>/config/``.

    Typical layout after this call (``demo_data/`` still downloaded
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
        FileNotFoundError: When no demo config source can be resolved.
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
            f"No demo config files found under {source_root}."
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
