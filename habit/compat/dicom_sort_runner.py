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
"""Standalone dcm2niix DICOM sort runner (L1 adapter).

Migrated from ``habit.core.dicom_sort.run`` so production code outside
``habit.core`` can execute sort-dicom without importing the v0.1 engine tree.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from habit.schemas.workflows.dicom_sort import DicomSortConfig
from habit.utils.log_utils import get_module_logger
from habit.utils.subprocess_utils import run_capture_text

__all__ = ["run_dicom_sort"]


def _f_value(raw: Dict[str, Any]) -> str:
    """
    Resolve the dcm2niix ``-f`` string from a plain dict.

    Args:
        raw: Mapping that may contain keys ``f`` and/or ``filename_format``.

    Returns:
        str: Pattern passed to dcm2niix ``-f``.

    Raises:
        ValueError: If neither ``f`` nor ``filename_format`` is a non-empty string.
    """
    value = raw.get("f")
    if value is not None and str(value).strip() != "":
        return str(value)
    legacy = raw.get("filename_format")
    if legacy is not None and str(legacy).strip() != "":
        return str(legacy)
    raise ValueError(
        "dicom_sort: set `f` to your dcm2niix -f pattern (deprecated alias: filename_format)."
    )


def _exe(path: Optional[str], log: logging.Logger) -> str:
    """
    Resolve the dcm2niix executable path.

    Args:
        path: Optional path from config, or None for PATH lookup.
        log: Logger for warnings.

    Returns:
        str: Executable string passed to ``subprocess``.
    """
    if not path:
        which = shutil.which("dcm2niix")
        return which or ("dcm2niix.exe" if os.name == "nt" else "dcm2niix")
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate.resolve())
    if candidate.is_dir():
        name = "dcm2niix.exe" if os.name == "nt" else "dcm2niix"
        return str((candidate / name).resolve())
    log.warning("dcm2niix_path missing: %s, using PATH", path)
    return shutil.which("dcm2niix") or ("dcm2niix.exe" if os.name == "nt" else "dcm2niix")


def _argv(out_dir: str, in_dir: str, raw: Dict[str, Any]) -> List[str]:
    """
    Build dcm2niix argv after the executable name.

    Args:
        out_dir: Absolute output directory for ``-o``.
        in_dir: Absolute input directory passed as the final positional argument.
        raw: Step dict with ``f`` / ``filename_format`` and optional ``extra_args``.

    Returns:
        List[str]: Arguments for ``subprocess`` (excluding the executable).
    """
    argv: List[str] = ["-r", "y", "-f", _f_value(raw)]
    argv.extend(str(item) for item in (raw.get("extra_args") or []))
    argv.extend(["-o", out_dir, in_dir])
    return argv


def _run(exe: str, argv: List[str], log: logging.Logger) -> None:
    """
    Execute dcm2niix and raise on non-zero exit.

    Args:
        exe: Resolved dcm2niix executable.
        argv: Arguments after the executable name.
        log: Logger for stdout/stderr at debug level.
    """
    cmd: List[str] = [exe, *argv]
    log.info("dcm2niix sort cwd=%s cmd=%r", os.getcwd(), cmd)
    process = run_capture_text(cmd, check=False)
    if process.stdout:
        log.debug("stdout: %s", process.stdout)
    if process.stderr:
        log.debug("stderr: %s", process.stderr)
    if process.returncode != 0:
        raise RuntimeError(
            f"dcm2niix exit {process.returncode}: {process.stderr or process.stdout}"
        )


def run_dicom_sort(
    cfg: DicomSortConfig,
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Run dcm2niix once: input ``cfg.data_dir``, output ``cfg.output_dir`` or ``cfg.out_dir``.

    Args:
        cfg: Validated DICOM sort configuration.
        logger: Optional logger; defaults to this module's logger.

    Raises:
        NotADirectoryError: If ``data_dir`` is not a directory.
        RuntimeError: If dcm2niix is missing or exits non-zero.
        ValueError: If ``f`` / ``filename_format`` resolution fails.
    """
    log = logger or get_module_logger(__name__)
    raw: Dict[str, Any] = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
    out_root = os.path.abspath(str(raw.get("output_dir") or cfg.out_dir))
    in_root = os.path.abspath(str(cfg.data_dir))
    if not Path(in_root).is_dir():
        raise NotADirectoryError(in_root)

    exe = _exe(raw.get("dcm2niix_path"), log)
    if shutil.which(exe) is None and not Path(exe).is_file():
        raise RuntimeError(f"dcm2niix not found: {exe!r}")

    _run(exe, _argv(out_root, in_root, raw), log)
