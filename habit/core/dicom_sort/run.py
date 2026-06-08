# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
Run a single dcm2niix invocation for DICOM sort (no BatchProcessor).

Command shape: ``[exe] -r y -f <f> [extra_args...] -o OUT IN``.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from habit.core.dicom_sort.config_schema import DicomSortConfig
from habit.utils.log_utils import get_module_logger
from habit.utils.subprocess_utils import run_capture_text


def _f_value(raw: Dict[str, Any]) -> str:
    """
    Resolve the dcm2niix ``-f`` string from a plain dict (``f`` or deprecated ``filename_format``).

    Args:
        raw: Mapping that may contain keys ``f`` and/or ``filename_format``.

    Returns:
        str: Pattern passed to dcm2niix ``-f``.

    Raises:
        ValueError: If neither ``f`` nor ``filename_format`` is a non-empty string.
    """
    v = raw.get("f")
    if v is not None and str(v).strip() != "":
        return str(v)
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
        path: Optional path from config: executable file, directory containing the binary, or None for PATH.
        log: Logger for warnings.

    Returns:
        str: Executable string passed to ``subprocess``.
    """
    if not path:
        w = shutil.which("dcm2niix")
        return w or ("dcm2niix.exe" if os.name == "nt" else "dcm2niix")
    p = Path(path)
    if p.is_file():
        return str(p.resolve())
    if p.is_dir():
        name = "dcm2niix.exe" if os.name == "nt" else "dcm2niix"
        return str((p / name).resolve())
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
    out: List[str] = ["-r", "y", "-f", _f_value(raw)]
    out.extend(str(x) for x in (raw.get("extra_args") or []))
    out.extend(["-o", out_dir, in_dir])
    return out


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
    p = run_capture_text(cmd, check=False)
    if p.stdout:
        log.debug("stdout: %s", p.stdout)
    if p.stderr:
        log.debug("stderr: %s", p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"dcm2niix exit {p.returncode}: {p.stderr or p.stdout}")


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
        ValueError: If ``f`` / ``filename_format`` resolution fails (should not occur for valid cfg).
    """
    log = logger or get_module_logger(__name__)
    raw: Dict[str, Any]
    if hasattr(cfg, "model_dump"):
        raw = cfg.model_dump()
    else:
        raw = cfg.dict()
    out_root = os.path.abspath(str(raw.get("output_dir") or cfg.out_dir))
    in_root = os.path.abspath(str(cfg.data_dir))
    if not Path(in_root).is_dir():
        raise NotADirectoryError(in_root)

    exe = _exe(raw.get("dcm2niix_path"), log)
    if shutil.which(exe) is None and not Path(exe).is_file():
        raise RuntimeError(f"dcm2niix not found: {exe!r}")

    _run(exe, _argv(out_root, in_root, raw), log)
