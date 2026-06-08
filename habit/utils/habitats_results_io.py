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
Read/write habitat analysis result tables (``habitats.parquet`` / ``habitats.csv``).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

HabitatsResultsFormat = Literal["parquet", "csv"]
SUPPORTED_HABITATS_RESULTS_FORMATS: tuple[str, ...] = ("parquet", "csv")
HABITATS_RESULTS_STEM: str = "habitats"


def normalize_habitats_results_format(
    habitats_results_format: str,
) -> HabitatsResultsFormat:
    """
    Validate and normalize the configured habitats results format.

    Args:
        habitats_results_format: User-facing format name.

    Returns:
        Literal format key, either ``parquet`` or ``csv``.

    Raises:
        ValueError: If the format is not supported.
    """
    normalized = str(habitats_results_format).strip().lower()
    if normalized not in SUPPORTED_HABITATS_RESULTS_FORMATS:
        supported = ", ".join(SUPPORTED_HABITATS_RESULTS_FORMATS)
        raise ValueError(
            f"Unsupported habitats_results_format: {habitats_results_format!r}. "
            f"Supported values: {supported}."
        )
    return normalized  # type: ignore[return-value]


def habitats_results_filename(habitats_results_format: str) -> str:
    """
    Build the on-disk filename for a habitats results table.

    Args:
        habitats_results_format: ``parquet`` or ``csv``.

    Returns:
        str: ``habitats.parquet`` or ``habitats.csv``.
    """
    fmt = normalize_habitats_results_format(habitats_results_format)
    if fmt == "parquet":
        return f"{HABITATS_RESULTS_STEM}.parquet"
    return f"{HABITATS_RESULTS_STEM}.csv"


def habitats_results_path(
    out_dir: Union[str, Path],
    habitats_results_format: str,
) -> Path:
    """
    Resolve the full output path for a habitats results table.

    Args:
        out_dir: Habitat analysis output directory.
        habitats_results_format: ``parquet`` or ``csv``.

    Returns:
        Path: Absolute-style path under ``out_dir``.
    """
    return Path(out_dir) / habitats_results_filename(habitats_results_format)


def find_habitats_results_file(
    out_dir: Union[str, Path],
    preferred_format: Optional[str] = None,
) -> Optional[Path]:
    """
    Locate an existing habitats results table inside an output directory.

    When ``preferred_format`` is set, that file is returned if present.
    Otherwise parquet is tried before csv for backward compatibility.

    Args:
        out_dir: Habitat analysis output directory.
        preferred_format: Optional format hint from configuration.

    Returns:
        Optional[Path]: Existing results file, or ``None`` when not found.
    """
    root = Path(out_dir)
    if preferred_format is not None:
        preferred_path = habitats_results_path(root, preferred_format)
        if preferred_path.is_file():
            return preferred_path

    for fmt in SUPPORTED_HABITATS_RESULTS_FORMATS:
        candidate = habitats_results_path(root, fmt)
        if candidate.is_file():
            return candidate
    return None


def load_habitats_results(source: Union[str, Path]) -> pd.DataFrame:
    """
    Load a habitats results table from a file path or output directory.

    Args:
        source: Path to ``habitats.parquet`` / ``habitats.csv``, or a directory
            that contains one of those files.

    Returns:
        pd.DataFrame: Loaded habitats results table.

    Raises:
        FileNotFoundError: If no supported habitats results file exists.
        ValueError: If the file extension is unsupported.
    """
    path = Path(source)
    if path.is_dir():
        resolved = find_habitats_results_file(path)
        if resolved is None:
            raise FileNotFoundError(
                f"No habitats results file found in directory: {path}"
            )
        path = resolved

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported habitats results file extension: {path.suffix}. "
        "Expected .parquet or .csv."
    )


def save_habitats_results(
    results_df: pd.DataFrame,
    out_dir: Union[str, Path],
    habitats_results_format: str,
    *,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Persist the habitats results table using the requested on-disk format.

    Args:
        results_df: Habitat clustering result table.
        out_dir: Output directory.
        habitats_results_format: ``parquet`` (default) or ``csv``.
        logger: Optional logger for timing and path messages.

    Returns:
        Path: Written results file path.

    Raises:
        ImportError: If parquet is requested but ``pyarrow`` is not installed.
    """
    fmt = normalize_habitats_results_format(habitats_results_format)
    out_path = habitats_results_path(out_dir, fmt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = time.monotonic()
    if fmt == "parquet":
        try:
            results_df.to_parquet(out_path, index=False, engine="pyarrow")
        except ImportError as exc:
            raise ImportError(
                "Parquet export requires pyarrow. Install with: pip install pyarrow"
            ) from exc
    else:
        results_df.to_csv(out_path, index=False)

    elapsed_sec = time.monotonic() - started_at
    if logger is not None:
        logger.info(
            "Habitats results saved to %s (format=%s, rows=%d, cols=%d, "
            "elapsed_sec=%.2f)",
            out_path,
            fmt,
            len(results_df),
            len(results_df.columns),
            elapsed_sec,
        )
    return out_path
