"""Filesystem adapters for the table-format ICC workflow.

The numerical reliability calculations live in
:mod:`habit.evaluation.reliability`. This module owns only the legacy
CSV/Excel directory convention and the historical JSON serialisation shape.
It intentionally does not participate in voxel-level precision ICC.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from habit.evaluation.reliability import analyze_feature_tables
from habit.utils.log_utils import get_module_logger
from habit.utils.optional_deps import require_excel_backend

__all__ = [
    "analyze_feature_files",
    "load_and_merge_data",
    "parse_icc_directories",
    "save_icc_results",
]

logger = get_module_logger(__name__)


def load_and_merge_data(file_paths: List[str]) -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Load historical ICC CSV or Excel inputs without altering their indices.

    Args:
        file_paths: Session-table paths. CSV and ``.xlsx``/``.xls`` inputs
            are supported exactly as in the legacy workflow.

    Returns:
        Loaded tables and their file-stem session names in input order.

    Raises:
        FileNotFoundError: If an input path does not exist.
        ValueError: If an input suffix is not supported.
        OptionalDependencyError: If an Excel input requires an unavailable
            optional backend.
    """
    data_frames: List[pd.DataFrame] = []
    file_names: List[str] = []
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix == ".csv":
            frame = pd.read_csv(file_path, index_col=0)
        elif path.suffix in [".xlsx", ".xls"]:
            require_excel_backend(purpose=f"reading the ICC input table {path.name}")
            frame = pd.read_excel(file_path, index_col=0)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        data_frames.append(frame)
        file_names.append(path.stem)
    return data_frames, file_names


def parse_icc_directories(dir_list: List[str]) -> List[List[str]]:
    """
    Form ICC input groups by matching CSV or Excel file stems across folders.

    Args:
        dir_list: Input directories in the caller's requested session order.

    Returns:
        One path group per common stem, sorted by stem. Paths inside every
        group retain the directory order from ``dir_list``.
    """
    dir_files: Dict[str, Dict[str, str]] = {}
    for dir_path in dir_list:
        path = Path(dir_path)
        if path.is_dir():
            dir_files[dir_path] = {
                file.stem: str(file)
                for file in path.iterdir()
                if file.suffix.lower() in [".csv", ".xlsx", ".xls"]
            }
    if not dir_files:
        return []
    common_stems = set.intersection(*(set(files) for files in dir_files.values()))
    return [
        [files[stem] for files in dir_files.values()]
        for stem in sorted(common_stems)
    ]


def analyze_feature_files(
    file_paths: List[str],
    metrics: List[str] | None = None,
    selected_features: List[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Adapt legacy session files into the in-memory table reliability analysis.

    Args:
        file_paths: Input CSV or Excel session tables.
        metrics: Reliability metrics to calculate.
        selected_features: Optional feature-name subset.

    Returns:
        The unchanged historical nested reliability-result payload.
    """
    logger.info("Loading %d files...", len(file_paths))
    data_frames, file_names = load_and_merge_data(file_paths)
    return analyze_feature_tables(data_frames, file_names, metrics, selected_features)


def save_icc_results(results: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """
    Serialise table-reliability results using the legacy indented JSON format.

    Args:
        results: Nested result payload produced by
            :func:`analyze_feature_files`.
        output_path: Target JSON file; missing parent directories are created.
    """
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as handle:
            json.dump(results, handle, indent=4)
        logger.info("Results successfully saved to %s", output_path)
    except Exception as exc:
        logger.error("Failed to save results to %s: %s", output_path, exc)
        raise
