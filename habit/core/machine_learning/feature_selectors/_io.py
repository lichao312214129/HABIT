"""
Shared I/O utilities for feature selector modules.

This private module centralises the file-type detection and multi-format data
loading logic that was previously copy-pasted verbatim into
``stepwise_selector``, ``python_stepwise_selector``, and
``univariate_logistic_selector``.  Any change to supported formats (e.g.
adding TSV/Parquet v2 support) now only needs to happen here.

Public symbols
--------------
detect_file_type(input_path) -> Optional[str]
load_data(input_data, target_column, file_type, columns) -> (DataFrame, Series)
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from habit.utils.log_utils import get_module_logger


LOGGER = get_module_logger("ml.feature_selectors.io")


def detect_file_type(input_path: str) -> Optional[str]:
    """
    Detect the tabular file type from its extension, falling back to content
    sniffing for ambiguous or extension-less files.

    Args:
        input_path: Path to the input file.

    Returns:
        One of ``'csv'``, ``'excel'``, ``'parquet'``, ``'json'``, ``'pickle'``,
        or *None* when detection fails.
    """
    file_ext: str = Path(input_path).suffix.lower()
    ext_map = {
        ".csv": "csv",
        ".xlsx": "excel",
        ".xls": "excel",
        ".parquet": "parquet",
        ".json": "json",
        ".pkl": "pickle",
        ".pickle": "pickle",
    }

    if file_ext in ext_map:
        return ext_map[file_ext]

    # Content sniffing for files without a recognised extension.
    try:
        with open(input_path, "r", encoding="utf-8") as fh:
            first_line = fh.readline().strip()
            if "," in first_line and len(first_line.split(",")) > 1:
                return "csv"
            if first_line.startswith("{") or first_line.startswith("["):
                return "json"
    except Exception:
        pass

    return None


def load_data(
    input_data: Union[str, pd.DataFrame],
    target_column: Optional[str] = None,
    file_type: Optional[str] = None,
    columns: Optional[Union[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load tabular data from a file path or an existing DataFrame and split it
    into features (X) and the target variable (y).

    Args:
        input_data: File path (str) or a pre-loaded DataFrame.
        target_column: Name of the label/target column.  Required.
        file_type: Override the auto-detected file type.  Recognised values:
            ``'csv'``, ``'excel'``, ``'parquet'``, ``'json'``, ``'pickle'``.
        columns: Feature column selection.  Accepts:
            - a list of column names,
            - a ``'start:end'`` index-range string (target column excluded),
            - a comma-separated string of column names.
            If *None*, all columns except ``target_column`` are used.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and label series y.

    Raises:
        FileNotFoundError: If ``input_data`` is a path that does not exist.
        ValueError: If the file type cannot be detected, is unsupported, or if
            the loaded data is empty or ``target_column`` is not specified.
    """
    loaders = {
        "csv": pd.read_csv,
        "excel": pd.read_excel,
        "parquet": pd.read_parquet,
        "json": pd.read_json,
        "pickle": pd.read_pickle,
    }

    if isinstance(input_data, pd.DataFrame):
        data = input_data
    else:
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"Error: File {input_data} does not exist")

        if file_type is None:
            file_type = detect_file_type(input_data)
            if file_type is None:
                raise ValueError(f"Cannot detect file type: {input_data}")
            LOGGER.info("Automatically detected file type: %s", file_type)

        if file_type.lower() not in loaders:
            raise ValueError(f"Unsupported file type: {file_type}")

        try:
            data = loaders[file_type.lower()](input_data)
            if data.empty:
                raise ValueError(f"Loaded data is empty: {input_data}")
        except Exception as exc:
            raise Exception(f"Error loading data: {exc}") from exc

    if target_column is None:
        raise ValueError("Target column name must be specified")

    # --- Feature column selection ---
    if columns is not None:
        if isinstance(columns, str):
            if ":" in columns:
                start_s, end_s = columns.split(":")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else None
                all_cols = data.columns.tolist()
                x_cols = all_cols[start:end]
                if target_column in x_cols:
                    x_cols.remove(target_column)
                X = data[x_cols]
            else:
                columns_list = [c.strip() for c in columns.split(",")]
                X = data[columns_list]
        elif isinstance(columns, list):
            X = data[columns]
        else:
            raise ValueError(
                "columns parameter must be a list of column names or a column "
                "range string"
            )
    else:
        X = data.drop(columns=[target_column])

    y = data[target_column]
    return X, y
