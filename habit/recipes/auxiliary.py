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
"""Thin L4 wrappers for auxiliary CLI utilities.

These recipes expose dice overlap, DICOM metadata extraction, and tabular
merging as library-callable functions so integrators need not invoke CLI
machinery or accept HABIT's command-line argument shapes.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple, Union

import pandas as pd

from habit.exceptions import HABITAPIError
from habit.utils.dice_calculator import compute_dice, run_dice_calculation
from habit.utils.dicom_utils import batch_read_dicom_info
from habit.utils.optional_deps import require_excel_backend

__all__ = ["dice", "dicom_info", "merge_tables"]


def dice(
    input1: str,
    input2: str,
    *,
    output: Optional[str] = None,
    mask_keyword: str = "masks",
    label_id: int = 1,
) -> pd.DataFrame:
    """
    Compute pairwise Dice coefficients between two mask batches.

    Thin wrapper around :func:`habit.utils.dice_calculator.run_dice_calculation`
    that optionally returns the results table instead of only writing CSV.

    Args:
        input1: First input directory or YAML path list.
        input2: Second input directory or YAML path list.
        output: Optional CSV destination; when omitted, results are returned only.
        mask_keyword: Subfolder keyword used to locate mask files.
        label_id: Label value to compare inside each mask.

    Returns:
        DataFrame with columns ``Subject``, ``MaskType``, ``Dice``, and paths.
    """
    if output:
        run_dice_calculation(input1, input2, output, mask_keyword, label_id)
        return pd.read_csv(output)

    # Re-use the path-matching logic without requiring an output file: call the
    # batch helper with a temporary CSV under the system temp directory.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
        temp_path = handle.name
    try:
        run_dice_calculation(input1, input2, temp_path, mask_keyword, label_id)
        return pd.read_csv(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def dicom_info(
    input_path: str,
    *,
    tags: Optional[Sequence[str]] = None,
    recursive: bool = True,
    output: Optional[str] = None,
    output_format: str = "csv",
    group_by_series: bool = True,
    one_file_per_folder: bool = False,
    dicom_extensions: Optional[Set[str]] = None,
    include_no_extension: bool = False,
    num_workers: Optional[int] = None,
    max_depth: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extract selected DICOM tags from files or directories.

    Args:
        input_path: DICOM directory, single file, or YAML config path.
        tags: Optional tag names to extract; ``None`` uses library defaults.
        recursive: Walk subdirectories when ``input_path`` is a directory.
        output: Optional path to persist results (format controlled by
            ``output_format``).
        output_format: ``csv``, ``excel``, or ``json`` when writing ``output``.
        group_by_series: Read one representative file per series.
        one_file_per_folder: Sample one DICOM per folder for faster scans.
        dicom_extensions: Valid extensions when ``one_file_per_folder`` is set.
        include_no_extension: Probe extensionless files via DICOM magic bytes.
        num_workers: Thread pool size; ``1`` disables parallelism.
        max_depth: Maximum directory recursion depth.

    Returns:
        DataFrame of extracted tag values.

    Raises:
        HABITAPIError: When pydicom is unavailable or extraction fails.
    """
    tag_list = list(tags) if tags is not None else None
    df = batch_read_dicom_info(
        input_path,
        tags=tag_list,
        recursive=recursive,
        group_by_series=group_by_series,
        one_file_per_folder=one_file_per_folder,
        dicom_extensions=dicom_extensions,
        include_no_extension=include_no_extension,
        num_workers=num_workers,
        max_depth=max_depth,
    )
    if df is None:
        raise HABITAPIError(
            "DICOM info extraction returned no data. Ensure pydicom is installed "
            "and the input path contains readable DICOM files."
        )
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "excel":
            require_excel_backend(
                purpose="writing the DICOM info table as .xlsx"
            )
            df.to_excel(destination, index=False)
        elif output_format == "json":
            df.to_json(destination, orient="records", indent=2)
        else:
            df.to_csv(destination, index=False)
    return df


def merge_tables(
    input_files: Sequence[str],
    *,
    index_cols: Optional[Sequence[str]] = None,
    separator: str = ",",
    encoding: str = "utf-8",
    join_type: str = "inner",
) -> pd.DataFrame:
    """
    Merge feature tables horizontally on a shared index column.

    This is the library counterpart of ``habit merge-csv``: multiple CSV or
    Excel files are joined on subject identifiers without going through CLI
    argument parsing.

    Args:
        input_files: Paths to CSV or Excel tables (at least two).
        index_cols: Index column name(s). ``None`` uses each file's first
            column; a single name applies to every file; one name per file
            when the sequence length matches ``input_files``.
        separator: CSV delimiter.
        encoding: Text encoding for CSV files.
        join_type: ``inner`` or ``outer`` pandas join mode.

    Returns:
        Merged dataframe indexed by the resolved subject identifier column.

    Raises:
        HABITAPIError: When fewer than two readable files are supplied.
    """
    paths = tuple(input_files)
    if len(paths) < 2:
        raise HABITAPIError("merge_tables requires at least two input files.")

    index_col_list = list(index_cols) if index_cols is not None else None
    if index_col_list is not None and len(index_col_list) > 1:
        if len(index_col_list) != len(paths):
            raise HABITAPIError(
                "index_cols length must be 1 or match the number of input files."
            )

    merged_df: Optional[pd.DataFrame] = None

    def _resolve_csv_index_col(file_path: str, configured_index_col: Optional[str]) -> str:
        if configured_index_col:
            try:
                header_df = pd.read_csv(
                    file_path,
                    sep=separator,
                    encoding=encoding,
                    nrows=0,
                )
                if configured_index_col in header_df.columns:
                    return configured_index_col
            except Exception:
                pass
        with open(file_path, "r", encoding=encoding, newline="") as csv_file:
            reader = csv.reader(csv_file, delimiter=separator)
            header = next(reader, None)
            if not header:
                raise HABITAPIError(f"Empty CSV file: {file_path}")
            return header[0]

    for index, file_path in enumerate(paths):
        if not os.path.exists(file_path):
            continue

        if index_col_list is None:
            current_index_col: Optional[str] = None
        elif len(index_col_list) == 1:
            current_index_col = index_col_list[0]
        else:
            current_index_col = index_col_list[index]

        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in (".xlsx", ".xls"):
            require_excel_backend(
                purpose=f"reading the spreadsheet {os.path.basename(file_path)}"
            )
            header_df = pd.read_excel(file_path, nrows=0)
            resolved_index_col = (
                current_index_col
                if current_index_col and current_index_col in header_df.columns
                else header_df.columns[0]
            )
            df = pd.read_excel(
                file_path,
                dtype={resolved_index_col: str},
                index_col=resolved_index_col,
            )
        else:
            resolved_index_col = _resolve_csv_index_col(file_path, current_index_col)
            df = pd.read_csv(
                file_path,
                sep=separator,
                encoding=encoding,
                dtype={resolved_index_col: str},
                index_col=resolved_index_col,
            )

        if df.empty:
            continue

        df.index = df.index.astype(str)
        if merged_df is None:
            merged_df = df
        else:
            overlapping_cols = [column for column in df.columns if column in merged_df.columns]
            if overlapping_cols:
                file_tag = os.path.splitext(os.path.basename(file_path))[0]
                rename_map = {
                    column: f"{column}__{file_tag}" for column in overlapping_cols
                }
                df = df.rename(columns=rename_map)
            merged_df = merged_df.join(df, how=join_type)

    if merged_df is None:
        raise HABITAPIError("No valid input files were merged.")
    return merged_df
