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
"""ICC reliability analysis runner (L1 adapter)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from habit.compat.icc_analyzer import (
    analyze_features,
    print_statistics,
    print_summary,
    save_results,
)
from habit.schemas.workflows.icc import ICCConfig
from habit.utils.log_utils import get_module_logger

__all__ = ["run_icc_analysis_from_config"]

logger = get_module_logger(__name__)


def _parse_directories(dir_list: List[str]) -> List[List[str]]:
    """
    Parse a list of directories and build file groups by matching filenames.

    Args:
        dir_list: List of directory paths.

    Returns:
        List of file groups; each group contains paths from each directory
        that share the same stem (filename without extension).
    """
    dir_files = {}
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

    common_stems = set.intersection(*[set(files.keys()) for files in dir_files.values()])
    file_groups: List[List[str]] = []
    for stem in sorted(common_stems):
        group = [files[stem] for _, files in dir_files.items()]
        file_groups.append(group)
    return file_groups


def run_icc_analysis_from_config(config: ICCConfig) -> None:
    """
    Orchestrate ICC analysis from a validated :class:`ICCConfig`.

    Args:
        config: Validated ICC configuration object.
    """
    output_path = config.output.path
    metrics = config.metrics or ["icc2", "icc3"]
    selected_features = config.selected_features

    if config.input.type == "files":
        file_groups = config.parse_file_groups()
    else:
        dir_list = config.parse_directories()
        file_groups = _parse_directories(dir_list)

    if not file_groups:
        logger.error("No valid file groups found in the configuration.")
        return

    logger.info("Found %d file group(s) to analyze.", len(file_groups))
    for index, group in enumerate(file_groups):
        logger.info(
            "  Group %d: %s",
            index + 1,
            ", ".join(os.path.basename(path) for path in group),
        )

    logger.info("Metrics to be calculated: %s", ", ".join(metrics))
    if selected_features:
        logger.info("Analyzing a subset of %d features.", len(selected_features))

    all_results = {}
    for group in file_groups:
        try:
            group_results = analyze_features(
                file_paths=group,
                metrics=metrics,
                selected_features=selected_features,
            )
            all_results.update(group_results)
        except Exception as exc:
            logger.error(
                "An error occurred while processing group %s: %s",
                group,
                exc,
                exc_info=True,
            )

    if not all_results:
        logger.warning("Analysis generated no results.")
        return

    save_results(all_results, output_path)
    print_summary(all_results)
    print_statistics(all_results)
    logger.info("ICC analysis process finished.")
