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
ICC Analysis Task Handler

This module acts as the primary downstream handler for the ICC command-line
interface. It takes an :class:`ICCConfig` instance, processes it, and uses
the core ``icc_analyzer`` module to run the analysis and report results.

Author: Li Chao
Email: lich356@mail.sysu.edu.cn
"""

import logging
import os
from pathlib import Path
from typing import List

from .config import ICCConfig
from .icc_analyzer import (
    analyze_features,
    save_results,
    print_summary,
    print_statistics,
)
from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)


def _parse_directories(dir_list: List[str]) -> List[List[str]]:
    """
    Parse a list of directories and build file groups by matching filenames.

    Args:
        dir_list: list of directory paths.

    Returns:
        List of file groups; each group contains paths from each directory
        that share the same stem (filename without extension).
    """
    dir_files = {}
    for dir_path in dir_list:
        p = Path(dir_path)
        if p.is_dir():
            dir_files[dir_path] = {
                f.stem: str(f)
                for f in p.iterdir()
                if f.suffix.lower() in ['.csv', '.xlsx', '.xls']
            }

    if not dir_files:
        return []

    # Find common stems across all directories.
    common_stems = set.intersection(*[set(files.keys()) for files in dir_files.values()])

    file_groups: List[List[str]] = []
    for stem in sorted(list(common_stems)):
        group = [files[stem] for _, files in dir_files.items()]
        file_groups.append(group)

    return file_groups


def run_icc_analysis_from_config(config: ICCConfig) -> None:
    """
    Orchestrate the ICC analysis from a validated :class:`ICCConfig`.

    Args:
        config: the validated configuration object.
    """
    output_path = config.output.path
    metrics = config.metrics or ['icc2', 'icc3']
    selected_features = config.selected_features

    if config.input.type == 'files':
        file_groups = config.parse_file_groups()
    else:
        dir_list = config.parse_directories()
        file_groups = _parse_directories(dir_list)

    if not file_groups:
        logger.error("No valid file groups found in the configuration.")
        return

    logger.info(f"Found {len(file_groups)} file group(s) to analyze.")
    for i, group in enumerate(file_groups):
        logger.info(f"  Group {i + 1}: {', '.join(os.path.basename(f) for f in group)}")

    logger.info(f"Metrics to be calculated: {', '.join(metrics)}")
    if selected_features:
        logger.info(f"Analyzing a subset of {len(selected_features)} features.")

    # Analyse each file group and aggregate results.
    all_results = {}
    for group in file_groups:
        try:
            group_results = analyze_features(
                file_paths=group,
                metrics=metrics,
                selected_features=selected_features,
            )
            all_results.update(group_results)
        except Exception as e:
            logger.error(f"An error occurred while processing group {group}: {e}", exc_info=True)

    if not all_results:
        logger.warning("Analysis generated no results.")
        return

    save_results(all_results, output_path)
    print_summary(all_results)
    print_statistics(all_results)

    logger.info("ICC analysis process finished.")
