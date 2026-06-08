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
Utilities module for HABIT package.

Provides common utilities including:
- parallel_utils: Parallel processing with unified interface
- log_utils: Centralized logging management
- progress_utils: Progress bar utilities
- config_utils: Configuration loading and validation
- io_utils: Input/output operations
"""

from .parallel_utils import (
    parallel_map,
    parallel_map_simple,
    ParallelProcessor,
    ProcessingResult,
)

__all__ = [
    "parallel_map",
    "parallel_map_simple", 
    "ParallelProcessor",
    "ProcessingResult",
]
