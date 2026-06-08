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
"""Configuration primitives: schema base, I/O, validation."""

from .base import BaseConfig, ConfigAccessor, ConfigValidationError
from .loader import (
    load_config,
    load_config_with_paths,
    resolve_config_paths,
    save_config,
    validate_config,
)
from .validator import ConfigValidator, load_and_validate_config

__all__ = [
    'BaseConfig',
    'ConfigAccessor',
    'ConfigValidationError',
    'load_config',
    'load_config_with_paths',
    'resolve_config_paths',
    'save_config',
    'validate_config',
    'ConfigValidator',
    'load_and_validate_config',
]
