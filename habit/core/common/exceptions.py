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
HABIT Exception Hierarchy

Defines the core exception classes used throughout the HABIT package.
Following top-tier open-source practices, all custom exceptions inherit
from a base HabitError.
"""

class HabitError(Exception):
    """Base exception class for all HABIT errors."""
    pass

class ConfigurationError(HabitError):
    """Raised when there is an error in the configuration (YAML or dict)."""
    pass

class DataFormatError(HabitError):
    """Raised when input data format is invalid or unsupported."""
    pass

class NotFittedError(HabitError, ValueError):
    """Raised when a model or transformer is used before being fitted.
    Inherits from ValueError for scikit-learn compatibility.
    """
    pass

class ComponentNotFoundError(HabitError):
    """Raised when a requested component (model, selector, etc.) is not found in the registry."""
    pass

class ProcessingError(HabitError):
    """Raised when an error occurs during data processing or pipeline execution."""
    pass
