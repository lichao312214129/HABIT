# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Public DICOM sort API (thin facade over ``habit.core.dicom_sort``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from habit.core.dicom_sort.config_schema import DicomSortConfig

__all__ = ["DicomSortConfig", "run_dicom_sort"]


def __getattr__(name: str) -> Any:
    if name == "DicomSortConfig":
        from habit.core.dicom_sort.config_schema import DicomSortConfig

        return DicomSortConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_dicom_sort(config: "DicomSortConfig") -> None:
    """
    Sort and convert DICOM series using a validated config object.

    Args:
        config: Loaded :class:`~habit.core.dicom_sort.config_schema.DicomSortConfig`.
    """
    from habit.core.dicom_sort.run import run_dicom_sort as _run_dicom_sort

    _run_dicom_sort(config)
