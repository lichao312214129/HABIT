# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Public DICOM sort API (thin facade over ``habit.core.dicom_sort``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Union

from habit.api.contracts import WorkflowResult, coerce_config

if TYPE_CHECKING:
    from habit.core.dicom_sort.config_schema import DicomSortConfig

__all__ = ["DicomSortConfig", "run_dicom_sort"]


def __getattr__(name: str) -> Any:
    if name == "DicomSortConfig":
        from habit.core.dicom_sort.config_schema import DicomSortConfig

        return DicomSortConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_dicom_sort(
    config: Union["DicomSortConfig", Mapping[str, Any]],
) -> WorkflowResult[None]:
    """
    Sort and convert DICOM series using a validated config object.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.core.dicom_sort.config_schema.DicomSortConfig`.

    Returns:
        A result with the dcm2niix destination in ``artifacts``.
    """
    from habit.core.dicom_sort.config_schema import DicomSortConfig
    from habit.core.dicom_sort.run import run_dicom_sort as _run_dicom_sort

    validated_config = coerce_config(config, DicomSortConfig)
    _run_dicom_sort(validated_config)
    return WorkflowResult(
        output_dir=validated_config.output_dir or validated_config.out_dir
    )
