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
"""Public DICOM sort API."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from habit.api.contracts import WorkflowResult, coerce_config
from habit.api.provenance import create_run_manifest, write_run_manifest

if TYPE_CHECKING:
    from habit.schemas.workflows.dicom_sort import DicomSortConfig

__all__ = ["DicomSortConfig", "run_dicom_sort"]


def __getattr__(name: str) -> Any:
    if name == "DicomSortConfig":
        from habit.schemas.workflows.dicom_sort import DicomSortConfig

        return DicomSortConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_dicom_sort(
    config: Union["DicomSortConfig", Mapping[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> WorkflowResult[None]:
    """
    Sort and convert DICOM series using a validated config object.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.schemas.workflows.dicom_sort.DicomSortConfig`.
        logger: Optional logger; core runner creates one when omitted.

    Returns:
        A result with the dcm2niix destination in ``artifacts``.
    """
    from habit.compat.dicom_sort_runner import run_dicom_sort as _run_dicom_sort
    from habit.schemas.workflows.dicom_sort import DicomSortConfig

    validated_config = coerce_config(config, DicomSortConfig)
    _run_dicom_sort(validated_config, logger=logger)
    output_dir = validated_config.output_dir or validated_config.out_dir
    manifest = create_run_manifest("dicom_sort", validated_config)
    manifest_path = write_run_manifest(manifest, output_dir)
    return WorkflowResult(
        output_dir=output_dir,
        metadata={
            "config_hash": manifest.config_hash,
            "habit_version": manifest.habit_version,
        },
        run_id=manifest.run_id,
        manifest_path=manifest_path,
    )
