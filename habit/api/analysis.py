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
"""Public auxiliary analysis API (table-format ICC reliability)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Union

from habit.api.contracts import WorkflowResult, coerce_config
from habit.api.provenance import create_run_manifest, write_run_manifest

if TYPE_CHECKING:
    from habit.schemas.workflows.icc import ICCConfig

__all__ = [
    "ICCConfig",
    "run_icc_analysis",
]


def __getattr__(name: str) -> Any:
    if name == "ICCConfig":
        from habit.schemas.workflows.icc import ICCConfig

        return ICCConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_icc_analysis(
    config: Union["ICCConfig", Mapping[str, Any]],
) -> WorkflowResult[None]:
    """
    Run ICC analysis from a validated config object.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.schemas.workflows.icc.ICCConfig`.

    Returns:
        A result with the ICC output directory in ``artifacts``.
    """
    from habit.recipes.icc_runner import run_icc_analysis_from_config
    from habit.schemas.workflows.icc import ICCConfig

    validated_config = coerce_config(config, ICCConfig)
    run_icc_analysis_from_config(validated_config)
    output_path = Path(validated_config.output.path)
    manifest = create_run_manifest("icc_analysis", validated_config)
    manifest_path = write_run_manifest(manifest, output_path.parent)
    return WorkflowResult(
        output_dir=output_path.parent,
        artifacts={"icc_result": output_path},
        metadata={
            "config_hash": manifest.config_hash,
            "habit_version": manifest.habit_version,
        },
        run_id=manifest.run_id,
        manifest_path=manifest_path,
    )
