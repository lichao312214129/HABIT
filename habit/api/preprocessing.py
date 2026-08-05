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
"""Public preprocessing API."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from habit.api.contracts import WorkflowResult, coerce_config
from habit.api.provenance import create_run_manifest, write_run_manifest

if TYPE_CHECKING:
    from habit.schemas.workflows.preprocessing import PreprocessingConfig

__all__ = ["PreprocessingConfig", "run_preprocess"]


def __getattr__(name: str) -> Any:
    if name == "PreprocessingConfig":
        from habit.schemas.workflows.preprocessing import PreprocessingConfig

        return PreprocessingConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_preprocess(
    config: Union["PreprocessingConfig", Mapping[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> WorkflowResult[None]:
    """
    Run the preprocessing batch pipeline from a validated config object.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.schemas.workflows.preprocessing.PreprocessingConfig`.
        logger: Optional logger; core runner creates one when omitted.

    Returns:
        A result with the workflow output directory in ``artifacts``.
    """
    from habit.compat.legacy_core import run_preprocess_from_config
    from habit.schemas.workflows.preprocessing import PreprocessingConfig

    validated_config = coerce_config(config, PreprocessingConfig)
    run_preprocess_from_config(validated_config, logger=logger)
    manifest = create_run_manifest("preprocess", validated_config)
    manifest_path = write_run_manifest(manifest, validated_config.out_dir)
    return WorkflowResult(
        output_dir=validated_config.out_dir,
        metadata={
            "config_hash": manifest.config_hash,
            "habit_version": manifest.habit_version,
        },
        run_id=manifest.run_id,
        manifest_path=manifest_path,
    )
