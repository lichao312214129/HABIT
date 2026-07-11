# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Public preprocessing API (thin facade over ``habit.core.preprocessing``)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from habit.api.contracts import WorkflowResult, coerce_config

if TYPE_CHECKING:
    from habit.core.preprocessing.config_schemas import PreprocessingConfig

__all__ = ["PreprocessingConfig", "run_preprocess"]


def __getattr__(name: str) -> Any:
    if name == "PreprocessingConfig":
        from habit.core.preprocessing.config_schemas import PreprocessingConfig

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
            :class:`~habit.core.preprocessing.config_schemas.PreprocessingConfig`.
        logger: Optional logger; core runner creates one when omitted.

    Returns:
        A result with the workflow output directory in ``artifacts``.
    """
    from habit.core.preprocessing.config_schemas import PreprocessingConfig
    from habit.core.preprocessing.run import run_preprocess_from_config

    validated_config = coerce_config(config, PreprocessingConfig)
    run_preprocess_from_config(validated_config, logger=logger)
    return WorkflowResult(output_dir=validated_config.out_dir)
