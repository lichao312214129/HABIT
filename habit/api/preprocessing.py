# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Public preprocessing API (thin facade over ``habit.core.preprocessing``)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from habit.core.preprocessing.config_schemas import PreprocessingConfig

__all__ = ["PreprocessingConfig", "run_preprocess"]


def __getattr__(name: str) -> Any:
    if name == "PreprocessingConfig":
        from habit.core.preprocessing.config_schemas import PreprocessingConfig

        return PreprocessingConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_preprocess(
    config: "PreprocessingConfig",
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Run the preprocessing batch pipeline from a validated config object.

    Args:
        config: Loaded :class:`~habit.core.preprocessing.config_schemas.PreprocessingConfig`.
        logger: Optional logger; core runner creates one when omitted.
    """
    from habit.core.preprocessing.run import run_preprocess_from_config

    run_preprocess_from_config(config, logger=logger)
