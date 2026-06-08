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
PreprocessingConfigurator: factory for the image preprocessing pipeline.
"""

from __future__ import annotations

from typing import Any

from habit.core.common.configurators.base import BaseConfigurator


class PreprocessingConfigurator(BaseConfigurator):
    """Factory for :class:`BatchProcessor`."""

    logger_name = 'preprocessing_configurator'

    def create_batch_processor(self) -> Any:
        """
        Return a configured :class:`BatchProcessor`.

        ``BatchProcessor`` currently consumes a YAML path directly (it owns
        its own loader); the configurator only forwards the file path stored
        on the active config object.

        Raises:
            ValueError: if the active config does not carry a ``config_file``
                attribute (which means the config was constructed from a raw
                dict instead of ``BaseConfig.from_file``).
        """
        from habit.core.preprocessing.image_processor_pipeline import BatchProcessor

        config_file = getattr(self.config, 'config_file', None)
        if not config_file:
            raise ValueError(
                "BatchProcessor requires 'config_file' to be set on the "
                "configuration object. Load the config via "
                "PreprocessingConfig.from_file(path)."
            )
        return BatchProcessor(config_path=config_file, verbose=True)
