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

        ``BatchProcessor`` accepts a validated :class:`PreprocessingConfig`
        object directly so YAML is not loaded twice.

        Returns:
            Configured :class:`BatchProcessor`.

        Raises:
            ValueError: if ``self.config`` is missing.
        """
        from habit.core.preprocessing.image_processor_pipeline import BatchProcessor

        if self.config is None:
            raise ValueError("PreprocessingConfigurator requires a config object.")
        return BatchProcessor(config=self.config, verbose=True)
