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
