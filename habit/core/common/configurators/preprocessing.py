"""
PreprocessingConfigurator: factory for the image preprocessing pipeline.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import BaseConfigurator


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
