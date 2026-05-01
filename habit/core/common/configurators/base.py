"""
Base configurator: shared infrastructure for domain-specific configurators.

V1 architecture: ``ServiceConfigurator`` has been split per business
domain (Habitat / ML / Preprocessing). Each domain configurator extends
:class:`BaseConfigurator` and adds its own ``create_*`` factories.

This base class owns:
    * the configuration object,
    * the output directory derivation,
    * the logger creation policy (reuses the CLI-level logger if one is
      already set up via :class:`LoggerManager`),
    * a small per-instance service cache.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from habit.utils.log_utils import LoggerManager, get_module_logger, setup_logger


class BaseConfigurator:
    """
    Shared infrastructure for domain configurators.

    Subclasses should override :attr:`logger_name` and add their own
    ``create_*`` factory methods. They MUST NOT bring in transitive imports
    from other business domains at module top level — defer those imports
    inside the factory methods to keep the import surface small.
    """

    #: Default logger name; subclasses override.
    logger_name: str = 'configurator'

    def __init__(
        self,
        config: Any,
        logger: Optional[Any] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            config: Validated config object (Pydantic ``BaseConfig`` subclass)
                or a plain dict that the subclass knows how to validate.
            logger: Optional pre-built logger (CLI usually passes one in).
            output_dir: Optional explicit output directory. Falls back to
                ``config.output_dir`` / ``config.out_dir`` / ``config.output``
                / ``./output``.
        """
        self.config: Any = config
        self.output_dir: str = (
            output_dir
            or getattr(config, 'output_dir', None)
            or getattr(config, 'out_dir', None)
            or getattr(config, 'output', None)
            or './output'
        )
        self.logger: Any = logger or self._create_logger()
        self._services: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infrastructure helpers
    # ------------------------------------------------------------------

    def _create_logger(self) -> Any:
        """Return a logger; reuses the CLI-level configuration if available."""
        manager = LoggerManager()
        if manager.get_log_file() is not None:
            logger = get_module_logger(self.logger_name)
            logger.info(
                "Using existing logging configuration from CLI entry point"
            )
            return logger
        return setup_logger(
            name=self.logger_name,
            output_dir=self.output_dir,
            log_filename=f'{self.logger_name}.log',
        )

    def _ensure_output_dir(self) -> str:
        """Create ``self.output_dir`` if missing and return it."""
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir

    # ------------------------------------------------------------------
    # Service cache
    # ------------------------------------------------------------------

    def get_service(self, name: str) -> Any:
        """
        Return a previously cached service instance.

        Args:
            name: Service key.

        Returns:
            The cached service instance.

        Raises:
            ValueError: if no service with ``name`` has been registered.
        """
        if name not in self._services:
            raise ValueError(
                f"Service '{name}' not found. Available: {list(self._services.keys())}"
            )
        return self._services[name]

    def register_service(self, name: str, service: Any) -> None:
        """Cache a service instance under ``name``."""
        self._services[name] = service

    def clear_cache(self) -> None:
        """Drop every cached service."""
        self._services.clear()
