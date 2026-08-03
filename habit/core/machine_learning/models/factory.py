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
Model Factory
Factory class for creating model instances
"""
from typing import Dict, Any, List
import importlib
import os

from habit.core.common.registry import ClassRegistry
from .base import BaseModel
from habit.utils.log_utils import get_module_logger

LOGGER = get_module_logger("ml.model_factory")


class ModelFactory(ClassRegistry[BaseModel]):
    """
    Factory for creating model instances.

    Uses the shared :class:`~habit.core.common.registry.ClassRegistry` contract
    (``register`` / ``create`` / ``get`` / ``available`` /
    ``register_params_model`` / ``get_params_model``) and adds lazy discovery of
    every ``models/*.py`` module so decorated models self-register on demand.

    ``create`` keeps the ML-specific convention of passing a single positional
    ``config`` dict to the model constructor.
    """

    kind = "model"

    @classmethod
    def create(cls, model_name: str, config: Dict[str, Any] = None) -> BaseModel:
        """
        Create a model instance by name.

        Args:
            model_name: Registered model name.
            config: Model configuration. Either the nested wrapper contract
                ``{'params': {...}}`` or a flat mapping of parameters, which is
                normalized by :meth:`_normalize_config`.

        Returns:
            BaseModel: Instantiated model.

        Raises:
            ValueError: If ``model_name`` is not registered (after discovery).
        """
        model_cls = cls.get(model_name)
        if model_cls is None:
            raise ValueError(
                f"Model '{model_name}' not registered. "
                f"Available models: {list(cls._registry.keys())}"
            )
        return model_cls(cls._normalize_config(config or {}))

    @staticmethod
    def _normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a model config to the nested ``{'params': {...}}`` contract.

        Model wrappers read their hyperparameters from ``config['params']``.
        Callers, however, legitimately hold a flat parameter mapping: the ML
        config schema stores hyperparameters under ``ModelConfig.params``, and
        both runners hand that mapping straight to
        :meth:`PipelineBuilder.build`. Wrapping it here keeps every entry point
        — pipeline, public API, plugins — on one contract, so a flat mapping can
        never again be silently read as "no parameters configured".

        Args:
            config: Nested or flat model configuration.

        Returns:
            Dict[str, Any]: Configuration guaranteed to expose a ``params`` key.
        """
        if isinstance(config.get('params'), dict):
            return config
        return {'params': dict(config)}

    @classmethod
    def available(cls) -> List[str]:
        """
        Return all model names, importing every model module first.

        Note:
            This eagerly imports optional heavy models (e.g. AutoGluon) and can
            be slow; GUI paths that only need built-ins import them explicitly.
        """
        cls._discover()
        return list(cls._registry.keys())

    @classmethod
    def _discover(cls) -> None:
        """Dynamically import all model modules so decorated models register."""
        models_dir = os.path.dirname(os.path.abspath(__file__))
        for filename in os.listdir(models_dir):
            if filename.endswith('.py') and not filename.startswith('__') and filename != 'factory.py':
                module_name = filename[:-3]
                try:
                    importlib.import_module(f".{module_name}", package="habit.core.machine_learning.models")
                    LOGGER.debug("Successfully imported model module: %s", module_name)
                except ImportError as e:
                    LOGGER.warning("Failed to import model module %s: %s", module_name, e)