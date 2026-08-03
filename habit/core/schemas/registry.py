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
Central registry for HABIT step parameter Pydantic models.

Plugin authors register new steps with::

    ParamSchemaRegistry.register("preprocessing", "my_step", MyStepParams)
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Type

from pydantic import BaseModel

from habit.core.schemas.steps.feature_selection import FEATURE_SELECTION_PARAM_MODELS
from habit.core.schemas.steps.ml_models import MODEL_PARAM_MODELS
from habit.core.schemas.steps.preprocessing import PREPROCESSING_PARAM_MODELS

Domain = Literal["preprocessing", "feature_selection", "model"]

# Workflow-level Pydantic models (whole YAML files / GUI workflow pages).
WORKFLOW_SCHEMA_IDS: tuple[str, ...] = (
    "dicom_sort",
    "preprocess",
    "ml",
    "compare",
    "habitat",
    "extract",
)


class ParamSchemaRegistry:
    """
    Maps (domain, step_type) to a Pydantic *Params model.

    Built-in models are registered at import time; plugins add entries via
    :meth:`register`.
    """

    _models: Dict[str, Dict[str, Type[BaseModel]]] = {
        "preprocessing": {},
        "feature_selection": {},
        "model": {},
    }
    _initialized: bool = False

    @classmethod
    def register(cls, domain: Domain, step_type: str, model: Type[BaseModel]) -> None:
        """
        Register a params schema for one registry step.

        Args:
            domain: ``preprocessing``, ``feature_selection``, or ``model``.
            step_type: Factory / selector / model registry key.
            model: Pydantic model describing user-configurable parameters.
        """
        cls._models.setdefault(domain, {})[step_type] = model

    @classmethod
    def get(cls, domain: Domain, step_type: str) -> Optional[Type[BaseModel]]:
        """
        Look up a registered params model.

        Args:
            domain: Registry domain.
            step_type: Step or model name.

        Returns:
            Optional[Type[BaseModel]]: Params model or None if not registered.
        """
        cls.ensure_initialized()
        return cls._models.get(domain, {}).get(step_type)

    @classmethod
    def ensure_initialized(cls) -> None:
        """Load built-in param models and wire factory registries once."""
        if cls._initialized:
            return
        for step_type, model in PREPROCESSING_PARAM_MODELS.items():
            cls.register("preprocessing", step_type, model)
        for step_type, model in FEATURE_SELECTION_PARAM_MODELS.items():
            cls.register("feature_selection", step_type, model)
        for step_type, model in MODEL_PARAM_MODELS.items():
            cls.register("model", step_type, model)
        cls._wire_factories()
        cls._initialized = True

    @classmethod
    def _wire_factories(cls) -> None:
        """Attach params models to core factories for introspection by plugins."""
        try:
            from habit.core.preprocessing.preprocessor_factory import PreprocessorFactory

            for step_type, model in cls._models.get("preprocessing", {}).items():
                PreprocessorFactory.register_params_model(step_type, model)
        except Exception:
            pass

        try:
            from habit.core.machine_learning.feature_selectors.selector_registry import (
                SelectorRegistry,
            )

            for step_type, model in cls._models.get("feature_selection", {}).items():
                SelectorRegistry.register_params_model(step_type, model)
        except Exception:
            pass

        try:
            from habit.core.machine_learning.models.factory import ModelFactory

            for step_type, model in cls._models.get("model", {}).items():
                ModelFactory.register_params_model(step_type, model)
        except Exception:
            pass

    @classmethod
    def all_step_types(cls, domain: Domain) -> list[str]:
        """Return registered step types for a domain."""
        cls.ensure_initialized()
        return sorted(cls._models.get(domain, {}).keys())
