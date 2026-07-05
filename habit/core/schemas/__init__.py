"""
HABIT configuration schemas — unified entry point.

Layout
------
workflows/
    Whole YAML configs (``PreprocessingConfig``, ``MLConfig``, …).
steps/
    Per-step parameter models (``ResampleParams``, ``LassoParams``, …).
registry.py
    :class:`ParamSchemaRegistry` — maps step names to *Params models.
validation.py
    :func:`validate_step_params` — coerce nested ``params`` dicts at load time.
reflect.py
    Pydantic → GUI JSON field descriptors.
field_reflect.py
    Low-level Pydantic field reflection (:class:`FieldDescriptor`).
"""

from habit.core.schemas import steps, workflows
from habit.core.schemas.reflect import (
    AUTO_FILLED_FIELD_NAMES,
    descriptor_to_json,
    reflect_param_model,
)
from habit.core.schemas.registry import Domain, ParamSchemaRegistry, WORKFLOW_SCHEMA_IDS
from habit.core.schemas.validation import RUNTIME_STEP_KEYS, validate_step_params

# Register built-in step schemas on import.
ParamSchemaRegistry.ensure_initialized()

__all__ = [
    "AUTO_FILLED_FIELD_NAMES",
    "Domain",
    "ParamSchemaRegistry",
    "RUNTIME_STEP_KEYS",
    "WORKFLOW_SCHEMA_IDS",
    "descriptor_to_json",
    "reflect_param_model",
    "steps",
    "validate_step_params",
    "workflows",
]
