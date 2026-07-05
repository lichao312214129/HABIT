"""
Validate nested step ``params`` blocks against registered *Params Pydantic models.

Used by workflow config schemas (phase 2) so YAML loading and GUI forms share
the same typed definitions in ``habit.core.schemas.steps``.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Literal, Optional

from pydantic import BaseModel, ValidationError

from habit.core.schemas.registry import Domain, ParamSchemaRegistry

# Keys injected by the pipeline / recipe builder — not part of step param schemas.
RUNTIME_STEP_KEYS: FrozenSet[str] = frozenset({"images", "before_z_score"})


def validate_step_params(
    domain: Domain,
    step_type: str,
    params: Dict[str, Any],
    *,
    preserve_keys: FrozenSet[str] = RUNTIME_STEP_KEYS,
) -> Dict[str, Any]:
    """
    Validate and coerce a parameter dict using the registered *Params model.

    Unknown registry keys (plugins without a registered schema) pass through unchanged.

    Args:
        domain: ``preprocessing``, ``feature_selection``, or ``model``.
        step_type: Registry step / method / model name.
        params: Raw parameter mapping (may include runtime keys in ``preserve_keys``).
        preserve_keys: Keys copied from ``params`` without schema validation.

    Returns:
        Dict[str, Any]: Validated mapping (defaults applied, types coerced).

    Raises:
        ValidationError: When ``params`` violate the registered schema.
    """
    ParamSchemaRegistry.ensure_initialized()
    model_cls: Optional[type[BaseModel]] = ParamSchemaRegistry.get(domain, step_type)
    if model_cls is None:
        return dict(params)

    preserved: Dict[str, Any] = {k: params[k] for k in preserve_keys if k in params}
    param_subset: Dict[str, Any] = {k: v for k, v in params.items() if k not in preserve_keys}

    validated_model = model_cls.model_validate(param_subset)
    merged: Dict[str, Any] = validated_model.model_dump(mode="python")
    merged.update(preserved)
    return merged


def try_validate_step_params(
    domain: Domain,
    step_type: str,
    params: Dict[str, Any],
    *,
    preserve_keys: FrozenSet[str] = RUNTIME_STEP_KEYS,
) -> tuple[Dict[str, Any], Optional[ValidationError]]:
    """
    Like :func:`validate_step_params` but returns validation errors instead of raising.

    Returns:
        tuple[Dict[str, Any], Optional[ValidationError]]: Coerced params and optional error.
    """
    try:
        return validate_step_params(domain, step_type, params, preserve_keys=preserve_keys), None
    except ValidationError as exc:
        return dict(params), exc
