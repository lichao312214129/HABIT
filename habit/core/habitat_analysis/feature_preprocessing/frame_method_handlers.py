"""
Registry for DataFrame-level preprocessing steps ("frame methods").

These operations drop or reorder **columns**, so they cannot run through the
value-only :func:`~habit.core.habitat_analysis.feature_preprocessing.value_transforms.process_features_pipeline`
path, which operates on ``numpy.ndarray`` without column identity.

Adding a new frame-level method:

1. Implement ``apply_*`` / ``select_*`` in a dedicated module under this package.
2. Register a handler in ``FRAME_METHOD_HANDLERS`` keyed by the config ``method`` string.
3. Add that string to ``FRAME_LEVEL_METHOD_NAMES`` (also drives
   ``config_schemas.DROPPING_PREPROCESSING_METHODS``).
4. Extend ``PreprocessingMethod.method`` literal in ``config_schemas`` if needed.
5. For group-level stateful replay, wire ``PreprocessingState.fit`` if the method
   must cache selected column names (see variance / correlation filters).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel

from .correlation_filter import apply_correlation_filter
from .variance_filter import apply_variance_filter

# Methods that change which feature columns exist; kept in sync with pydantic
# validation (two-step subject-level guardrail) via config_schemas import.
FRAME_LEVEL_METHOD_NAMES: FrozenSet[str] = frozenset({
    "variance_filter",
    "correlation_filter",
})

FrameMethodHandler = Callable[
    [pd.DataFrame, Union[Dict[str, Any], BaseModel]],
    pd.DataFrame,
]


def _read_method_field(
    method_config: Union[Dict[str, Any], BaseModel],
    attr: str,
    default: Any = None,
) -> Any:
    """
    Read one field from either a dict config or a Pydantic model.

    Semantics match ``preprocessing_state._get_method_attr`` so YAML/dict and
    ``PreprocessingMethod`` configs behave identically everywhere.

    Args:
        method_config: Preprocessing step configuration object.
        attr: Field name (e.g. ``"variance_threshold"``).
        default: Value used when the attribute is missing or explicitly ``None``.

    Returns:
        Resolved field value or ``default``.
    """
    if hasattr(method_config, attr):
        try:
            value = getattr(method_config, attr)
            return value if value is not None else default
        except (AttributeError, TypeError):
            pass

    if isinstance(method_config, dict):
        return method_config.get(attr, default)
    if hasattr(method_config, "get") and callable(getattr(method_config, "get")):
        return method_config.get(attr, default)

    return default


def resolve_variance_threshold(
    method_config: Union[Dict[str, Any], BaseModel],
) -> float:
    """
    Parse variance-filter threshold from a preprocessing method config.

    Args:
        method_config: Declarative config for one preprocessing step.

    Returns:
        Numeric threshold passed to :func:`apply_variance_filter`.
    """
    raw = _read_method_field(method_config, "variance_threshold", None)
    return float(raw) if raw is not None else 0.0


def resolve_correlation_filter_params(
    method_config: Union[Dict[str, Any], BaseModel],
) -> Tuple[float, str]:
    """
    Parse correlation-filter settings from a preprocessing method config.

    Args:
        method_config: Declarative config for one preprocessing step.

    Returns:
        ``(corr_threshold, corr_method)`` for :func:`apply_correlation_filter`.
    """
    thresh_raw = _read_method_field(method_config, "corr_threshold", None)
    threshold = float(thresh_raw) if thresh_raw is not None else 0.95
    corr_method = _read_method_field(method_config, "corr_method", None) or "spearman"
    return threshold, str(corr_method)


def _handle_variance_filter(
    df: pd.DataFrame,
    method_config: Union[Dict[str, Any], BaseModel],
) -> pd.DataFrame:
    return apply_variance_filter(df, resolve_variance_threshold(method_config))


def _handle_correlation_filter(
    df: pd.DataFrame,
    method_config: Union[Dict[str, Any], BaseModel],
) -> pd.DataFrame:
    threshold, corr_method = resolve_correlation_filter_params(method_config)
    return apply_correlation_filter(df, threshold, corr_method)


FRAME_METHOD_HANDLERS: Dict[str, FrameMethodHandler] = {
    "variance_filter": _handle_variance_filter,
    "correlation_filter": _handle_correlation_filter,
}

if set(FRAME_METHOD_HANDLERS.keys()) != set(FRAME_LEVEL_METHOD_NAMES):
    raise RuntimeError(
        "FRAME_METHOD_HANDLERS keys must match FRAME_LEVEL_METHOD_NAMES; "
        f"handlers={sorted(FRAME_METHOD_HANDLERS)!s}, "
        f"registry={sorted(FRAME_LEVEL_METHOD_NAMES)!s}"
    )


def apply_registered_frame_method(
    df: pd.DataFrame,
    method_config: Union[Dict[str, Any], BaseModel],
) -> Optional[pd.DataFrame]:
    """
    If ``method_config`` refers to a registered frame-level method, apply it.

    Args:
        df: Feature matrix to transform.
        method_config: Single step from ``PreprocessingConfig.methods``.

    Returns:
        Transformed DataFrame when a handler exists; ``None`` when the step
        should be handled by the value-only pipeline instead.
    """
    method_name = _read_method_field(method_config, "method", None)
    if method_name is None:
        return None
    handler = FRAME_METHOD_HANDLERS.get(method_name)
    if handler is None:
        return None
    return handler(df, method_config)
