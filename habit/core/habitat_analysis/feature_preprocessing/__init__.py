"""
Feature-preprocessing package for habitat analysis.

This package houses **all** preprocessing logic — both stateless algorithms
and the stateful :class:`PreprocessingState` manager used for group-level
train/test consistency.

Column-dropping operations (variance / correlation filter) change the
DataFrame shape, which is why they cannot be expressed through the
value-only :func:`process_features_pipeline` utility used for scaling /
discretisation.

Each column-dropping algorithm comes in two forms:

* ``select_*_columns``  -> returns the surviving column names. Useful when
  a stateful preprocessor must cache the column list at fit time and
  replay it at predict time.
* ``apply_*_filter``    -> returns the filtered DataFrame directly.

Both forms share a single source of truth so the algorithm cannot drift
between the subject-level and group-level code paths.

Value-only transforms (minmax, zscore, robust, binning, winsorize, log)
live in :mod:`value_transforms` and operate on ``np.ndarray``.

Stateful group-level preprocessing (fit/transform with parameter caching)
lives in :mod:`preprocessing_state`.
"""
from .correlation_filter import apply_correlation_filter, select_correlation_columns
from .frame_method_handlers import (
    FRAME_LEVEL_METHOD_NAMES,
    FRAME_METHOD_HANDLERS,
    apply_registered_frame_method,
    resolve_correlation_filter_params,
    resolve_variance_threshold,
)
from .value_transforms import (
    handle_extreme_values,
    create_discretizer,
    preprocess_features,
    process_features_pipeline,
)
from .variance_filter import apply_variance_filter, select_variance_columns


def __getattr__(name: str):
    if name == "PreprocessingState":
        from .preprocessing_state import PreprocessingState
        return PreprocessingState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FRAME_LEVEL_METHOD_NAMES",
    "FRAME_METHOD_HANDLERS",
    "PreprocessingState",
    "apply_correlation_filter",
    "apply_registered_frame_method",
    "apply_variance_filter",
    "create_discretizer",
    "handle_extreme_values",
    "preprocess_features",
    "process_features_pipeline",
    "resolve_correlation_filter_params",
    "resolve_variance_threshold",
    "select_correlation_columns",
    "select_variance_columns",
]
