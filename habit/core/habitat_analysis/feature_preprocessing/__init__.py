"""
Feature-preprocessing algorithms used between feature extraction and clustering.

These are *column-dropping* operations (they change the DataFrame shape),
which is why they cannot be expressed through the value-only
``process_features_pipeline`` utility used for scaling / discretisation.

Each algorithm comes in two forms:

* ``select_*_columns``  -> returns the surviving column names. Useful when
  a stateful preprocessor must cache the column list at fit time and
  replay it at predict time.
* ``apply_*_filter``    -> returns the filtered DataFrame directly.

Both forms share a single source of truth so the algorithm cannot drift
between the subject-level and group-level code paths.
"""
from .variance_filter import apply_variance_filter, select_variance_columns
from .correlation_filter import apply_correlation_filter, select_correlation_columns

__all__ = [
    "apply_variance_filter",
    "apply_correlation_filter",
    "select_variance_columns",
    "select_correlation_columns",
]
