"""
Value-only (numpy) preprocessing helpers.

These low-level utilities are used internally by registered DataFrame handlers
and the unified preprocessing pipeline. External callers should prefer
:func:`pipeline.apply_stateless_preprocessing` or :class:`PreprocessingState`.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Union, Literal

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from pydantic import BaseModel


def _get_method_attr(
    method_config: Union[Dict[str, Any], BaseModel],
    attr: str,
    default: Any = None,
) -> Any:
    if hasattr(method_config, attr):
        try:
            value = getattr(method_config, attr)
            return value if value is not None else default
        except (AttributeError, TypeError):
            pass

    if isinstance(method_config, dict):
        return method_config.get(attr, default)
    elif hasattr(method_config, "get") and callable(getattr(method_config, "get")):
        return method_config.get(attr, default)

    return default


def handle_extreme_values(
    features: np.ndarray, strategy: str = "mean_replacement"
) -> np.ndarray:
    features_clean = features.copy()

    if np.any(np.isinf(features_clean)):
        for col in range(features_clean.shape[1]):
            col_data = features_clean[:, col]
            mask_inf = np.isinf(col_data)

            if np.all(mask_inf):
                features_clean[:, col] = 0
            elif np.any(mask_inf):
                valid_data = col_data[~mask_inf]
                if strategy == "mean_replacement":
                    replace_value = np.mean(valid_data)
                elif strategy == "median_replacement":
                    replace_value = np.median(valid_data)
                else:
                    replace_value = 0

                features_clean[mask_inf, col] = replace_value

    if np.any(np.isnan(features_clean)):
        for col in range(features_clean.shape[1]):
            col_data = features_clean[:, col]
            mask_nan = np.isnan(col_data)

            if np.all(mask_nan):
                features_clean[:, col] = 0
            elif np.any(mask_nan):
                valid_data = col_data[~mask_nan]
                if strategy == "mean_replacement":
                    replace_value = np.mean(valid_data)
                elif strategy == "median_replacement":
                    replace_value = np.median(valid_data)
                else:
                    replace_value = 0

                features_clean[mask_nan, col] = replace_value

    return features_clean


def create_discretizer(
    n_bins: int = 10, bin_strategy: str = "uniform"
) -> KBinsDiscretizer:
    return KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy=bin_strategy
    )


def process_features_pipeline(
    features: np.ndarray,
    methods: List[Union[Dict[str, Any], BaseModel]],
) -> np.ndarray:
    processed_features = features.copy()

    for method_config in methods:
        method_name = _get_method_attr(method_config, "method", "minmax")

        if isinstance(method_config, dict):
            config = method_config.copy()
            if "method" in config:
                del config["method"]
        else:
            config = method_config.model_dump(exclude={"method"})
            config = {k: v for k, v in config.items() if v is not None}

        processed_features = preprocess_features(
            processed_features, method=method_name, **config
        )

    return processed_features


_VALUE_METHODS = frozenset({
    "minmax",
    "zscore",
    "robust",
    "binning",
    "global_minmax",
    "global_zscore",
    "winsorize",
    "log",
})


def preprocess_features(
    features: np.ndarray,
    method: Literal[
        "minmax",
        "zscore",
        "robust",
        "binning",
        "global_minmax",
        "global_zscore",
        "winsorize",
        "log",
    ] = "minmax",
    n_bins: int = 10,
    bin_strategy: str = "uniform",
    global_normalize: bool = False,
    winsor_limits: tuple = (0.05, 0.05),
    discretizer: Optional[KBinsDiscretizer] = None,
    methods: Optional[List[Dict[str, Any]]] = None,
) -> np.ndarray:
    if methods is not None:
        return process_features_pipeline(features, methods)

    features = handle_extreme_values(features)

    if method == "minmax":
        if global_normalize:
            min_val = np.min(features)
            max_val = np.max(features)
            return (features - min_val) / (max_val - min_val + 1e-6)
        else:
            min_vals = np.min(features, axis=0)
            max_vals = np.max(features, axis=0)
            return (features - min_vals) / (max_vals - min_vals + 1e-6)

    elif method == "zscore":
        if global_normalize:
            mean = np.mean(features)
            std = np.std(features)
            return (features - mean) / (std + 1e-6)
        else:
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            return (features - mean) / (std + 1e-6)

    elif method == "robust":
        if global_normalize:
            q1 = np.percentile(features.flatten(), 25)
            q3 = np.percentile(features.flatten(), 75)
            iqr = q3 - q1
            median = np.median(features.flatten())
            return (features - median) / (iqr + 1e-6)
        else:
            q1 = np.percentile(features, 25, axis=0)
            q3 = np.percentile(features, 75, axis=0)
            iqr = q3 - q1
            median = np.median(features, axis=0)
            return (features - median) / (iqr + 1e-6)

    elif method == "binning":
        if global_normalize:
            original_shape = features.shape
            flattened = features.flatten().reshape(-1, 1)
            if discretizer is None:
                discretizer = KBinsDiscretizer(
                    n_bins=n_bins, encode="ordinal", strategy=bin_strategy
                )
                binned_flattened = discretizer.fit_transform(flattened)
            else:
                binned_flattened = discretizer.transform(flattened)
            binned_features = binned_flattened.reshape(original_shape)
        else:
            if discretizer is None:
                discretizer = KBinsDiscretizer(
                    n_bins=n_bins, encode="ordinal", strategy=bin_strategy
                )
                binned_features = discretizer.fit_transform(features)
            else:
                binned_features = discretizer.transform(features)
        return binned_features

    elif method == "winsorize":
        if global_normalize:
            flattened = features.flatten()
            lower = np.percentile(flattened, winsor_limits[0] * 100)
            upper = np.percentile(flattened, (1 - winsor_limits[1]) * 100)
            return np.clip(features, lower, upper)
        else:
            lower = np.percentile(features, winsor_limits[0] * 100, axis=0)
            upper = np.percentile(
                features, (1 - winsor_limits[1]) * 100, axis=0
            )
            return np.clip(features, lower, upper)

    elif method == "log":
        if global_normalize:
            min_val = np.min(features)
            shifted_features = features - min_val + 1.0
            return np.log(shifted_features)
        else:
            min_vals = np.min(features, axis=0)
            shifted_features = features - min_vals + 1.0
            return np.log(shifted_features)

    else:
        raise ValueError(f"Unknown preprocessing method: {method}")
