"""
Class-imbalance resampling utilities for machine learning workflows.

This module exposes a single public function :func:`apply_resampling` that
implements random oversampling, random undersampling, and SMOTE.  Extracting
the logic here (away from :class:`BaseWorkflow`) has two benefits:

1. **Locality** – the algorithm lives in one place; adding a new method only
   requires editing this file.
2. **Testability** – the function is a pure transformation (no ``self``),
   so unit tests can call it directly with in-memory arrays without
   constructing a full workflow instance.

Supported methods (``sampling.method`` in the config):
    - ``random_over``   – random oversampling of the minority class.
    - ``random_under``  – random undersampling of the majority class.
    - ``smote``         – SMOTE oversampling (requires ``imbalanced-learn``).
"""

import logging
from typing import Any, Tuple

import numpy as np
import pandas as pd


class Resampler:
    """
    Adapter that turns the imperative :func:`apply_resampling` function into a
    small, testable object with a stable interface.

    The class deliberately wraps a single configuration plus its random seed
    and logger so that runners can inject one ``Resampler`` instance without
    knowing the specific resampling algorithm.  Two thin methods are exposed:

    * :meth:`resample` - return resampled (X, y); pure transformation.
    * :meth:`fit_with_resampling` - resample then fit a sklearn estimator.

    This replaces the previous ``BaseWorkflow._train_with_optional_sampling``
    private method and makes the seam explicit (the runner depends on a
    ``Resampler``-shaped object, not on the workflow internals).
    """

    def __init__(
        self,
        sampling_cfg: Any,
        random_state: int,
        logger: logging.Logger,
    ) -> None:
        self.sampling_cfg = sampling_cfg
        self.random_state = random_state
        self.logger = logger

    def resample(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Run the configured resampling step on training data only."""
        return apply_resampling(
            X_train=X_train,
            y_train=y_train,
            sampling_cfg=self.sampling_cfg,
            random_state=self.random_state,
            logger=self.logger,
        )

    def fit_with_resampling(
        self,
        estimator: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Any:
        """
        Resample the training set (when configured) and fit ``estimator``.

        Parameters
        ----------
        estimator:
            sklearn-compatible estimator/pipeline already constructed.
        X_train:
            Training features.
        y_train:
            Training labels aligned with ``X_train``.

        Returns
        -------
        Any
            The fitted estimator (the same object is returned for chaining).
        """
        X_fit, y_fit = self.resample(X_train, y_train)
        estimator.fit(X_fit, y_fit)
        return estimator


def apply_resampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_cfg: Any,
    random_state: int,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply class-imbalance resampling to training data.

    Only binary label distributions are supported.  If the label set has more
    than two distinct values the function logs a warning and returns the
    original data unchanged.

    Args:
        X_train: Training feature matrix.
        y_train: Training label series (must align with ``X_train`` by index).
        sampling_cfg: Pydantic sub-config object with at least the attributes
            ``enabled`` (bool), ``method`` (str), ``ratio`` (float), and
            optionally ``random_state`` (int).  Pass *None* to skip resampling.
        random_state: Fallback random seed used when ``sampling_cfg`` does not
            carry its own ``random_state`` attribute.
        logger: Logger instance for informational and warning messages.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Resampled (or original) training data.
            The returned objects are always *copies* of the inputs; the
            originals are never mutated.

    Raises:
        ValueError: If ``sampling.method`` is unrecognised or if
            ``random_under`` is used with ``ratio > 1.0``.
        ImportError: If ``smote`` method is requested but ``imbalanced-learn``
            is not installed.
    """
    if sampling_cfg is None or not getattr(sampling_cfg, "enabled", False):
        return X_train, y_train

    X_df: pd.DataFrame = X_train.copy()
    y_series: pd.Series = pd.Series(y_train).copy()
    y_series.index = X_df.index

    class_counts = y_series.value_counts()
    if class_counts.shape[0] != 2:
        logger.warning(
            "Resampling is designed for binary labels. "
            "Found %d classes; skipping.",
            class_counts.shape[0],
        )
        return X_df, y_series

    majority_label = class_counts.idxmax()
    minority_label = class_counts.idxmin()
    majority_count = int(class_counts[majority_label])
    minority_count = int(class_counts[minority_label])

    method = str(getattr(sampling_cfg, "method", "random_over")).strip().lower()
    ratio = float(getattr(sampling_cfg, "ratio", 1.0))
    seed = int(getattr(sampling_cfg, "random_state", random_state))

    logger.info(
        "Resampling: method=%s, ratio=%.4f, counts_before={%s: %d, %s: %d}",
        method,
        ratio,
        str(majority_label),
        majority_count,
        str(minority_label),
        minority_count,
    )

    if method == "random_over":
        target_minority_count = int(np.ceil(majority_count * ratio))
        if target_minority_count <= minority_count:
            logger.info("random_over skipped: minority count already >= target.")
            return X_df, y_series

        minority_indices = y_series[y_series == minority_label].index
        extra_indices = (
            pd.Series(minority_indices)
            .sample(n=target_minority_count - minority_count, replace=True, random_state=seed)
            .values
        )
        X_res = pd.concat([X_df, X_df.loc[extra_indices]], axis=0)
        y_res = pd.concat([y_series, y_series.loc[extra_indices]], axis=0)

    elif method == "random_under":
        if ratio > 1.0:
            raise ValueError("For random_under, sampling.ratio must be <= 1.0")

        target_majority_count = int(np.floor(minority_count / ratio))
        if target_majority_count >= majority_count:
            logger.info("random_under skipped: majority count already <= target.")
            return X_df, y_series

        majority_indices = y_series[y_series == majority_label].index
        minority_indices = y_series[y_series == minority_label].index
        kept_majority = (
            pd.Series(majority_indices)
            .sample(n=target_majority_count, replace=False, random_state=seed)
            .values
        )
        kept_indices = np.concatenate([kept_majority, minority_indices.values])
        X_res = X_df.loc[kept_indices]
        y_res = y_series.loc[kept_indices]

    elif method == "smote":
        try:
            from imblearn.over_sampling import SMOTE  # type: ignore
        except Exception as exc:
            raise ImportError(
                "SMOTE requires imbalanced-learn. "
                "Install with `pip install imbalanced-learn`."
            ) from exc

        smote = SMOTE(sampling_strategy=ratio, random_state=seed)
        X_sm, y_sm = smote.fit_resample(X_df, y_series)
        X_res = (
            pd.DataFrame(X_sm, columns=X_df.columns)
            if not isinstance(X_sm, pd.DataFrame)
            else X_sm
        )
        y_res = pd.Series(y_sm)

    else:
        raise ValueError(f"Unsupported resampling method: '{method}'")

    # Shuffle after resampling to prevent ordering bias in downstream training.
    permutation = (
        pd.Series(range(len(y_res)))
        .sample(frac=1.0, random_state=seed)
        .values
    )
    X_res = X_res.reset_index(drop=True).iloc[permutation].reset_index(drop=True)
    y_res = y_res.reset_index(drop=True).iloc[permutation].reset_index(drop=True)

    res_counts = y_res.value_counts()
    logger.info(
        "Resampling done: counts_after={%s: %d, %s: %d}",
        str(majority_label),
        int(res_counts.get(majority_label, 0)),
        str(minority_label),
        int(res_counts.get(minority_label, 0)),
    )
    return X_res, y_res
