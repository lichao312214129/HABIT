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
Preprocessing State Management for Habitat Analysis.

Group-level preprocessing uses :func:`pipeline.apply_preprocessing_pipeline` with
``fit=True`` to learn baseline statistics and per-step handler state, then replays
the same registered handlers at predict time.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel

from habit.utils.log_utils import get_module_logger

from .base_preprocessing import BaselineStats
from .pipeline import apply_preprocessing_pipeline


class PreprocessingState:
    """
    Manages state for group-level preprocessing operations.

    Supports capturing parameters during training and applying them during testing.
    All methods are dispatched through ``PreprocessingMethodFactory``.
    """

    def __init__(self) -> None:
        self.baseline: Optional[BaselineStats] = None
        self.methods_config: List[Union[Dict[str, Any], BaseModel]] = []
        self.step_states: List[Any] = []

    def fit(self, df: pd.DataFrame, methods: List[Union[Dict[str, Any], BaseModel]]) -> None:
        """
        Learn baseline and per-step preprocessing state from training data.

        Args:
            df: Combined cohort feature table (may include metadata columns).
            methods: Ordered preprocessing step configurations.
        """
        logger = get_module_logger(__name__)
        logger.info(
            f"PreprocessingState.fit() input shape={df.shape}, "
            f"dtypes={df.dtypes.value_counts().to_dict()}"
        )
        logger.debug(f"Input columns sample: {list(df.columns)[:10]}")

        if not methods:
            self.methods_config = []
            self.step_states = []
            self.baseline = None
            return

        output_df, baseline, step_states = apply_preprocessing_pipeline(
            df,
            methods,
            fit=True,
        )
        del output_df  # fit path transforms in-memory; callers use transform() for outputs

        if baseline is None:
            raise ValueError("No numeric feature columns found in DataFrame. Cannot perform preprocessing.")

        self.methods_config = methods
        self.baseline = baseline
        self.step_states = step_states

        logger.info(
            f"PreprocessingState.fit() completed with {len(step_states)} step state(s), "
            f"baseline feature count={len(baseline.means)}"
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted preprocessing to new data.

        Args:
            df: Input table with the same metadata / feature layout as training.

        Returns:
            Preprocessed DataFrame.

        Raises:
            ValueError: If :meth:`fit` has not been called yet.
        """
        if self.baseline is None:
            raise ValueError("PreprocessingState has not been fitted. Call fit() first.")

        if not self.methods_config:
            return df.copy()

        output_df, _, _ = apply_preprocessing_pipeline(
            df,
            self.methods_config,
            step_states=self.step_states,
            baseline=self.baseline,
            fit=False,
        )
        return output_df

    def save(self, output_dir: str, filename: str = "preprocessing_state.pkl") -> None:
        """
        Persist preprocessing state to disk.

        Args:
            output_dir: Directory that will receive the pickle file.
            filename: Pickle file name.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, "wb") as file_handle:
            pickle.dump(self, file_handle)

    @classmethod
    def load(cls, output_dir: str, filename: str = "preprocessing_state.pkl") -> "PreprocessingState":
        """
        Load preprocessing state from disk.

        Args:
            output_dir: Directory containing the pickle file.
            filename: Pickle file name.

        Returns:
            Loaded ``PreprocessingState`` instance.
        """
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessing state file not found at {path}")

        with open(path, "rb") as file_handle:
            return pickle.load(file_handle)
