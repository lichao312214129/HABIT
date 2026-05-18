"""
Preprocessing State Management for Habitat Analysis.

This module provides **stateful** preprocessing (group-level operations with
train/test separation). The stateless algorithms it delegates to live in
sibling modules within this package.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from sklearn.preprocessing import KBinsDiscretizer
from pydantic import BaseModel
from habit.utils.log_utils import get_module_logger

from ..config_schemas import ResultColumns
from .correlation_filter import select_correlation_columns
from .frame_method_handlers import (
    FRAME_LEVEL_METHOD_NAMES,
    resolve_correlation_filter_params,
    resolve_variance_threshold,
)
from .value_transforms import create_discretizer, handle_extreme_values
from .variance_filter import select_variance_columns


def _get_method_attr(method_config: Union[Dict[str, Any], BaseModel], attr: str, default: Any = None) -> Any:
    if hasattr(method_config, attr):
        try:
            value = getattr(method_config, attr)
            return value if value is not None else default
        except (AttributeError, TypeError):
            pass

    if isinstance(method_config, dict):
        return method_config.get(attr, default)
    elif hasattr(method_config, 'get') and callable(getattr(method_config, 'get')):
        return method_config.get(attr, default)

    return default


class PreprocessingState:
    """
    Manages state for group-level preprocessing operations.

    Supports capturing parameters during training and applying them during testing.
    Leverages shared utility functions from sibling modules in this package.
    """

    def __init__(self):
        self.means: Optional[pd.Series] = None
        self.stds: Optional[pd.Series] = None
        self.mins: Optional[pd.Series] = None
        self.maxs: Optional[pd.Series] = None

        self.medians: Optional[pd.Series] = None
        self.q1s: Optional[pd.Series] = None
        self.q3s: Optional[pd.Series] = None

        self.discretizers: Dict[str, KBinsDiscretizer] = {}

        self.winsor_lowers: Optional[pd.Series] = None
        self.winsor_uppers: Optional[pd.Series] = None

        self.log_offsets: Optional[pd.Series] = None

        self.global_params: Dict[str, Any] = {}

        self.methods_config: List[Union[Dict[str, Any], BaseModel]] = []
        self.selected_columns_by_step: Dict[int, List[str]] = {}

    def fit(self, df: pd.DataFrame, methods: List[Union[Dict[str, Any], BaseModel]]) -> None:
        self.methods_config = methods
        self.selected_columns_by_step = {}

        logger = get_module_logger(__name__)
        logger.info(
            f"PreprocessingState.fit() input shape={df.shape}, "
            f"dtypes={df.dtypes.value_counts().to_dict()}"
        )
        logger.debug(f"Input columns sample: {list(df.columns)[:10]}")

        numeric_df = self._get_numeric_columns(df)
        logger.info(f"PreprocessingState.fit() numeric feature columns: {len(numeric_df.columns)}")
        logger.debug(f"Sample numeric columns: {list(numeric_df.columns)[:5]}")

        if numeric_df.empty:
            logger.error(f"All DataFrame columns: {list(df.columns)}")
            logger.error(f"DataFrame dtypes:\n{df.dtypes}")
            raise ValueError("No numeric columns found in DataFrame. Cannot perform preprocessing.")

        self.means = numeric_df.mean()
        self.stds = numeric_df.std().replace(0, 1.0)
        self.mins = numeric_df.min()
        self.maxs = numeric_df.max()

        temp_df = numeric_df.fillna(self.means)

        for step_index, method_config in enumerate(methods):
            method_name = _get_method_attr(method_config, 'method', 'minmax')
            global_normalize = _get_method_attr(method_config, 'global_normalize', False)

            if method_name in ['z_score', 'zscore', 'standardization']:
                if global_normalize:
                    self.global_params['zscore'] = {
                        'mean': temp_df.values.mean(),
                        'std': temp_df.values.std() if temp_df.values.std() != 0 else 1.0
                    }

            elif method_name in ['min_max', 'minmax', 'normalize']:
                if global_normalize:
                    self.global_params['minmax'] = {
                        'min': temp_df.values.min(),
                        'max': temp_df.values.max()
                    }

            elif method_name == 'robust':
                if global_normalize:
                    flat_values = temp_df.values.flatten()
                    self.global_params['robust'] = {
                        'median': np.median(flat_values),
                        'q1': np.percentile(flat_values, 25),
                        'q3': np.percentile(flat_values, 75)
                    }
                else:
                    self.medians = temp_df.median()
                    self.q1s = temp_df.quantile(0.25)
                    self.q3s = temp_df.quantile(0.75)

            elif method_name == 'binning':
                n_bins = _get_method_attr(method_config, 'n_bins', 10)
                bin_strategy = _get_method_attr(method_config, 'bin_strategy', 'uniform')

                if global_normalize:
                    discretizer = create_discretizer(n_bins, bin_strategy)
                    flat_values = temp_df.values.flatten().reshape(-1, 1)
                    discretizer.fit(flat_values)
                    self.discretizers['global'] = discretizer
                else:
                    discretizer = create_discretizer(n_bins, bin_strategy)
                    discretizer.fit(temp_df.values)
                    self.discretizers['per_feature'] = discretizer

            elif method_name == 'winsorize':
                winsor_limits_raw = _get_method_attr(method_config, 'winsor_limits', None)
                if winsor_limits_raw is None:
                    winsor_limits = (0.05, 0.05)
                elif isinstance(winsor_limits_raw, list):
                    winsor_limits = tuple(winsor_limits_raw)
                else:
                    winsor_limits = winsor_limits_raw

                if global_normalize:
                    flat_values = temp_df.values.flatten()
                    self.global_params['winsorize'] = {
                        'lower': np.percentile(flat_values, winsor_limits[0] * 100),
                        'upper': np.percentile(flat_values, (1 - winsor_limits[1]) * 100)
                    }
                else:
                    self.winsor_lowers = temp_df.quantile(winsor_limits[0])
                    self.winsor_uppers = temp_df.quantile(1 - winsor_limits[1])

            elif method_name == 'log':
                if global_normalize:
                    self.global_params['log_offset'] = temp_df.values.min()
                else:
                    self.log_offsets = temp_df.min()

            elif method_name == 'variance_filter':
                variance_threshold = resolve_variance_threshold(method_config)
                selected_cols = select_variance_columns(temp_df, variance_threshold)
                self.selected_columns_by_step[step_index] = selected_cols
                temp_df = temp_df[selected_cols]

            elif method_name == 'correlation_filter':
                corr_threshold, corr_method = resolve_correlation_filter_params(method_config)
                selected_cols = select_correlation_columns(
                    temp_df, corr_threshold, corr_method
                )
                self.selected_columns_by_step[step_index] = selected_cols
                temp_df = temp_df[selected_cols]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.means is None:
            raise ValueError("PreprocessingState has not been fitted. Call fit() first.")

        numeric_df = self._get_numeric_columns(df)
        metadata_df = df[[col for col in df.columns if not ResultColumns.is_feature_column(col)]]

        if numeric_df.empty:
            return df.copy()

        df_transformed = numeric_df.copy()

        df_transformed = df_transformed.fillna(self.means)

        df_values = handle_extreme_values(df_transformed.values, strategy='mean_replacement')
        df_transformed = pd.DataFrame(df_values, columns=df_transformed.columns, index=df_transformed.index)

        for step_index, method_config in enumerate(self.methods_config):
            method_name = _get_method_attr(method_config, 'method', 'minmax')
            global_normalize = _get_method_attr(method_config, 'global_normalize', False)

            df_transformed = self._apply_method(
                df_transformed,
                method_name,
                global_normalize,
                method_config,
                step_index
            )

        if not metadata_df.empty:
            df_transformed = pd.concat([metadata_df, df_transformed], axis=1)
            original_order = [col for col in df.columns if col in df_transformed.columns]
            df_transformed = df_transformed[original_order]

        return df_transformed

    def _get_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        logger = get_module_logger(__name__)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if ResultColumns.is_feature_column(col)]

        logger.debug(
            f"_get_numeric_columns: {len(numeric_cols)} numeric, "
            f"{len(feature_cols)} feature (after dropping metadata)"
        )

        if not feature_cols:
            logger.warning(
                f"No feature columns found. Metadata cols: "
                f"{[c for c in df.columns if c in ResultColumns.metadata_columns()]}; "
                f"first non-numeric cols: "
                f"{df.select_dtypes(exclude=[np.number]).columns.tolist()[:10]}"
            )
            return pd.DataFrame()

        return df[feature_cols]

    def _apply_method(
        self,
        df: pd.DataFrame,
        method_name: str,
        global_normalize: bool,
        method_config: Union[Dict[str, Any], BaseModel],
        step_index: int
    ) -> pd.DataFrame:
        if method_name in ['z_score', 'zscore', 'standardization']:
            if global_normalize and 'zscore' in self.global_params:
                params = self.global_params['zscore']
                return (df - params['mean']) / params['std']
            else:
                return (df - self.means) / self.stds

        elif method_name in ['min_max', 'minmax', 'normalize']:
            if global_normalize and 'minmax' in self.global_params:
                params = self.global_params['minmax']
                denom = (params['max'] - params['min']) if params['max'] != params['min'] else 1.0
                return (df - params['min']) / denom
            else:
                denom = (self.maxs - self.mins).replace(0, 1.0)
                return (df - self.mins) / denom

        elif method_name == 'robust':
            if global_normalize and 'robust' in self.global_params:
                params = self.global_params['robust']
                iqr = params['q3'] - params['q1']
                iqr = iqr if iqr != 0 else 1.0
                return (df - params['median']) / iqr
            else:
                iqr = (self.q3s - self.q1s).replace(0, 1.0)
                return (df - self.medians) / iqr

        elif method_name == 'binning':
            if global_normalize and 'global' in self.discretizers:
                discretizer = self.discretizers['global']
                original_shape = df.shape
                flat_values = df.values.flatten().reshape(-1, 1)
                binned = discretizer.transform(flat_values)
                return pd.DataFrame(
                    binned.reshape(original_shape),
                    columns=df.columns,
                    index=df.index
                )
            elif 'per_feature' in self.discretizers:
                discretizer = self.discretizers['per_feature']
                binned = discretizer.transform(df.values)
                return pd.DataFrame(binned, columns=df.columns, index=df.index)
            else:
                return df

        elif method_name == 'winsorize':
            if global_normalize and 'winsorize' in self.global_params:
                params = self.global_params['winsorize']
                return df.clip(lower=params['lower'], upper=params['upper'])
            else:
                return df.clip(lower=self.winsor_lowers, upper=self.winsor_uppers, axis=1)

        elif method_name == 'log':
            if global_normalize and 'log_offset' in self.global_params:
                offset = self.global_params['log_offset']
                return np.log(df - offset + 1.0)
            else:
                return np.log(df - self.log_offsets + 1.0)

        elif method_name in FRAME_LEVEL_METHOD_NAMES:
            selected_cols = self.selected_columns_by_step.get(step_index, list(df.columns))
            valid_cols = [col for col in selected_cols if col in df.columns]
            if not valid_cols:
                return df
            return df[valid_cols]

        else:
            return df

    def save(self, output_dir: str, filename: str = 'preprocessing_state.pkl') -> None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, output_dir: str, filename: str = 'preprocessing_state.pkl') -> 'PreprocessingState':
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessing state file not found at {path}")

        with open(path, 'rb') as f:
            return pickle.load(f)
