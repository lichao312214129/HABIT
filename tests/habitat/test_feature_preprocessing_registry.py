"""
Tests for unified habitat feature preprocessing registry and pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from habit.core.habitat_analysis.config_schemas import DROPPING_PREPROCESSING_METHODS
from habit.core.habitat_analysis.feature_preprocessing import (
    PreprocessingMethodFactory,
    PreprocessingState,
    apply_stateless_preprocessing,
    register_preprocessing,
)
from habit.core.habitat_analysis.feature_preprocessing.base_preprocessing import (
    BaseFeaturePreprocessing,
)
from habit.core.habitat_analysis.feature_preprocessing.builtin_methods import MinMaxPreprocessing


class TestPreprocessingRegistry:
    def test_builtin_methods_registered(self) -> None:
        names = PreprocessingMethodFactory.registered_method_names()
        assert "minmax" in names
        assert "zscore" in names
        assert "variance_filter" in names
        assert "correlation_filter" in names

    def test_dropping_methods_match_config_constant(self) -> None:
        assert DROPPING_PREPROCESSING_METHODS == frozenset(
            {"variance_filter", "correlation_filter"}
        )

    def test_handler_instances_are_stateless(self) -> None:
        handler_a = PreprocessingMethodFactory.get_handler("minmax")
        handler_b = PreprocessingMethodFactory.get_handler("minmax")
        assert type(handler_a) is MinMaxPreprocessing
        assert type(handler_b) is MinMaxPreprocessing

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preprocessing method"):
            PreprocessingMethodFactory.get_handler("not_a_real_method")


class TestStatelessPipeline:
    def test_minmax_scales_to_unit_interval(self) -> None:
        df = pd.DataFrame({"A": [0.0, 5.0, 10.0], "B": [1.0, 2.0, 3.0]})
        methods = [{"method": "minmax", "global_normalize": False}]
        out = apply_stateless_preprocessing(df, methods)
        assert out["A"].min() == pytest.approx(0.0)
        assert out["A"].max() == pytest.approx(1.0)

    def test_variance_filter_drops_constant_column(self) -> None:
        df = pd.DataFrame({"const": [1.0] * 5, "vary": np.linspace(0, 1, 5)})
        methods = [{"method": "variance_filter", "variance_threshold": 0.0}]
        out = apply_stateless_preprocessing(df, methods)
        assert "const" not in out.columns
        assert "vary" in out.columns


class TestStatefulPipeline:
    def test_fit_transform_replays_column_selection(self) -> None:
        train_df = pd.DataFrame(
            {
                "Subject": ["s1"] * 5,
                "const": [1.0] * 5,
                "vary": np.linspace(0, 1, 5),
            }
        )
        test_df = pd.DataFrame(
            {
                "Subject": ["s2"] * 5,
                "const": [1.0] * 5,
                "vary": np.linspace(2, 3, 5),
            }
        )
        methods = [{"method": "variance_filter", "variance_threshold": 0.0}]

        state = PreprocessingState()
        state.fit(train_df, methods)
        out = state.transform(test_df)

        assert "const" not in out.columns
        assert "vary" in out.columns
        assert "Subject" in out.columns

    def test_transform_before_fit_raises(self) -> None:
        state = PreprocessingState()
        with pytest.raises(ValueError, match="has not been fitted"):
            state.transform(pd.DataFrame({"A": [1.0, 2.0]}))


class TestCustomRegistration:
    def test_register_decorator_adds_handler(self) -> None:
        @register_preprocessing("test_echo_scale")
        class EchoScalePreprocessing(BaseFeaturePreprocessing):
            changes_columns = False

            @classmethod
            def method_name(cls) -> str:
                return "test_echo_scale"

            def fit(self, feature_df, method_config, baseline=None):
                return None

            def transform(self, feature_df, method_config, state, baseline=None):
                factor = method_config.get("factor", 1.0) if isinstance(method_config, dict) else 1.0
                return feature_df * factor

        handler = PreprocessingMethodFactory.get_handler("test_echo_scale")
        assert isinstance(handler, EchoScalePreprocessing)

        df = pd.DataFrame({"A": [1.0, 2.0]})
        out = apply_stateless_preprocessing(df, [{"method": "test_echo_scale", "factor": 3.0}])
        assert out["A"].tolist() == pytest.approx([3.0, 6.0])

        # Cleanup registry entry so other tests are isolated.
        PreprocessingMethodFactory._registry.pop("test_echo_scale", None)
