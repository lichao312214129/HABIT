"""Tests for habit.utils.random_utils."""

from __future__ import annotations

import numpy as np

from habit.utils.random_utils import (
    DEFAULT_RANDOM_STATE,
    merge_random_state_into_params,
    resolve_random_state,
    seed_numpy_global,
)


def test_resolve_random_state_explicit_overrides_global() -> None:
    assert resolve_random_state(explicit=7, global_seed=99) == 7


def test_resolve_random_state_inherits_global() -> None:
    assert resolve_random_state(explicit=None, global_seed=123) == 123


def test_resolve_random_state_default_fallback() -> None:
    assert resolve_random_state(explicit=None, global_seed=None) == DEFAULT_RANDOM_STATE


def test_merge_random_state_preserves_explicit() -> None:
    params = {"random_state": 5, "C": 1.0}
    merged = merge_random_state_into_params(params, global_seed=99)
    assert merged["random_state"] == 5
    assert merged["C"] == 1.0
    assert params["random_state"] == 5


def test_merge_random_state_injects_global() -> None:
    merged = merge_random_state_into_params({}, global_seed=11)
    assert merged["random_state"] == 11


def test_seed_numpy_global_is_deterministic() -> None:
    seed_numpy_global(42)
    first = np.random.randint(0, 1000, size=3)
    seed_numpy_global(42)
    second = np.random.randint(0, 1000, size=3)
    assert np.array_equal(first, second)
