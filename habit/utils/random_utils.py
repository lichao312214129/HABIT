"""
Utilities for resolving and propagating random seeds across HABIT configs.

YAML top-level ``random_state`` acts as the global default. Submodule fields
set explicitly in YAML override the global value; omitted or null submodule
fields inherit from the global seed.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

DEFAULT_RANDOM_STATE: int = 42
RANDOM_STATE_KEY: str = "random_state"


def resolve_random_state(
    explicit: Optional[int],
    global_seed: Optional[int],
    default: int = DEFAULT_RANDOM_STATE,
) -> int:
    """
    Resolve the effective random seed.

    Priority: explicit submodule value > global YAML value > default.

    Args:
        explicit: Submodule ``random_state`` from YAML (``None`` = inherit).
        global_seed: Top-level config ``random_state``.
        default: Fallback when both explicit and global are unset.

    Returns:
        int: Resolved non-negative integer seed.
    """
    if explicit is not None:
        return int(explicit)
    if global_seed is not None:
        return int(global_seed)
    return int(default)


def resolve_random_state_chain(
    *explicit_candidates: Optional[int],
    global_seed: Optional[int] = None,
    default: int = DEFAULT_RANDOM_STATE,
) -> int:
    """
    Resolve a seed from an ordered list of optional submodule values, then global.

    The first non-``None`` candidate wins; otherwise fall back to
    :func:`resolve_random_state` with ``explicit=None``.

    Args:
        *explicit_candidates: Submodule seeds in priority order (highest first).
        global_seed: Top-level config ``random_state``.
        default: Final fallback when no candidate and no global seed are set.

    Returns:
        int: Resolved non-negative integer seed.
    """
    for candidate in explicit_candidates:
        if candidate is not None:
            return int(candidate)
    return resolve_random_state(None, global_seed, default=default)


def merge_random_state_into_params(
    params: Optional[Dict[str, Any]],
    global_seed: Optional[int],
    *,
    key: str = RANDOM_STATE_KEY,
    default: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Return a shallow copy of ``params`` with ``random_state`` injected when absent.

    Existing explicit ``random_state`` in ``params`` is never overwritten.

    Args:
        params: Parameter mapping (may be ``None``).
        global_seed: Top-level config seed used when ``params`` lacks ``key``.
        key: Parameter name to inject (default ``random_state``).
        default: Fallback seed when ``global_seed`` is also unset.

    Returns:
        Dict[str, Any]: Params dict guaranteed to contain ``key`` when resolvable.
    """
    merged: Dict[str, Any] = deepcopy(params) if params else {}
    if key not in merged:
        merged[key] = resolve_random_state(None, global_seed, default=default)
    return merged


def seed_numpy_global(seed: int) -> None:
    """
    Seed NumPy's legacy global RNG once at a pipeline entry point.

    Args:
        seed: Integer seed applied via ``numpy.random.seed``.
    """
    np.random.seed(int(seed))
