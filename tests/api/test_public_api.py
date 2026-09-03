# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Contract tests for the v2 capability-package public API."""

from __future__ import annotations

import importlib
import sys

import pytest

from habit.api.registry import PUBLIC_API_SYMBOLS, PUBLIC_NAMESPACES


@pytest.mark.unit
def test_package_root_exposes_only_version_metadata() -> None:
    """V2 removes the unstructured root-level component mirror."""
    import habit

    assert habit.__all__ == ["__version__"]
    assert PUBLIC_API_SYMBOLS == ()
    assert isinstance(habit.__version__, str)
    assert habit.__version__ == "2.0.0"
    assert not hasattr(habit, "RawVoxelFeatures")


@pytest.mark.unit
def test_capability_namespaces_match_declared_exports() -> None:
    """Each capability package declares exactly its registered public names."""
    for namespace, declared in PUBLIC_NAMESPACES.items():
        package = importlib.import_module(namespace)
        assert tuple(package.__all__) == declared


@pytest.mark.unit
def test_capability_symbols_resolve_from_canonical_package() -> None:
    """Every declared public symbol is available from its sole namespace."""
    seen: dict[str, str] = {}
    for namespace, declared in PUBLIC_NAMESPACES.items():
        package = importlib.import_module(namespace)
        for symbol in declared:
            assert symbol not in seen, (symbol, seen.get(symbol), namespace)
            object_ = getattr(package, symbol)
            assert object_ is not None
            seen[symbol] = namespace


@pytest.mark.unit
def test_removed_domain_package_is_not_importable() -> None:
    """The v1 aggregate namespace must not survive as a namespace package."""
    sys.modules.pop("habit.domain", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("habit.domain")


@pytest.mark.unit
def test_import_habit_stays_lightweight() -> None:
    """Package metadata import must not eagerly load scientific backends."""
    import subprocess

    script = (
        "import sys, habit\n"
        "print('loaded', sorted({'radiomics', 'sklearn', 'SimpleITK'} & set(sys.modules)))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert "loaded []" in completed.stdout
