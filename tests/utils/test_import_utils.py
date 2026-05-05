"""
Unit tests for habit.utils.import_utils.

Tests cover: is_available(), import_error(), lazy importing of optional packages.
"""

from __future__ import annotations

import pytest

from habit.utils.import_utils import import_error, is_available


class TestIsAvailable:
    def test_numpy_is_available(self) -> None:
        """numpy is a core dependency and should always be importable."""
        assert is_available("numpy") is True

    def test_sklearn_is_available(self) -> None:
        assert is_available("sklearn") is True

    def test_nonexistent_package_not_available(self) -> None:
        assert is_available("totally_fake_package_xyz_123") is False

    def test_returns_bool(self) -> None:
        result = is_available("numpy")
        assert isinstance(result, bool)


class TestImportError:
    def test_available_package_returns_none(self) -> None:
        """import_error should return None (no error) for available packages."""
        err = import_error("numpy")
        assert err is None

    def test_unavailable_package_returns_string(self) -> None:
        err = import_error("totally_fake_package_xyz_123")
        assert err is not None
        assert isinstance(err, str)
        assert len(err) > 0
