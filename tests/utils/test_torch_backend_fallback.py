# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Tests for safe TorchRadiomics fallback when PyTorch cannot be imported."""

from __future__ import annotations

import builtins
import logging
from types import ModuleType
from typing import Any, Callable
from unittest.mock import Mock

import pytest

from habit.utils import persistent_worker_entry
from habit.utils.torch_radiomics_utils import (
    is_cuda_available,
    is_torch_available,
    resolve_voxel_radiomics_backend,
)


def _install_broken_torch_import(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """
    Replace Python's import hook with one that simulates a broken Torch DLL.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to restore the import hook.

    Returns:
        Mock: Callable import hook whose call history can be inspected.
    """
    original_import: Callable[..., ModuleType] = builtins.__import__

    def broken_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "torch":
            raise OSError(126, "Error loading torch/lib/fbgemm.dll")
        return original_import(name, globals, locals, fromlist, level)

    import_mock = Mock(side_effect=broken_import)
    monkeypatch.setattr(builtins, "__import__", import_mock)
    return import_mock


def test_torch_availability_probes_treat_dll_error_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native Torch DLL error must be reported as backend unavailability."""
    _install_broken_torch_import(monkeypatch)

    assert is_torch_available() is False
    assert is_cuda_available() is False


@pytest.mark.parametrize("mode", ["auto", "true"])
def test_broken_torch_falls_back_to_cpu_pyradiomics(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """Auto and explicitly enabled Torch modes must retain a working CPU path."""
    _install_broken_torch_import(monkeypatch)

    backend, device = resolve_voxel_radiomics_backend(
        use_torch_radiomics=mode,
        torch_device="auto",
    )

    assert backend == "pyradiomics"
    assert device is None


def test_disabled_torch_does_not_probe_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicitly disabled Torch mode must return before attempting an import."""
    import_mock = _install_broken_torch_import(monkeypatch)

    backend, device = resolve_voxel_radiomics_backend(
        use_torch_radiomics="false",
        torch_device="auto",
    )

    torch_imports = [
        call for call in import_mock.call_args_list if call.args and call.args[0] == "torch"
    ]
    assert backend == "pyradiomics"
    assert device is None
    assert torch_imports == []


def test_cache_cleanup_dll_error_never_escapes_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional CUDA cleanup must not crash a worker when Torch DLL loading fails."""
    _install_broken_torch_import(monkeypatch)
    monkeypatch.setattr(
        persistent_worker_entry,
        "_TORCH_CACHE_CLEANUP_WARNING_LOGGED",
        False,
    )
    logger = Mock(spec=logging.Logger)

    persistent_worker_entry._maybe_empty_cuda_cache(logger)
    persistent_worker_entry._maybe_empty_cuda_cache(logger)

    logger.warning.assert_called_once()
