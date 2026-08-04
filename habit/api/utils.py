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
"""Lightweight utilities re-exported from the top-level ``habit`` namespace."""

from __future__ import annotations

import importlib.util

from typing import Mapping

from habit.contracts.provenance import software_fingerprint
from habit.utils.log_utils import setup_logger

__all__ = ["setup_logger", "is_available", "show_versions", "check_component"]


def is_available(module_name: str) -> bool:
    """
    Return whether an optional third-party module can be imported.

    Args:
        module_name: Top-level package name (e.g. ``"radiomics"``, ``"torch"``).

    Returns:
        True when ``importlib`` finds a spec for the module.
    """
    return importlib.util.find_spec(module_name) is not None


def check_component(name: str, domain: str) -> bool:
    """
    Return whether ``name`` is registered in a HABIT plugin domain.

    This is a lightweight pre-flight check before building a
    :class:`~habit.spec.specs.HabitatSpec` or calling
    :func:`~habit.api.plugins.list_plugins`. It resolves the domain through
    the same registry mapping used by the plugin discovery API.

    Args:
        name: Registered implementation name (e.g. ``"kmeans"``,
            ``"slic"``).
        domain: Plugin domain key (e.g. ``"habitat_model_fitter"``,
            ``"supervoxelizer"``). See :func:`~habit.api.plugins.list_plugins`
            for the supported domain names.

    Returns:
        ``True`` when ``name`` appears in the domain registry;
        ``False`` when the domain is unknown or the name is not registered.
    """
    from habit.api.plugins import _registry_for_domain

    from habit.exceptions import HABITAPIError

    try:
        registry = _registry_for_domain(domain)
    except HABITAPIError:
        return False
    return name in registry.available()


def show_versions() -> Mapping[str, str]:
    """
    Return HABIT and key dependency versions for reproducibility and debugging.

    Public entry point for the same version fingerprint embedded in
    :class:`~habit.contracts.Provenance` records and run manifests. Call it
    from notebooks, bug reports, or manuscript supplements when you need to
    record the exact software stack without executing a full workflow.

    Returns:
        Mapping of distribution name to installed version string, always
        including the ``habit`` entry. Optional dependencies that are not
        installed are omitted rather than raising.
    """
    return software_fingerprint()
