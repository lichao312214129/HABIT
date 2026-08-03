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
"""YAML isomorphism for specifications and run policies.

The Spec/YAML mapping is one-to-one -- what you see in YAML is exactly what
exists in Python, with no hidden schema translation. All YAML handling of the
v1.0 API is confined to this module so the core never imports ``yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

import yaml

from habit.api.exceptions import HABITAPIError
from habit.spec.policy import RunPolicy
from habit.spec.specs import HabitatSpec

__all__ = ["load_habitat_spec", "save_habitat_spec", "load_run_policy", "save_run_policy"]


def _read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Read one YAML mapping file with a clear error contract."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"YAML file not found: {source}")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HABITAPIError(f"Invalid YAML in {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HABITAPIError(
            f"{source} must contain a YAML mapping at the top level."
        )
    return payload


def _write_yaml(payload: Mapping[str, Any], path: Union[str, Path]) -> Path:
    """Write one YAML mapping file, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination


def load_habitat_spec(path: Union[str, Path]) -> HabitatSpec:
    """
    Load a :class:`HabitatSpec` from its YAML form.

    Args:
        path: YAML file holding a ``HabitatSpec.to_dict`` payload.

    Returns:
        The parsed specification.
    """
    return HabitatSpec.from_dict(_read_yaml(path))


def save_habitat_spec(spec: HabitatSpec, path: Union[str, Path]) -> Path:
    """
    Write a :class:`HabitatSpec` as YAML.

    Args:
        spec: The specification to serialise.
        path: Destination file.

    Returns:
        The written path.
    """
    return _write_yaml(spec.to_dict(), path)


def load_run_policy(path: Union[str, Path]) -> RunPolicy:
    """
    Load a :class:`RunPolicy` from a YAML mapping.

    Args:
        path: YAML file with run-policy keys; missing keys take defaults.

    Returns:
        The parsed run policy.
    """
    return RunPolicy.from_dict(_read_yaml(path))


def save_run_policy(policy: RunPolicy, path: Union[str, Path]) -> Path:
    """
    Write a :class:`RunPolicy` as YAML.

    Args:
        policy: The policy to serialise.
        path: Destination file.

    Returns:
        The written path.
    """
    return _write_yaml(policy.to_dict(), path)
