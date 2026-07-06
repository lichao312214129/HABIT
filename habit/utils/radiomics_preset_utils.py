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
Resolve PyRadiomics ``params_file`` values against bundled package presets.

Goal: let users omit ``params_file`` for any radiomics-based feature extractor
or workflow. When the user provides no path (or an explicit ``@preset:<key>``
reference), HABIT falls back to a preset YAML shipped inside the package under
``habit/resources/radiomics`` so results stay reproducible without requiring a
checked-out repository.

Preset keys
-----------
* ``voxel``      : voxel_radiomics texture (CT R3B12, 21 stable GLCM features).
* ``supervoxel`` : supervoxel_radiomics ROI texture (full texture classes).
* ``roi``        : traditional ROI radiomics (``habit radiomics``; full set incl. shape).
* ``habitat``    : habitat-map radiomics (``habit extract`` habitat maps).

Usage
-----
User-provided paths always win. A missing value resolves to the preset::

    resolve_params_file(None, "voxel")           # -> bundled voxel preset path
    resolve_params_file("@preset:roi", "roi")    # -> bundled roi preset path
    resolve_params_file("./my_params.yaml", "voxel")  # -> "./my_params.yaml"
"""

from __future__ import annotations

import importlib.resources as importlib_resources
from pathlib import Path
from typing import Dict, Optional

# Preset key -> bundled filename under ``habit/resources/radiomics``.
PRESET_FILES: Dict[str, str] = {
    "voxel": "params_voxel_radiomics.yaml",
    "supervoxel": "params_supervoxel_radiomics.yaml",
    "roi": "parameter.yaml",
    "habitat": "parameter_habitat.yaml",
}

# Package that physically stores the preset YAML files.
_RESOURCE_PACKAGE: str = "habit.resources.radiomics"

# Explicit reference prefix a user may write in YAML to force a named preset,
# e.g. ``params_file: "@preset:voxel"``.
_PRESET_PREFIX: str = "@preset:"


def available_presets() -> tuple[str, ...]:
    """
    Return the tuple of valid preset keys.

    Returns:
        tuple[str, ...]: Sorted preset keys (e.g. ``("habitat", "roi", ...)``).
    """
    return tuple(sorted(PRESET_FILES.keys()))


def is_preset_ref(value: object) -> bool:
    """
    Check whether a value is an explicit ``@preset:<key>`` reference.

    Args:
        value: Raw ``params_file`` value from config (any type).

    Returns:
        bool: True when ``value`` is a string starting with ``@preset:``.
    """
    return isinstance(value, str) and value.strip().startswith(_PRESET_PREFIX)


def _preset_key_from_ref(value: str) -> str:
    """
    Extract the preset key from an ``@preset:<key>`` reference string.

    Args:
        value: A string beginning with ``@preset:``.

    Returns:
        str: The preset key (e.g. ``"voxel"``).

    Raises:
        ValueError: When the referenced key is not a known preset.
    """
    key: str = value.strip()[len(_PRESET_PREFIX):].strip().lower()
    if key not in PRESET_FILES:
        raise ValueError(
            f"Unknown radiomics preset '{key}' in '{value}'. "
            f"Valid presets: {', '.join(available_presets())}."
        )
    return key


def get_preset_path(preset: str) -> str:
    """
    Return the absolute filesystem path of a bundled preset YAML.

    Args:
        preset: Preset key (see :data:`PRESET_FILES`).

    Returns:
        str: Absolute path to the bundled preset file on disk.

    Raises:
        ValueError: When ``preset`` is not a known preset key.
        FileNotFoundError: When the bundled resource cannot be located on disk.
    """
    key: str = str(preset).lower()
    if key not in PRESET_FILES:
        raise ValueError(
            f"Unknown radiomics preset '{preset}'. "
            f"Valid presets: {', '.join(available_presets())}."
        )

    filename: str = PRESET_FILES[key]
    resource = importlib_resources.files(_RESOURCE_PACKAGE).joinpath(filename)
    resource_path = Path(str(resource))
    if not resource_path.is_file():
        raise FileNotFoundError(
            f"Bundled radiomics preset '{key}' not found at {resource_path}. "
            "The HABIT installation may be incomplete."
        )
    return str(resource_path)


def resolve_params_file(
    user_value: Optional[str],
    preset: str,
) -> str:
    """
    Resolve an effective ``params_file`` path, falling back to a bundled preset.

    Resolution order:

    1. ``user_value`` is an ``@preset:<key>`` reference -> that preset's path.
    2. ``user_value`` is a non-empty path -> returned unchanged (user wins).
    3. ``user_value`` is None / empty -> the ``preset`` argument's bundled path.

    Note:
        This function does not perform YAML-relative path normalization; callers
        that need relative-to-config resolution should do so on the user value
        before/after calling this (bundled presets are always absolute).

    Args:
        user_value: Raw ``params_file`` value from config (path, ``@preset:*``,
            or None/empty).
        preset: Default preset key to use when ``user_value`` is missing.

    Returns:
        str: Effective absolute (preset) or user-provided ``params_file`` path.

    Raises:
        ValueError: When an ``@preset:*`` reference or ``preset`` key is invalid.
    """
    if is_preset_ref(user_value):
        return get_preset_path(_preset_key_from_ref(str(user_value)))

    if user_value is not None and str(user_value).strip() != "":
        return str(user_value)

    return get_preset_path(preset)
