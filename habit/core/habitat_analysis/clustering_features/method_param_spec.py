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
Per-method parameter contracts for the functional feature-construction DSL.

Motivation
----------
The ``feature_construction.voxel_level.method`` / ``supervoxel_level.method``
expressions are a small functional DSL, e.g.::

    concat(voxel_radiomics(T2, params_file, kernel_radius))

Historically the meaning of a parenthesis token was ambiguous: some names were
placeholders resolved from a sibling ``params`` dict, while other keys lived
only in ``params`` and were silently merged into *every* step. Users could not
tell which parameter bound to which function.

This module defines a single, declarative contract, :class:`MethodParamSpec`,
that every feature extractor (single-modality *and* combiner) attaches to its
class as ``method_param_spec``. The contract states, per method:

* which parameter names **must** appear inside the function's parentheses
  (``required``);
* which parameters are **optional** and their built-in default values
  (``optional``), so users can omit them entirely ("less input, use defaults");
* whether ``params_file`` may be omitted, falling back to a bundled preset
  (``default_params_file_preset``); and
* whether the method is a **combiner** (``concat`` / ``kinetic`` /
  ``mean_voxel_features``) rather than a per-modality extractor.

Extensibility
-------------
Adding a new algorithm requires **no central edit**: a plugin author simply
sets ``method_param_spec = MethodParamSpec(...)`` on their
``BaseClusteringExtractor`` subclass. :class:`MethodParamRegistry` reads the
attribute lazily from :class:`FeatureExtractorRegistry`, so the DSL parser and
validators pick the new method up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class MethodParamSpec:
    """
    Declarative parameter contract for one feature-construction method.

    Attributes:
        required: Parameter names that MUST be listed inside the method's
            parentheses (excluding the leading image/modality token). These are
            genuinely mandatory bindings; ``params_file`` is intentionally *not*
            required when ``default_params_file_preset`` is set.
        optional: Mapping of optional parameter name -> default value. When a
            name is not supplied by the user (neither bound in parentheses nor
            present in ``params``), the default is injected so the extractor
            still receives an explicit value. Keeps user input minimal.
        default_params_file_preset: Preset key (see
            :mod:`habit.utils.radiomics_preset_utils`) used to resolve
            ``params_file`` when the user omits it. ``None`` means the method
            does not consume a ``params_file``.
        combiner: True for cross-image combiners (``concat``, ``kinetic``) and
            supervoxel aggregators (``mean_voxel_features``) that wrap or
            aggregate per-modality sub-expressions rather than reading one image.
        takes_image: Whether the first parenthesis token is an image/modality
            name. Combiners like ``concat`` take sub-expressions, not an image
            token, but ``kinetic`` wraps ``raw(...)`` sub-expressions too.
    """

    required: Tuple[str, ...] = ()
    optional: Dict[str, Any] = field(default_factory=dict)
    default_params_file_preset: Optional[str] = None
    combiner: bool = False
    takes_image: bool = True

    def known_param_names(self) -> Tuple[str, ...]:
        """
        Return every parameter name this method understands.

        Includes ``required``, ``optional`` keys, and ``params_file`` when the
        method resolves one via a preset.

        Returns:
            Tuple[str, ...]: Sorted, de-duplicated known parameter names.
        """
        names = set(self.required) | set(self.optional.keys())
        if self.default_params_file_preset is not None:
            names.add("params_file")
        return tuple(sorted(names))

    def uses_params_file(self) -> bool:
        """
        Whether this method consumes a ``params_file`` (preset-backed).

        Returns:
            bool: True when a bundled preset backs ``params_file``.
        """
        return self.default_params_file_preset is not None


# Default contract for methods that declare nothing (e.g. legacy custom
# extractors without a ``method_param_spec``). Empty and permissive.
DEFAULT_METHOD_PARAM_SPEC = MethodParamSpec()


class MethodParamRegistry:
    """
    Read-only view over per-method :class:`MethodParamSpec` contracts.

    Specs are sourced from each extractor class's ``method_param_spec`` attribute
    via :class:`FeatureExtractorRegistry`, so there is no central table to keep
    in sync. Lookups are case-insensitive to match the extractor registry.
    """

    @classmethod
    def get(cls, method: str) -> MethodParamSpec:
        """
        Return the parameter contract for ``method``.

        Args:
            method: Method name as written in the DSL (e.g. ``voxel_radiomics``).

        Returns:
            MethodParamSpec: The registered spec, or
            :data:`DEFAULT_METHOD_PARAM_SPEC` when the extractor declares none.
        """
        # Imported lazily to avoid a circular import at module load time
        # (base_extractor imports this module's sibling extractors).
        from .base_extractor import FeatureExtractorRegistry

        extractor_cls = FeatureExtractorRegistry.get(method)
        if extractor_cls is None:
            return DEFAULT_METHOD_PARAM_SPEC
        spec = getattr(extractor_cls, "method_param_spec", None)
        if isinstance(spec, MethodParamSpec):
            return spec
        return DEFAULT_METHOD_PARAM_SPEC

    @classmethod
    def is_known(cls, method: str) -> bool:
        """
        Whether ``method`` maps to a registered feature extractor.

        Args:
            method: Method name from the DSL.

        Returns:
            bool: True when a feature extractor is registered under ``method``.
        """
        from .base_extractor import FeatureExtractorRegistry

        return FeatureExtractorRegistry.get(method) is not None
