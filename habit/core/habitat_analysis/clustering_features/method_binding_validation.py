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
Validate parameter binding for a feature-construction ``method`` expression.

This enforces the DSL contract that the parenthesis token list of each function
is the single source of truth for which parameters bind to that function:

* **Error** when a method omits a *required* parameter name from its parentheses
  (e.g. ``kinetic(...)`` without ``timestamps``).
* **Warning (deprecated)** when a key in the sibling ``params`` dict maps to a
  known parameter of a used method but was **not** listed in that method's
  parentheses — the historical "ambiguous binding" that hid which function a
  parameter belonged to. Users should list the name in the relevant parentheses.
* **Warning** when a ``params`` key is not referenced by any function and is not
  a known parameter of any used method (a likely typo / orphan).

Validation is intentionally lenient (warnings, not errors) for the ambiguous and
orphan cases so that existing configs keep running; only genuinely missing
required bindings raise.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .feature_expression_parser import FeatureExpressionParser
from .method_param_spec import MethodParamRegistry


def _collect_binding(
    method_expr: str,
    params: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], Set[str], Set[str]]:
    """
    Parse an expression and collect binding information.

    Args:
        method_expr: The ``method`` expression string.
        params: The sibling ``params`` dict.

    Returns:
        Tuple of:
        * cross_image_method: outer combiner method name.
        * cross_image_params: parameter names bound to the outer combiner.
        * processing_steps: parsed per-modality steps (each with method/params).
        * bound_names: every parameter name listed in any function's parentheses
          (per-step step['params'] keys plus outer combiner param names).
        * used_methods: set of all method names appearing in the expression.
    """
    parser = FeatureExpressionParser()
    cross_image_method, cross_image_params, processing_steps = parser.parse(
        {"method": method_expr, "params": dict(params)}
    )

    bound_names: Set[str] = set(cross_image_params.keys())
    used_methods: Set[str] = set()
    if cross_image_method:
        used_methods.add(cross_image_method)
    for step in processing_steps:
        used_methods.add(step["method"])
        bound_names.update(step["params"].keys())

    return (
        cross_image_method,
        cross_image_params,
        processing_steps,
        bound_names,
        used_methods,
    )


def validate_feature_method_binding(
    method_expr: str,
    params: Optional[Dict[str, Any]],
    *,
    level_name: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Validate parameter binding for one feature-construction level.

    Args:
        method_expr: The ``method`` expression (e.g. ``concat(voxel_radiomics(T2))``).
        params: The sibling ``params`` dict (may be None/empty).
        level_name: Human-readable level label for messages (e.g.
            ``feature_construction.voxel_level``).
        logger: Optional logger; when None, warnings go to this module's logger.

    Raises:
        ValueError: When a used method omits a required parameter name from its
            parentheses.
    """
    log = logger or logging.getLogger(__name__)
    params = params or {}

    try:
        (
            cross_image_method,
            cross_image_params,
            processing_steps,
            bound_names,
            used_methods,
        ) = _collect_binding(method_expr, params)
    except Exception as exc:  # parsing errors are surfaced elsewhere; skip binding check
        log.debug("Skip binding validation for %s (parse failed): %s", level_name, exc)
        return

    # 1. Required parameters must be listed in the relevant parentheses.
    #    Check the outer combiner and each per-modality step against its spec.
    def _check_required(method_name: str, bound: Set[str]) -> None:
        spec = MethodParamRegistry.get(method_name)
        missing = [name for name in spec.required if name not in bound]
        if missing:
            raise ValueError(
                f"{level_name}: method '{method_name}' is missing required "
                f"parameter name(s) {missing} inside its parentheses. Write them "
                f"in the expression, e.g. {method_name}(<modality>, "
                f"{', '.join(spec.required)}), and give values under params."
            )

    if cross_image_method:
        _check_required(cross_image_method, set(cross_image_params.keys()))
    for step in processing_steps:
        _check_required(step["method"], set(step["params"].keys()))

    # 2. Build the set of parameter names known to any used method.
    known_by_used: Set[str] = set()
    for method in used_methods:
        known_by_used.update(MethodParamRegistry.get(method).known_param_names())

    # 3. Inspect each params key.
    for key in params.keys():
        if key in bound_names:
            continue  # explicitly bound in parentheses — good
        if key in known_by_used:
            # Known to a used method but not listed in parentheses: ambiguous.
            log.warning(
                "%s: params key '%s' is not listed in any function's parentheses "
                "in method '%s'. Binding it implicitly is deprecated; add '%s' to "
                "the relevant function's parentheses to make the binding explicit.",
                level_name,
                key,
                method_expr,
                key,
            )
        else:
            # Not referenced anywhere and unknown to used methods: likely orphan.
            log.warning(
                "%s: params key '%s' is not used by any method in '%s' "
                "(unknown parameter). Remove it or check for a typo.",
                level_name,
                key,
                method_expr,
            )
