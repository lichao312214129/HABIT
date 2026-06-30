# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Value collector: Gradio widget values → config dict (and reverse).

The collector eliminates manual ``config_data = {"field": value, ...}`` dict
construction. It reads :class:`FieldDescriptor` metadata to perform automatic
type conversion (string → int/float/bool/list), so the collected dict can be
passed directly to ``MyConfig(**dict)`` for validation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from habit.gui.schema_form.reflect import FieldDescriptor, reflect_schema


# ---------------------------------------------------------------------------
# Collect: widget values → dict
# ---------------------------------------------------------------------------
def _convert_value(desc: FieldDescriptor, raw: Any) -> Any:
    """Convert a raw widget value to the correct Python type for the field."""
    if raw is None or raw == "":
        if desc.is_optional or not desc.required:
            return desc.default if desc.default is not None else None
        # Required field with empty value — let Pydantic raise the error
        return None

    # Bool
    if desc.kind == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.lower() in ("true", "1", "yes")
        return bool(raw)

    # Int
    if desc.kind == "int":
        try:
            return int(raw)
        except (ValueError, TypeError):
            return raw  # Let Pydantic handle the error

    # Float
    if desc.kind == "float":
        try:
            return float(raw)
        except (ValueError, TypeError):
            return raw

    # List
    if desc.kind == "list":
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            # Comma-separated string → list
            parts = [s.strip() for s in raw.split(",") if s.strip()]
            # Try to convert items if inner type is known
            if parts and desc.inner_type is not None:
                from typing import get_args, get_origin
                item_type = None
                if get_origin(desc.inner_type) is list:
                    args = get_args(desc.inner_type)
                    item_type = args[0] if args else None
                if item_type is int:
                    return [int(p) for p in parts if p]
                elif item_type is float:
                    return [float(p) for p in parts if p]
            return parts
        return raw

    # Dict
    if desc.kind == "dict":
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw  # Let Pydantic handle
        return raw

    # Nested model
    if desc.kind == "nested":
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw

    # Path or string — return as-is
    return raw


def collect_values(
    descriptors: List[FieldDescriptor],
    widget_values: List[Any],
) -> Dict[str, Any]:
    """Collect widget values into a config dict, keyed by field name.

    Args:
        descriptors: Field descriptors in the same order as widget values.
        widget_values: Values from Gradio widgets (e.g. from ``inputs=[...]``).

    Returns:
        A dict suitable for ``MyConfig(**result)``. Only fields with non-None
        values are included (so defaults are used for unset optional fields).
    """
    result: Dict[str, Any] = {}
    for desc, raw in zip(descriptors, widget_values):
        converted = _convert_value(desc, raw)
        # Skip None values for optional fields (let Pydantic use defaults)
        if converted is None and not desc.required:
            continue
        result[desc.name] = converted
    return result


def collect_from_mapping(
    descriptors: List[FieldDescriptor],
    widget_map: Dict[str, Any],
) -> Dict[str, Any]:
    """Collect values from a name→value mapping (alternative to list-based collection).

    This is useful when widget values are accessed by component reference rather
    than by positional list.
    """
    result: Dict[str, Any] = {}
    for desc in descriptors:
        if desc.name not in widget_map:
            continue
        raw = widget_map[desc.name]
        converted = _convert_value(desc, raw)
        if converted is None and not desc.required:
            continue
        result[desc.name] = converted
    return result


# ---------------------------------------------------------------------------
# Populate: dict → gr.update list (for loading YAML into widgets)
# ---------------------------------------------------------------------------
def _widget_value_for_populate(desc: FieldDescriptor, config_value: Any) -> Any:
    """Convert a config dict value to the format the Gradio widget expects."""
    if config_value is None:
        return desc.default if desc.default is not None else ""

    # List → comma-separated string (for Textbox) or list (for CheckboxGroup)
    if desc.kind == "list":
        if isinstance(config_value, list):
            # Check if inner type is Literal → return as list for CheckboxGroup
            from habit.gui.schema_form.reflect import _extract_literal_values
            if desc.inner_type is not None:
                from typing import get_args, get_origin
                if get_origin(desc.inner_type) is list:
                    args = get_args(desc.inner_type)
                    if args and _extract_literal_values(args[0]) is not None:
                        return config_value  # CheckboxGroup expects list
            return ", ".join(str(v) for v in config_value)
        return config_value

    # Dict / Nested → JSON string
    if desc.kind in ("dict", "nested"):
        if isinstance(config_value, (dict, list)):
            return json.dumps(config_value, indent=2, default=str)
        return str(config_value) if config_value else ""

    # Bool
    if desc.kind == "bool":
        return bool(config_value)

    # Int / Float
    if desc.kind in ("int", "float"):
        return config_value

    # Literal / str / path
    return str(config_value) if config_value is not None else ""


def populate_widgets(
    descriptors: List[FieldDescriptor],
    config_dict: Dict[str, Any],
) -> List[Any]:
    """Build a list of ``gr.update(value=...)`` for loading a config dict into widgets.

    The returned list is in the same order as ``descriptors``, suitable for
    ``outputs=[widget1, widget2, ...]``.

    Args:
        descriptors: Field descriptors in widget order.
        config_dict: Configuration dict (e.g. loaded from YAML).

    Returns:
        List of values (not gr.update — just raw values, as Gradio auto-wraps).
    """
    values: List[Any] = []
    for desc in descriptors:
        config_value = config_dict.get(desc.name, desc.default)
        values.append(_widget_value_for_populate(desc, config_value))
    return values


def build_gr_updates(
    descriptors: List[FieldDescriptor],
    config_dict: Dict[str, Any],
) -> List[Any]:
    """Alias for :func:`populate_widgets`, returning gr.update objects.

    Use this when you need explicit ``gr.update`` objects instead of raw values.
    """
    import gradio as gr
    values = populate_widgets(descriptors, config_dict)
    return [gr.update(value=v) for v in values]
