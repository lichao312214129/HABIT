# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Schema reflection: Pydantic model → FieldDescriptor list for dynamic UI generation.

This module is the foundation of the schema-driven GUI. It inspects Pydantic
model fields and produces structured descriptors that the widget factory can
render into Gradio components — without any hardcoded field names.

Key design:
- Schema is the single source of truth; adding a Field() in a Pydantic model
  automatically makes it appear in the GUI.
- UI metadata (group, order, widget type, conditional visibility) is declared
  via ``json_schema_extra`` on each Field — non-breaking for CLI/backend.
- Path fields are auto-detected by name patterns (synced with PathResolver).
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

try:
    from pydantic_core import PydanticUndefined
except ImportError:
    PydanticUndefined = None  # fallback for older pydantic

# ---------------------------------------------------------------------------
# Path field detection (synced with habit.core.common.configs.loader.PathResolver)
# ---------------------------------------------------------------------------
_PATH_SUFFIXES: frozenset[str] = frozenset({
    "_path", "_dir", "_file", "_folder", "_directory", "_root", "_config", "_location",
})
_PATH_EXACT: frozenset[str] = frozenset({
    "path", "dir", "file", "folder", "directory", "root",
    "data_dir", "out_dir", "output_dir", "input_dir",
    "config", "config_file", "mask_path", "image_path",
    "source", "destination", "target", "pipeline_path", "checkpoint_dir",
})
_PATH_DIR_KEYWORDS: frozenset[str] = frozenset({
    "dir", "directory", "root", "data_dir", "out_dir", "output_dir",
    "input_dir", "checkpoint_dir", "_dir", "_directory", "_root",
})

# Fields that are internal and should not be rendered in the GUI.
_INTERNAL_FIELDS: frozenset[str] = frozenset({"config_file", "config_version"})


# ---------------------------------------------------------------------------
# FieldDescriptor
# ---------------------------------------------------------------------------
@dataclass
class FieldDescriptor:
    """Structured description of a single Pydantic field, for UI rendering.

    The widget factory reads this descriptor to decide which Gradio component
    to create, what label/default/constraints to use, and when to show/hide it.
    """

    name: str
    """Field name as it appears in the Pydantic model and YAML config."""

    annotation: Any
    """Original Python type annotation (e.g. ``Optional[int]``, ``Literal['a','b']``)."""

    default: Any
    """Default value, or ``None`` if required without default."""

    description: Optional[str]
    """Human-readable description from ``Field(description=...)``."""

    required: bool
    """Whether the field has no default and must be provided."""

    # --- Type kind ---
    kind: str = "str"
    """One of: literal, bool, int, float, str, path, nested, list, dict."""

    # --- Literal / choices ---
    choices: Optional[List[Any]] = None
    """Enumerated values when kind == 'literal'."""

    # --- Numeric constraints ---
    ge: Optional[float] = None
    gt: Optional[float] = None
    le: Optional[float] = None
    lt: Optional[float] = None

    # --- String / list constraints ---
    min_length: Optional[int] = None
    max_length: Optional[int] = None

    # --- Container types ---
    is_list: bool = False
    is_optional: bool = False
    inner_type: Optional[Any] = None
    """Unwrapped inner type for ``List[X]`` or ``Optional[X]``."""

    # --- Nested model ---
    nested_model: Optional[Type[BaseModel]] = None
    """Pydantic model class when kind == 'nested'."""

    # --- Path field ---
    is_path: bool = False
    path_type: str = "file"
    """'file' or 'dir' — determines browse button behavior."""

    # --- UI metadata (from json_schema_extra) ---
    group: Optional[str] = None
    """Display group name for accordion/section layout."""

    order: int = 0
    """Sort order within a group (lower = first)."""

    widget: Optional[str] = None
    """Override widget type: 'slider', 'textarea', 'path', 'path_dir', etc."""

    visible_if: Optional[Dict[str, Any]] = None
    """Conditional visibility: ``{field_name: expected_value}``."""

    label: Optional[str] = None
    """Custom display label (defaults to field name)."""

    help_text: Optional[str] = None
    """Additional help text beyond description."""

    @property
    def is_nested(self) -> bool:
        return self.nested_model is not None

    @property
    def display_label(self) -> str:
        return self.label or self.name


# ---------------------------------------------------------------------------
# Type inspection helpers
# ---------------------------------------------------------------------------
def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Unwrap ``Optional[X]`` → ``(X, True)``.

    ``Optional[X]`` is ``Union[X, None]``. If the annotation is a Union of
    more than one non-None type, it is returned unchanged.
    """
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            return non_none[0], True
    # Python 3.10+ X | None syntax
    if hasattr(typing, "UnionType"):
        try:
            if isinstance(annotation, typing.UnionType):
                args = get_args(annotation)
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1 and len(args) == 2:
                    return non_none[0], True
        except (TypeError, AttributeError):
            pass
    return annotation, False


def _extract_literal_values(annotation: Any) -> Optional[List[Any]]:
    """Extract values from ``Literal[...]``. Returns ``None`` if not Literal."""
    origin = get_origin(annotation)
    if origin is typing.Literal:
        return list(get_args(annotation))
    return None


def _is_path_field(name: str) -> tuple[bool, str]:
    """Check if field name suggests a file/directory path.

    Returns ``(is_path, path_type)`` where path_type is 'file' or 'dir'.
    """
    lower = name.lower()
    if lower in _PATH_EXACT:
        path_type = "dir" if lower in _PATH_DIR_KEYWORDS else "file"
        return True, path_type
    for suffix in _PATH_SUFFIXES:
        if lower.endswith(suffix):
            path_type = "dir" if suffix in _PATH_DIR_KEYWORDS else "file"
            return True, path_type
    return False, "file"


def _extract_constraints(field_info: FieldInfo) -> dict:
    """Extract numeric/string constraints from ``FieldInfo.metadata``."""
    constraints: dict[str, Any] = {}
    try:
        from annotated_types import Ge, Gt, Le, Lt, MinLen, MaxLen
    except ImportError:
        return constraints
    for m in field_info.metadata:
        if isinstance(m, Ge):
            constraints["ge"] = m.ge
        elif isinstance(m, Gt):
            constraints["gt"] = m.gt
        elif isinstance(m, Le):
            constraints["le"] = m.le
        elif isinstance(m, Lt):
            constraints["lt"] = m.lt
        elif isinstance(m, MinLen):
            constraints["min_length"] = m.min_length
        elif isinstance(m, MaxLen):
            constraints["max_length"] = m.max_length
    return constraints


def _extract_ui_metadata(field_info: FieldInfo) -> dict:
    """Extract UI metadata from ``json_schema_extra``.

    Supports dict form: ``json_schema_extra={"group": "...", "order": 10, ...}``
    """
    extra = field_info.json_schema_extra
    if extra is None or callable(extra):
        return {}
    if isinstance(extra, dict):
        ui_keys = ("group", "order", "widget", "visible_if", "label", "help_text")
        return {k: extra[k] for k in ui_keys if k in extra}
    return {}


def _get_default(field_info: FieldInfo) -> Any:
    """Get the default value, handling PydanticUndefined and default_factory."""
    default = field_info.default
    if PydanticUndefined is not None and default is PydanticUndefined:
        if field_info.default_factory is not None:
            try:
                return field_info.default_factory()
            except Exception:
                return None
        return None
    return default


# ---------------------------------------------------------------------------
# Core reflection functions
# ---------------------------------------------------------------------------
def reflect_field(name: str, field_info: FieldInfo) -> FieldDescriptor:
    """Reflect a single Pydantic field into a :class:`FieldDescriptor`.

    This function inspects the field's type annotation, constraints, default
    value, and UI metadata to produce a complete descriptor that the widget
    factory can render without any hardcoded knowledge of the field.
    """
    annotation = field_info.annotation
    description = field_info.description
    required = field_info.is_required()
    default = _get_default(field_info)

    # Unwrap Optional[X]
    inner, is_optional = _unwrap_optional(annotation)

    # Literal choices
    choices = _extract_literal_values(inner)

    # List detection
    is_list = get_origin(inner) is list

    # Nested model detection
    nested_model = None
    try:
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            nested_model = inner
    except TypeError:
        pass

    # Path detection — only for string-type fields (bool/int/float are never paths)
    if inner is str or is_list:
        is_path, path_type = _is_path_field(name)
    else:
        is_path = False
        path_type = "file"

    # Determine kind
    if choices is not None:
        kind = "literal"
    elif inner is bool:
        kind = "bool"
    elif inner is int:
        kind = "int"
    elif inner is float:
        kind = "float"
    elif is_list:
        kind = "list"
    elif nested_model is not None:
        kind = "nested"
    elif get_origin(inner) is dict:
        kind = "dict"
    elif inner is str:
        kind = "str"
    elif inner is Any or inner is None:
        kind = "str"
    else:
        # Fallback: treat as string
        kind = "str"

    # If it's a list of paths (List[str] where name suggests paths)
    if is_list and not is_path:
        # Check if inner list item type or field name suggests paths
        list_args = get_args(inner)
        if list_args and _is_path_field(name)[0]:
            is_path = True
            path_type = _is_path_field(name)[1]

    # Extract constraints and UI metadata
    constraints = _extract_constraints(field_info)
    ui_meta = _extract_ui_metadata(field_info)

    # Widget override
    widget = ui_meta.get("widget")
    if widget in ("path", "path_file", "path_dir"):
        is_path = True
        path_type = "dir" if widget == "path_dir" else "file"
    elif is_path:
        widget = widget or "path"

    return FieldDescriptor(
        name=name,
        annotation=annotation,
        default=default,
        description=description,
        required=required,
        kind=kind,
        choices=choices,
        is_list=is_list,
        is_optional=is_optional,
        inner_type=inner if (is_list or is_optional) else None,
        nested_model=nested_model,
        is_path=is_path,
        path_type=path_type,
        ge=constraints.get("ge"),
        gt=constraints.get("gt"),
        le=constraints.get("le"),
        lt=constraints.get("lt"),
        min_length=constraints.get("min_length"),
        max_length=constraints.get("max_length"),
        group=ui_meta.get("group"),
        order=ui_meta.get("order", 0),
        widget=widget,
        visible_if=ui_meta.get("visible_if"),
        label=ui_meta.get("label"),
        help_text=ui_meta.get("help_text"),
    )


def reflect_schema(model_cls: Type[BaseModel]) -> List[FieldDescriptor]:
    """Reflect a Pydantic model into an ordered list of :class:`FieldDescriptor`.

    Internal fields (``config_file``, ``config_version``) are skipped.
    Descriptors preserve declaration order; grouping/sorting is handled by
    :func:`group_descriptors`.
    """
    descriptors: List[FieldDescriptor] = []
    for name, field_info in model_cls.model_fields.items():
        if name in _INTERNAL_FIELDS:
            continue
        descriptors.append(reflect_field(name, field_info))
    return descriptors


def group_descriptors(
    descriptors: List[FieldDescriptor],
    group_order: Optional[List[str]] = None,
) -> "Dict[str, List[FieldDescriptor]]":
    """Group descriptors by their ``group`` attribute.

    Args:
        descriptors: Flat list from :func:`reflect_schema`.
        group_order: Optional explicit ordering of group names. Groups not in
            this list are appended in first-appearance order.

    Returns:
        Ordered dict of ``{group_name: [descriptors]}``. Within each group,
        descriptors are sorted by ``order`` then original position.
    """
    groups: Dict[str, List[FieldDescriptor]] = {}
    for idx, d in enumerate(descriptors):
        key = d.group or "General"
        if key not in groups:
            groups[key] = []
        # Store original position for stable sort
        groups[key].append((idx, d))

    # Sort within each group by order, then original position
    for key in groups:
        groups[key] = [d for _, d in sorted(groups[key], key=lambda pair: (pair[1].order, pair[0]))]

    # Order groups
    if group_order:
        ordered: Dict[str, List[FieldDescriptor]] = {}
        for g in group_order:
            if g in groups:
                ordered[g] = groups.pop(g)
        # Append remaining groups in first-appearance order
        for g in groups:
            ordered[g] = groups[g]
        return ordered

    return groups


def nested_descriptors(
    desc: FieldDescriptor,
) -> List[FieldDescriptor]:
    """Reflect the nested model of a ``kind == 'nested'`` descriptor.

    Returns an empty list if the descriptor is not nested.
    """
    if desc.nested_model is None:
        return []
    return reflect_schema(desc.nested_model)
