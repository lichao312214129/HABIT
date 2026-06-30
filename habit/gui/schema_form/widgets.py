# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Widget factory: FieldDescriptor → Gradio component.

Given a list of :class:`~habit.gui.schema_form.reflect.FieldDescriptor`, this
module creates the appropriate Gradio widgets **in the current Gradio context**.
No field names are hardcoded — the widget type is determined purely from the
descriptor's ``kind``, ``choices``, constraints, and UI metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gradio as gr

from habit.gui.schema_form.conditions import evaluate_visible_if
from habit.gui.schema_form.reflect import (
    FieldDescriptor,
    group_descriptors,
    reflect_schema,
)


# ---------------------------------------------------------------------------
# Override spec — allows tab-level customization of auto-generated widgets
# ---------------------------------------------------------------------------
@dataclass
class OverrideSpec:
    """Override the default widget behavior for a specific field.

    This is how tabs inject dynamic data (e.g. model names from a registry)
    into the schema-driven form without hardcoding.
    """

    choices: Optional[List[Any]] = None
    """Override dropdown/checkbox-group choices (e.g. from a dynamic registry)."""

    widget: Optional[str] = None
    """Force a widget type: 'slider', 'textarea', 'path', 'path_dir', 'list_editor'."""

    visible: bool = True
    """Initial visibility (used for fields that start hidden)."""

    label: Optional[str] = None
    """Override the display label."""

    value: Any = None
    """Override the default value."""

    interactive: Optional[bool] = None
    """Force interactive/disabled state."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _default_label(desc: FieldDescriptor) -> str:
    """Build a human-friendly label from the field name."""
    return desc.label or desc.name.replace("_", " ").capitalize()


def _help_text(desc: FieldDescriptor) -> Optional[str]:
    """Combine description and help_text for the Gradio help/tooltip."""
    parts = []
    if desc.description:
        parts.append(desc.description)
    if desc.help_text:
        parts.append(desc.help_text)
    return " | ".join(parts) if parts else None


def _resolve_default(desc: FieldDescriptor, override: Optional[OverrideSpec]) -> Any:
    if override and override.value is not None:
        return override.value
    return desc.default


def _resolve_choices(desc: FieldDescriptor, override: Optional[OverrideSpec]) -> Optional[List[Any]]:
    if override and override.choices is not None:
        return override.choices
    return desc.choices


# ---------------------------------------------------------------------------
# Single-field widget creation
# ---------------------------------------------------------------------------
def create_widget(
    desc: FieldDescriptor,
    override: Optional[OverrideSpec] = None,
) -> gr.Component:
    """Create a single Gradio widget for a field descriptor.

    The widget is created in the **current Gradio context** (i.e. it must be
    called inside a ``with gr.Group():`` or similar block).

    For nested models, use :func:`render_nested` instead, which creates an
    Accordion with child widgets.
    """
    label = _default_label(desc)
    help_txt = _help_text(desc)
    default_val = _resolve_default(desc, override)
    choices = _resolve_choices(desc, override)
    visible = override.visible if override else True
    interactive = override.interactive if override else None

    # --- Path field → Textbox (browse button is added at tab level) ---
    if desc.is_path:
        return gr.Textbox(
            label=label,
            value=default_val or "",
            info=help_txt,
            placeholder=f"{'Directory' if desc.path_type == 'dir' else 'File'} path",
            interactive=interactive if interactive is not None else True,
            visible=visible,
            scale=4,
        )

    # --- Literal → Dropdown ---
    if desc.kind == "literal" and choices is not None:
        # If choices are few bool-like, could use Radio; default Dropdown
        return gr.Dropdown(
            label=label,
            choices=choices,
            value=default_val if default_val in choices else (choices[0] if choices else None),
            info=help_txt,
            interactive=interactive if interactive is not None else True,
            visible=visible,
            allow_custom_value=False,
        )

    # --- Bool → Checkbox ---
    if desc.kind == "bool":
        return gr.Checkbox(
            label=label,
            value=bool(default_val) if default_val is not None else False,
            info=help_txt,
            interactive=interactive if interactive is not None else True,
            visible=visible,
        )

    # --- Int / Float → Number or Slider ---
    if desc.kind in ("int", "float"):
        is_slider = (
            override and override.widget == "slider"
        ) or (
            desc.widget == "slider"
            and desc.ge is not None
            and desc.le is not None
        )
        if is_slider and desc.ge is not None and desc.le is not None:
            step = 1 if desc.kind == "int" else (desc.le - desc.ge) / 100.0
            return gr.Slider(
                label=label,
                minimum=float(desc.ge),
                maximum=float(desc.le),
                value=float(default_val) if default_val is not None else float(desc.ge),
                step=step,
                info=help_txt,
                interactive=interactive if interactive is not None else True,
            visible=visible,
            )
        return gr.Number(
            label=label,
            value=default_val if default_val is not None else 0,
            info=help_txt,
            precision=0 if desc.kind == "int" else None,
            interactive=interactive if interactive is not None else True,
            visible=visible,
        )

    # --- List → Textbox (comma-separated) or CheckboxGroup ---
    if desc.kind == "list":
        # Override choices -> CheckboxGroup
        if override and override.choices is not None:
            return gr.CheckboxGroup(
                label=label,
                choices=override.choices,
                value=default_val if isinstance(default_val, list) else [],
                info=help_txt,
                interactive=interactive if interactive is not None else True,
                visible=visible,
            )
        # If inner type is a Literal, use CheckboxGroup
        from habit.gui.schema_form.reflect import _extract_literal_values
        inner_choices = None
        if desc.inner_type is not None:
            list_args = _get_list_item_type(desc.inner_type)
            if list_args is not None:
                inner_choices = _extract_literal_values(list_args)

        if inner_choices is not None:
            return gr.CheckboxGroup(
                label=label,
                choices=inner_choices,
                value=default_val if isinstance(default_val, list) else [],
                info=help_txt,
                interactive=interactive if interactive is not None else True,
            visible=visible,
            )
        # Otherwise, comma-separated textbox
        default_str = ", ".join(str(v) for v in default_val) if isinstance(default_val, list) else ""
        return gr.Textbox(
            label=label,
            value=default_str,
            info=help_txt,
            placeholder="Comma-separated values",
            interactive=interactive if interactive is not None else True,
            visible=visible,
        )

    # --- Dict → Textbox (JSON string) ---
    if desc.kind == "dict":
        import json
        default_str = json.dumps(default_val, indent=2) if default_val else "{}"
        return gr.Textbox(
            label=label,
            value=default_str,
            info=help_txt,
            placeholder="JSON object",
            lines=3,
            interactive=interactive if interactive is not None else True,
            visible=visible,
        )

    # --- Nested model → should use render_nested, but fallback to JSON ---
    if desc.kind == "nested":
        import json
        default_str = json.dumps(default_val, indent=2, default=str) if default_val else "{}"
        return gr.Textbox(
            label=label,
            value=default_str,
            info=help_txt,
            placeholder="Nested configuration (JSON)",
            lines=5,
            interactive=interactive if interactive is not None else True,
            visible=visible,
        )

    # --- Default: str → Textbox ---
    widget_type = override.widget if override else desc.widget
    if widget_type == "textarea":
        return gr.Textbox(
            label=label,
            value=default_val or "",
            info=help_txt,
            lines=3,
            interactive=interactive if interactive is not None else True,
            visible=visible,
        )
    return gr.Textbox(
        label=label,
        value=default_val if default_val is not None else "",
        info=help_txt,
        interactive=interactive if interactive is not None else True,
    )


def _get_list_item_type(inner_type: Any) -> Optional[Any]:
    """Extract the item type from ``List[X]`` annotation."""
    from typing import get_args, get_origin
    if get_origin(inner_type) is list:
        args = get_args(inner_type)
        return args[0] if args else None
    return None


# ---------------------------------------------------------------------------
# Batch rendering
# ---------------------------------------------------------------------------
def render_fields(
    descriptors: List[FieldDescriptor],
    *,
    overrides: Optional[Dict[str, OverrideSpec]] = None,
    group_order: Optional[List[str]] = None,
    open_groups: Optional[set] = None,
    path_browse_fn: Optional[Callable[["FieldDescriptor"], Any]] = None,
    path_picker: Any = None,
) -> Dict[str, gr.Component]:
    """Render Gradio widgets for all descriptors in the current context.

    Fields are grouped into ``gr.Accordion`` sections based on their ``group``
    attribute. Within each group, fields are sorted by ``order``.

    Args:
        descriptors: Field list from :func:`reflect_schema`.
        overrides: Optional per-field overrides (e.g. dynamic choices).
        group_order: Explicit ordering of group names.
        open_groups: Set of group names that should start expanded.
        path_browse_fn: Optional callback for path fields. When provided,
            each path field gets a "Browse" button that calls this function
            with the :class:`FieldDescriptor` and expects a path string back.

    Returns:
        Mapping of ``{field_name: gr.Component}``. Nested model fields map to
        a sub-dict of ``{sub_field_name: gr.Component}``.
    """
    overrides = overrides or {}
    open_groups = open_groups or set()
    widgets: Dict[str, Any] = {}
    groups = group_descriptors(descriptors, group_order)

    for group_name, group_descs in groups.items():
        is_open = group_name in open_groups or group_name == "General"
        with gr.Accordion(group_name, open=is_open):
            for desc in group_descs:
                override = overrides.get(desc.name)
                # Compute initial visibility for visible_if fields
                if desc.visible_if:
                    defaults = {d.name: d.default for d in descriptors}
                    should_show = evaluate_visible_if(desc.visible_if, defaults)
                    if not should_show:
                        if override is None:
                            override = OverrideSpec()
                        override.visible = False
                if desc.kind == "nested" and desc.nested_model is not None:
                    # Render nested model in a sub-accordion
                    sub_widgets = render_nested(desc, overrides=overrides)
                    widgets[desc.name] = sub_widgets
                elif desc.is_path and (path_browse_fn is not None or path_picker is not None):
                    # Path field with browse button: Row(Textbox, Button)
                    with gr.Row():
                        widget = create_widget(desc, override=override)
                        browse_btn = gr.Button("Browse", scale=1, size="sm")
                        if path_picker is not None:
                            pick_mode = "folder" if desc.path_type == "dir" else "file"
                            path_picker.add(browse_btn, widget, pick=pick_mode)
                        else:
                            _make_browse_handler(browse_btn, widget, desc, path_browse_fn)
                    widgets[desc.name] = widget
                else:
                    widget = create_widget(desc, override=override)
                    widgets[desc.name] = widget

    return widgets


def _make_browse_handler(
    btn: gr.Button,
    target: gr.Component,
    desc: "FieldDescriptor",
    browse_fn: Callable[["FieldDescriptor"], Any],
) -> None:
    """Wire a browse button to call *browse_fn* and fill *target*."""
    btn.click(fn=lambda d=desc: browse_fn(d), outputs=target)


def render_nested(
    desc: FieldDescriptor,
    *,
    overrides: Optional[Dict[str, OverrideSpec]] = None,
) -> Dict[str, gr.Component]:
    """Render a nested model field as an Accordion with child widgets.

    Args:
        desc: A descriptor with ``kind == 'nested'`` and ``nested_model`` set.
        overrides: Optional overrides, keyed by sub-field name.

    Returns:
        Mapping of ``{sub_field_name: gr.Component}``.
    """
    assert desc.nested_model is not None, "render_nested requires a nested_model"
    sub_descriptors = reflect_schema(desc.nested_model)
    label = _default_label(desc)
    with gr.Accordion(label, open=False):
        return render_fields(sub_descriptors, overrides=overrides)


def render_model(
    model_cls: Any,
    *,
    overrides: Optional[Dict[str, OverrideSpec]] = None,
    group_order: Optional[List[str]] = None,
    open_groups: Optional[set] = None,
    exclude: Optional[set] = None,
    path_browse_fn: Optional[Callable[["FieldDescriptor"], Any]] = None,
    path_picker: Any = None,
) -> Dict[str, gr.Component]:
    """Top-level convenience: reflect a model and render its widgets.

    Args:
        model_cls: A Pydantic BaseModel subclass.
        overrides: Per-field overrides.
        group_order: Explicit group ordering.
        open_groups: Groups that start expanded.
        exclude: Field names to skip.
        path_browse_fn: Optional callback for path field browse buttons.
        path_picker: Optional PathPickerRegistry — when provided, path fields
            get a Browse button wired via ``picker.add(btn, widget, pick=...)``.

    Returns:
        Mapping of ``{field_name: gr.Component}``.
    """
    descriptors = reflect_schema(model_cls)
    if exclude:
        descriptors = [d for d in descriptors if d.name not in exclude]
    return render_fields(
        descriptors,
        overrides=overrides,
        group_order=group_order,
        open_groups=open_groups,
        path_browse_fn=path_browse_fn,
        path_picker=path_picker,
    )
