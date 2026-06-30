# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Schema-driven dynamic form engine for HABIT GUI.

This package eliminates hardcoded GUI widgets. Instead of manually writing
``gr.Textbox(label="field_name")`` for every parameter, tabs declare which
Pydantic model they use, and the form engine automatically:

1. **Reflects** the Pydantic model into field descriptors (type, constraints,
   UI metadata from ``json_schema_extra``).
2. **Renders** the appropriate Gradio widgets (Dropdown for Literal, Checkbox
   for bool, Slider for bounded numbers, etc.).
3. **Collects** widget values back into a dict — no manual ``config_data = {}``
   assembly.
4. **Populates** widgets from a loaded YAML config dict.
5. **Wires** conditional visibility (``visible_if`` rules).

Quick start (in a tab)::

    from habit.gui.schema_form import SchemaForm, OverrideSpec

    form = SchemaForm.build(MyConfig, overrides={
        "model_name": OverrideSpec(choices=dynamic_model_list),
    })

    submit.click(
        fn=lambda *vals: run_pipeline(form.collect(*vals)),
        inputs=form.inputs(),
        outputs=[log],
    )

Adding a new parameter to the GUI is now zero-code: just add a ``Field()`` to
the Pydantic model and it appears automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from habit.gui.schema_form.reflect import (
    FieldDescriptor,
    group_descriptors,
    nested_descriptors,
    reflect_field,
    reflect_schema,
)
from habit.gui.schema_form.widgets import (
    OverrideSpec,
    create_widget,
    render_fields,
    render_model,
    render_nested,
)
from habit.gui.schema_form.collector import (
    build_gr_updates,
    collect_from_mapping,
    collect_values,
    populate_widgets,
)
from habit.gui.schema_form.conditions import (
    compute_visibility,
    evaluate_visible_if,
    visibility_updates,
    wire_conditional_visibility,
)
from habit.gui.schema_form.registry import (
    get_model_choices,
    get_prep_step_choices,
    get_selector_choices,
)

__all__ = [
    # High-level API
    "SchemaForm",
    # Reflection
    "FieldDescriptor",
    "reflect_schema",
    "reflect_field",
    "group_descriptors",
    "nested_descriptors",
    # Widgets
    "OverrideSpec",
    "create_widget",
    "render_fields",
    "render_model",
    "render_nested",
    # Collector
    "collect_values",
    "collect_from_mapping",
    "populate_widgets",
    "build_gr_updates",
    # Conditions
    "evaluate_visible_if",
    "compute_visibility",
    "visibility_updates",
    "wire_conditional_visibility",
]


# ---------------------------------------------------------------------------
# SchemaForm — the high-level API that tabs use
# ---------------------------------------------------------------------------
@dataclass
class SchemaForm:
    """A schema-driven form, rendered in the current Gradio context.

    Created via :meth:`build`. Encapsulates descriptors, widgets, and provides
    convenience methods for collecting/populating values and wiring visibility.
    """

    model_cls: Type
    """The Pydantic model class this form is based on."""

    descriptors: List[FieldDescriptor]
    """Field descriptors in render order (excluding nested model sub-fields)."""

    widgets: Dict[str, Any]
    """Mapping of field name → Gradio component.

    For nested model fields, the value is a sub-dict of
    ``{sub_field_name: gr.Component}``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        model_cls: Type,
        *,
        overrides: Optional[Dict[str, OverrideSpec]] = None,
        group_order: Optional[List[str]] = None,
        open_groups: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
        path_browse_fn: Optional[Callable[[FieldDescriptor], Any]] = None,
        path_picker: Any = None,
    ) -> "SchemaForm":
        """Reflect a Pydantic model and render its widgets in the current context.

        Args:
            model_cls: A Pydantic ``BaseModel`` subclass (e.g. ``DicomSortConfig``).
            overrides: Per-field overrides for dynamic choices, custom widgets, etc.
            group_order: Explicit ordering of group names for accordion layout.
            open_groups: Groups that should start expanded.
            exclude: Field names to skip (e.g. internal or deprecated fields).
            path_browse_fn: Optional callback for path field browse buttons.
                When provided, each path field gets a "Browse" button.
            path_picker: Optional PathPickerRegistry — when provided, path
                fields get a Browse button wired via the picker (declarative
                mode, used for native/web path picker integration).

        Returns:
            A :class:`SchemaForm` ready for ``inputs()``, ``collect()``, etc.
        """
        descriptors = reflect_schema(model_cls)
        if exclude:
            descriptors = [d for d in descriptors if d.name not in exclude]
        widgets = render_fields(
            descriptors,
            overrides=overrides,
            group_order=group_order,
            open_groups=open_groups,
            path_browse_fn=path_browse_fn,
            path_picker=path_picker,
        )
        return cls(
            model_cls=model_cls,
            descriptors=descriptors,
            widgets=widgets,
        )

    # ------------------------------------------------------------------
    # Input/output helpers for Gradio event binding
    # ------------------------------------------------------------------
    def inputs(self) -> list:
        """Return a flat list of Gradio components for ``inputs=[...]``.

        Only top-level (non-nested) widgets are included. Nested model fields
        are handled separately by the tab (or via a sub-form).
        """
        result: list = []
        for desc in self.descriptors:
            widget = self.widgets.get(desc.name)
            if widget is None:
                continue
            if isinstance(widget, dict):
                # Nested model — skip (tab handles separately)
                continue
            result.append(widget)
        return result

    def outputs(self) -> list:
        """Alias for :meth:`inputs`, for use as ``outputs=[...]`` in load events."""
        return self.inputs()

    # ------------------------------------------------------------------
    # Value collection (widget values → dict)
    # ------------------------------------------------------------------
    def collect(self, *values: Any) -> Dict[str, Any]:
        """Collect positional widget values into a config dict.

        The number of ``values`` must match the number of widgets in
        :meth:`inputs`.

        Returns:
            A dict suitable for ``MyConfig(**result)`` validation.
        """
        flat_descs = [d for d in self.descriptors if not d.is_nested]
        return collect_values(flat_descs, list(values))

    def collect_from_dict(self, value_map: Dict[str, Any]) -> Dict[str, Any]:
        """Collect values from a name→value mapping.

        Useful when widget values are accessed by component reference rather
        than by positional list.
        """
        flat_descs = [d for d in self.descriptors if not d.is_nested]
        return collect_from_mapping(flat_descs, value_map)

    # ------------------------------------------------------------------
    # Value population (dict → widget values for YAML loading)
    # ------------------------------------------------------------------
    def populate(self, config_dict: Dict[str, Any]) -> List[Any]:
        """Build a list of values from a config dict, for loading into widgets.

        The returned list is in the same order as :meth:`inputs`, suitable for
        ``outputs=form.outputs()`` in a load event.
        """
        flat_descs = [d for d in self.descriptors if not d.is_nested]
        return populate_widgets(flat_descs, config_dict)

    def populate_updates(self, config_dict: Dict[str, Any]) -> List[Any]:
        """Like :meth:`populate` but returns ``gr.update(value=...)`` objects."""
        flat_descs = [d for d in self.descriptors if not d.is_nested]
        return build_gr_updates(flat_descs, config_dict)

    # ------------------------------------------------------------------
    # Conditional visibility
    # ------------------------------------------------------------------
    def wire_visibility(self) -> list:
        """Wire ``visible_if`` conditional visibility for all dependent fields.

        Must be called after all widgets are created. Returns a list of Gradio
        event references.
        """
        flat_widgets: Dict[str, Any] = {
            name: w for name, w in self.widgets.items() if not isinstance(w, dict)
        }
        flat_descs = [d for d in self.descriptors if not d.is_nested]
        return wire_conditional_visibility(flat_widgets, flat_descs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def path_fields(self) -> List[FieldDescriptor]:
        """Return descriptors for path-type fields (for browse button wiring)."""
        return [d for d in self.descriptors if d.is_path]

    def widget(self, name: str) -> Any:
        """Get a widget by field name. Returns ``None`` if not found."""
        return self.widgets.get(name)

    def descriptor(self, name: str) -> Optional[FieldDescriptor]:
        """Get a descriptor by field name. Returns ``None`` if not found."""
        for d in self.descriptors:
            if d.name == name:
                return d
        return None
