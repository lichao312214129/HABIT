# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Conditional visibility: evaluate ``visible_if`` rules from FieldDescriptors.

When a field declares ``json_schema_extra={"visible_if": {"other_field": "value"}}``,
it should only be visible when ``other_field`` equals ``"value"``. This module
provides the evaluation logic and a helper to wire Gradio change events.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

import gradio as gr

from habit.gui.schema_form.reflect import FieldDescriptor


def evaluate_visible_if(
    visible_if: Optional[Dict[str, Any]],
    current_values: Dict[str, Any],
) -> bool:
    """Evaluate a ``visible_if`` rule against current field values.

    Args:
        visible_if: ``{field_name: expected_value}`` or ``None``.
        current_values: Current values of all fields in the form.

    Returns:
        ``True`` if the field should be visible, ``False`` otherwise.
        A ``None`` rule always returns ``True``.
    """
    if not visible_if:
        return True
    for field_name, expected in visible_if.items():
        actual = current_values.get(field_name)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False
    return True


def compute_visibility(
    descriptors: List[FieldDescriptor],
    current_values: Dict[str, Any],
) -> Dict[str, bool]:
    """Compute visibility for all fields given current values.

    Returns:
        ``{field_name: should_be_visible}``.
    """
    result: Dict[str, bool] = {}
    for desc in descriptors:
        result[desc.name] = evaluate_visible_if(desc.visible_if, current_values)
    return result


def visibility_updates(
    descriptors: List[FieldDescriptor],
    changed_field: str,
    new_value: Any,
    all_values: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute which widgets need visibility updates after a field changes.

    Args:
        descriptors: All field descriptors.
        changed_field: Name of the field that just changed.
        new_value: New value of the changed field.
        all_values: Current values of all fields (will be updated with new_value).

    Returns:
        ``{field_name: gr.update(visible=...)}`` for fields whose visibility
        changed. Only fields with ``visible_if`` rules referencing the changed
        field are included.
    """
    # Update the values dict with the new value
    updated_values = {**all_values, changed_field: new_value}

    updates: Dict[str, Any] = {}
    for desc in descriptors:
        if desc.visible_if is None:
            continue
        # Check if this field's visibility depends on the changed field
        if changed_field not in desc.visible_if:
            continue
        should_show = evaluate_visible_if(desc.visible_if, updated_values)
        updates[desc.name] = gr.update(visible=should_show)

    return updates


def wire_conditional_visibility(
    widgets: Dict[str, gr.Component],
    descriptors: List[FieldDescriptor],
) -> List:
    """Wire Gradio change events for all ``visible_if`` conditional fields.

    This sets up ``.change()`` listeners on every field that is referenced by
    any ``visible_if`` rule, so that when it changes, dependent fields update
    their visibility.

    Must be called **after** all widgets are created and **before** the form
    is used.

    Args:
        widgets: Mapping of field name → Gradio component.
        descriptors: Field descriptors (must match widgets).

    Returns:
        A list of event references (for cleanup/testing).
    """
    # Build a reverse index: which fields affect which other fields' visibility
    dependents: Dict[str, List[str]] = {}
    for desc in descriptors:
        if desc.visible_if is None:
            continue
        for ref_field in desc.visible_if:
            if ref_field not in dependents:
                dependents[ref_field] = []
            dependents[ref_field].append(desc.name)

    events: List = []

    for trigger_field, affected_fields in dependents.items():
        trigger_widget = widgets.get(trigger_field)
        if trigger_widget is None:
            continue

        # Collect all widgets that might need visibility updates
        affected_widgets = [widgets[name] for name in affected_fields if name in widgets]
        if not affected_widgets:
            continue

        # Build the callback
        def make_callback(
            trigger_name: str,
            affected_names: List[str],
            all_descs: List[FieldDescriptor],
        ):
            def callback(trigger_value: Any, *all_values):
                # Reconstruct current values from all form widgets
                # all_values comes from the inputs we specify
                current: Dict[str, Any] = {trigger_name: trigger_value}
                # We only need the trigger value for simple visible_if rules
                updates = {}
                for desc in all_descs:
                    if desc.visible_if is None:
                        continue
                    if trigger_name not in desc.visible_if:
                        continue
                    should_show = evaluate_visible_if(desc.visible_if, current)
                    updates[desc.name] = gr.update(visible=should_show)
                # Return updates in the order of affected_widgets
                result = []
                for name in affected_names:
                    if name in updates:
                        result.append(updates[name])
                    else:
                        result.append(gr.update())  # no change
                return result
            return callback

        callback = make_callback(trigger_field, affected_fields, descriptors)

        # We need the trigger widget's value and nothing else for simple rules
        event = trigger_widget.change(
            fn=callback,
            inputs=[trigger_widget],
            outputs=affected_widgets,
        )
        events.append(event)

    return events
