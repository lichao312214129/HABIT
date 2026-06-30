# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License -- see the LICENSE file in the
# project root for the full text.

"""Registry of per-step path fields for bulk auto-fill when a project opens."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple

from habit.gui.step_integration import fill_on_project_open


@dataclass
class StepPathRegistration:
    """One workflow step's path textboxes registered for project auto-fill."""

    step_id: str
    field_names: List[str]
    components: List[Any] = field(default_factory=list)


_REGISTRY: List[StepPathRegistration] = []


def register_step_paths(
    step_id: str,
    field_names: List[str],
    components: List[Any],
) -> None:
    """
    Register path textboxes for a workflow step.

    Args:
        step_id: Workflow step identifier.
        field_names: Logical names aligned with ProjectContext.gui_updates_for_step.
        components: Gradio components updated on project open.
    """
    _REGISTRY.append(
        StepPathRegistration(
            step_id=step_id,
            field_names=list(field_names),
            components=list(components),
        )
    )


def all_path_components() -> List[Any]:
    """Flat list of all registered path components in registration order."""
    out: List[Any] = []
    for entry in _REGISTRY:
        out.extend(entry.components)
    return out


def fill_all_registered_paths(root: str) -> Tuple[Any, ...]:
    """
    Build gr.update values for every registered step path field.

    Args:
        root: Project root directory.

    Returns:
        Tuple of gr.update objects matching all_path_components() order.
    """
    updates: List[Any] = []
    for entry in _REGISTRY:
        updates.extend(fill_on_project_open(root, entry.step_id, entry.field_names))
    return tuple(updates)


def clear_registry() -> None:
    """Clear registrations (for tests)."""
    _REGISTRY.clear()
