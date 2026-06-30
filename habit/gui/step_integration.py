# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License -- see the LICENSE file in the
# project root for the full text.

"""Shared helpers for wiring workflow tabs to ProjectContext."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from habit.gui.project.context import ProjectContext, gr_updates_for_paths


def register_project_path_fill(
    project_root_state: Any,
    step_id: str,
    field_names: List[str],
    components: List[Any],
    *,
    extra_fill_fn: Optional[Callable[[ProjectContext], Dict[str, str]]] = None,
) -> gr.Button:
    """
    Add a "Fill from project" button and wire it to ProjectContext paths.

    Args:
        project_root_state: gr.State holding the active project root.
        step_id: Workflow step identifier.
        field_names: Logical field names matching components order.
        components: Gradio Textbox (or compatible) components to update.
        extra_fill_fn: Optional hook to merge extra field values.

    Returns:
        gr.Button: The created fill button (already wired).
    """
    btn = gr.Button("Fill paths from project", size="sm", variant="secondary")

    def _fill(root: str) -> Tuple[Any, ...]:
        ctx = ProjectContext.load(root)
        if ctx is None:
            return tuple(gr.update() for _ in field_names)
        values = ctx.gui_updates_for_step(step_id)
        if extra_fill_fn is not None:
            values.update(extra_fill_fn(ctx))
        return gr_updates_for_paths(values, field_names)

    btn.click(_fill, inputs=[project_root_state], outputs=components)
    return btn


def fill_on_project_open(
    root: str,
    step_id: str,
    field_names: List[str],
) -> Tuple[Any, ...]:
    """
    Build gr.update values when a project is opened (used by app.py chains).

    Args:
        root: Project root path.
        step_id: Workflow step id.
        field_names: Fields to fill in order.

    Returns:
        Tuple of gr.update for each field.
    """
    ctx = ProjectContext.load(root)
    if ctx is None:
        return tuple(gr.update() for _ in field_names)
    values = ctx.gui_updates_for_step(step_id)
    return gr_updates_for_paths(values, field_names)
