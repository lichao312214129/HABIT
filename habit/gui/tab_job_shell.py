# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Shared Gradio job wiring for HABIT workflow tabs.

Centralizes the repeated submit/stop pattern, validation error formatting, and
optional workflow step hooks so schema-driven tabs stay small and consistent.
"""

from __future__ import annotations

from typing import Any, Callable, Generator, Iterable, List, Optional, Sequence, Union

import gradio as gr
from pydantic import ValidationError

from habit.gui.job_controls import (
    job_end_button_updates,
    job_start_button_updates,
    on_stop_job_click,
)
from habit.gui.utils import extract_validation_msgs, translate_pydantic_error

YieldFn = Callable[..., Generator[Any, None, None]]


def format_run_error(exc: Exception) -> str:
    """
    Turn a caught exception into a user-facing log message.

    Args:
        exc: Any exception raised during validate/run.

    Returns:
        str: Multi-line message suitable for the console log textbox.
    """
    if isinstance(exc, ValidationError):
        msgs = translate_pydantic_error(exc)
        return "⚠️ Validation errors:\n" + "\n".join(f"- {m}" for m in msgs)
    val_msgs = extract_validation_msgs(exc)
    if val_msgs:
        return "⚠️ Validation errors:\n" + "\n".join(f"- {m}" for m in val_msgs)
    return f"❌ Failed: {exc}"


def wire_standard_job(
    submit_btn: gr.Button,
    stop_btn: gr.Button,
    run_fn: YieldFn,
    inputs: Sequence[Any],
    outputs: Sequence[Any],
) -> None:
    """
    Attach the standard HABIT job button chain to a tab run function.

    Pattern: job_start → run (yielding) → job_end, plus stop handler.

    Args:
        submit_btn: Primary run button.
        stop_btn: Cancel button (initially disabled).
        run_fn: Generator function that yields log/output updates.
        inputs: Gradio inputs passed to ``run_fn``.
        outputs: Gradio outputs updated by ``run_fn``.
    """
    submit_btn.click(
        job_start_button_updates,
        outputs=[submit_btn, stop_btn],
    ).then(
        run_fn,
        inputs=list(inputs),
        outputs=list(outputs),
    ).then(
        job_end_button_updates,
        outputs=[submit_btn, stop_btn],
    )
    stop_btn.click(on_stop_job_click, inputs=[], outputs=[])


def wire_yaml_autoload(
    yaml_textbox: gr.Textbox,
    load_fn: Callable[[str], List[Any]],
    form_outputs: Iterable[Any],
) -> None:
    """
    Wire YAML path textbox → populate form widgets on change.

    Args:
        yaml_textbox: Textbox holding the config YAML path.
        load_fn: Callable returning a list of values/gr.update for form fields.
        form_outputs: Gradio components to update (same order as load_fn output).
    """
    yaml_textbox.change(load_fn, inputs=yaml_textbox, outputs=list(form_outputs))


def mark_step_if_project(project_root: str, step_id: str) -> None:
    """
    Mark a workflow step as running when a project root is active.

    Args:
        project_root: Current project root path (may be empty).
        step_id: Workflow step identifier (e.g. ``dicom_sort``).
    """
    if project_root:
        from habit.gui.project.step_hooks import mark_step_running

        mark_step_running(str(project_root), step_id)


def finalize_step_if_project(
    project_root: str,
    step_id: str,
    last_log: str,
    *,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    """
    Persist workflow step status after a background job completes.

    Args:
        project_root: Current project root path (may be empty).
        step_id: Workflow step identifier.
        last_log: Final console log text from the job.
        config_path: Optional saved YAML path.
        output_dir: Optional primary output directory.
    """
    if project_root:
        from habit.gui.project.step_hooks import finalize_step_from_log

        finalize_step_from_log(
            str(project_root),
            step_id,
            last_log,
            config_path=config_path,
            output_dir=output_dir,
        )
