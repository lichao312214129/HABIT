# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Shared Gradio run/stop button helpers for HABIT GUI tabs."""

from __future__ import annotations

from typing import Any, Tuple

import gradio as gr

from habit.utils.job_cancel import request_job_cancel


def job_start_button_updates() -> Tuple[Any, Any]:
    """Disable submit and enable stop when a background job starts."""
    return gr.update(interactive=False), gr.update(interactive=True)


def job_end_button_updates() -> Tuple[Any, Any]:
    """Re-enable submit and disable stop when a background job ends."""
    return gr.update(interactive=True), gr.update(interactive=False)


def on_stop_job_click() -> None:
    """Gradio callback: request cooperative cancellation of the running job."""
    request_job_cancel()
