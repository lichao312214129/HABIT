# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License -- see the LICENSE file.

"""Workflow state management for HABIT GUI.

Tracks the status of each workflow step and persists state to
``habit_project.json``, extending project_manager with step-level tracking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from habit.gui.project_manager import (
    default_project_dict,
    load_project,
    save_project,
)

# Ordered workflow step identifiers
WORKFLOW_STEPS: List[str] = [
    "dicom_sort",
    "preprocess",
    "habitat",
    "extract",
    "ml",
    "compare",
]

STEP_LABELS: Dict[str, str] = {
    "dicom_sort": "DICOM Sort",
    "preprocess": "Preprocessing",
    "habitat": "Habitat Analysis",
    "extract": "Feature Extraction",
    "ml": "Machine Learning",
    "compare": "Model Comparison",
}

STEP_DESCRIPTIONS: Dict[str, str] = {
    "dicom_sort": "Sort and rename raw DICOM files into a structured layout.",
    "preprocess": "Register, resample, and normalize images for analysis.",
    "habitat": "Cluster tumor voxels into habitats from multi-parametric images.",
    "extract": "Extract radiomics and habitat features to CSV tables.",
    "ml": "Train and evaluate predictive models on feature tables.",
    "compare": "Compare multiple models (ROC, DCA, calibration, DeLong).",
}

STEP_DEPENDENCIES: Dict[str, List[str]] = {
    "dicom_sort": [],
    "preprocess": [],
    "habitat": [],
    "extract": [],
    "ml": [],
    "compare": [],
}

STEP_PATH_KEYS: Dict[str, str] = {
    "dicom_sort": "dicom_sorted",
    "preprocess": "preprocessed_root",
    "habitat": "habitat_out",
    "extract": "features",
    "ml": "ml",
    "compare": "compare",
}

OPTIONAL_STEPS = {"extract", "ml", "compare"}

VALID_STATUSES = {"pending", "in_progress", "completed", "skipped", "failed"}

_STATUS_ICONS = {
    "pending": "\u25cb",
    "in_progress": "\u25cf",
    "completed": "\u2713",
    "skipped": "\u2212",
    "failed": "\u2717",
}


def get_workflow_state(root: str) -> Dict[str, Any]:
    """Load the full workflow-state dict from project metadata."""
    meta = load_project(root)
    if not meta:
        return {}
    return meta.get("workflow_state", {})


def get_step_status(root: str, step: str) -> str:
    """Return the status string for a single step (default ``pending``)."""
    state = get_workflow_state(root)
    return state.get(step, {}).get("status", "pending")


def get_step_info(root: str, step: str) -> Dict[str, Any]:
    """Return the full info dict for a step."""
    state = get_workflow_state(root)
    return dict(state.get(step, {}))


def set_step_status(root: str, step: str, status: str, **extra: Any) -> None:
    """Update a step's status (and arbitrary extra fields) and persist."""
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}")
    if step not in WORKFLOW_STEPS:
        raise ValueError(f"Unknown step '{step}'. Must be one of {WORKFLOW_STEPS}")
    meta = load_project(root)
    if not meta:
        meta = default_project_dict(name=root.rstrip("/").split("/")[-1], root=root)
    ws = dict(meta.get("workflow_state", {}))
    step_state = dict(ws.get(step, {}))
    step_state["status"] = status
    step_state.update(extra)
    ws[step] = step_state
    meta["workflow_state"] = ws
    save_project(meta)


def can_run_step(root: str, step: str) -> bool:
    """Check whether a step's dependencies are satisfied."""
    deps = STEP_DEPENDENCIES.get(step, [])
    for dep in deps:
        if get_step_status(root, dep) not in ("completed", "skipped"):
            return False
    return True


def get_step_output_path(root: str, step: str, meta: Optional[Dict] = None) -> Optional[str]:
    """Return the output directory for a step, if known."""
    doc = meta or load_project(root)
    if not doc:
        return None
    paths = doc.get("paths", {})
    key = STEP_PATH_KEYS.get(step)
    if key and key in paths:
        from pathlib import Path

        return str(Path(root) / paths[key])
    return None


def render_stepper_html(root: str, active_step: int = 0) -> str:
    """Generate HTML for the vertical workflow stepper sidebar.

    Args:
        root: Project root path (may be empty before a project is opened).
        active_step: 1-based index of the currently selected step (0 = none).

    Returns:
        HTML string for the stepper.
    """
    items: List[str] = []
    for i, step_id in enumerate(WORKFLOW_STEPS):
        status = get_step_status(root, step_id) if root else "pending"
        icon = _STATUS_ICONS.get(status, "\u25cb")
        label = STEP_LABELS[step_id]
        desc = STEP_DESCRIPTIONS[step_id]
        num = i + 1
        css = f"habit-step-item {status}"
        if num == active_step:
            css += " active"
        optional_tag = " (optional)" if step_id in OPTIONAL_STEPS else ""
        items.append(
            f'<div class="{css}" data-step="{num}">'
            f'<span class="habit-step-num">{num}</span>'
            f'<span class="habit-step-icon">{icon}</span>'
            f'<span class="habit-step-text">'
            f'<span class="habit-step-label">{label}{optional_tag}</span>'
            f'<span class="habit-step-desc">{desc}</span>'
            f"</span></div>"
        )
    return f'<div class="habit-stepper">{"".join(items)}</div>'


def get_progress_summary(root: str) -> Dict[str, Any]:
    """Return a summary of overall workflow progress for dashboard display."""
    state = get_workflow_state(root)
    total = len(WORKFLOW_STEPS)
    completed = sum(
        1 for s in WORKFLOW_STEPS if state.get(s, {}).get("status") == "completed"
    )
    in_progress = sum(
        1 for s in WORKFLOW_STEPS if state.get(s, {}).get("status") == "in_progress"
    )
    failed = sum(
        1 for s in WORKFLOW_STEPS if state.get(s, {}).get("status") == "failed"
    )
    return {
        "total": total,
        "completed": completed,
        "in_progress": in_progress,
        "failed": failed,
        "pending": total - completed - in_progress - failed,
        "percent": round(completed / total * 100) if total else 0,
    }
