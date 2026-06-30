# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License -- see the LICENSE file in the
# project root for the full text.

"""Workflow step lifecycle hooks — persist run status to habit_project.json."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from habit.gui.project_manager import load_project, save_project
from habit.gui.workflow_state import WORKFLOW_STEPS, set_step_status

# Default relative paths for new / migrated projects
DEFAULT_PATHS: Dict[str, str] = {
    "raw": "01_raw",
    "dicom_sorted": "01_raw/sorted_dicom",
    "preprocessed_root": "02_preprocessed",
    "preprocessed": "02_preprocessed/processed_images",
    "habitat_out": "03_habitat",
    "features": "04_features",
    "ml": "05_ml",
    "compare": "05_ml/comparison",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_workflow_state() -> Dict[str, Dict[str, str]]:
    return {step: {"status": "pending"} for step in WORKFLOW_STEPS}


def migrate_project_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure older habit_project.json files have new path keys and workflow_state.

    Args:
        meta: Loaded project metadata.

    Returns:
        Dict[str, Any]: Migrated metadata (saved when changes were applied).
    """
    changed = False
    doc = dict(meta)
    paths = dict(doc.get("paths") or {})

    for key, default_rel in DEFAULT_PATHS.items():
        if key not in paths:
            # Legacy: preprocessed pointed at processed_images — keep it, add root
            if key == "preprocessed_root" and "preprocessed" in paths:
                rel = str(paths["preprocessed"])
                if rel.endswith("/processed_images") or rel.endswith("\\processed_images"):
                    paths[key] = rel.rsplit("/", 1)[0].rsplit("\\", 1)[0]
                elif rel.endswith("processed_images"):
                    paths[key] = "02_preprocessed"
                else:
                    paths[key] = default_rel
            else:
                paths[key] = default_rel
            changed = True

    doc["paths"] = paths

    if "workflow_state" not in doc or not isinstance(doc.get("workflow_state"), dict):
        doc["workflow_state"] = _default_workflow_state()
        changed = True
    else:
        ws = dict(doc["workflow_state"])
        for step in WORKFLOW_STEPS:
            if step not in ws:
                ws[step] = {"status": "pending"}
                changed = True
        doc["workflow_state"] = ws

    if "last_run" not in doc or not isinstance(doc.get("last_run"), dict):
        doc["last_run"] = {}
        changed = True

    if changed:
        save_project(doc)
    return doc


def mark_step_running(root: str, step: str) -> None:
    """Record that a step has started."""
    if not root or step not in WORKFLOW_STEPS:
        return
    set_step_status(root, step, "in_progress", started_at=_utc_now_iso())


def mark_step_completed(
    root: str,
    step: str,
    *,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record successful step completion and update last_run metadata.

    Args:
        root: Project root path.
        step: Workflow step id.
        config_path: GUI-generated YAML path, if any.
        output_dir: Primary output directory for the step.
        extra: Additional fields stored under last_run[step].
    """
    if not root or step not in WORKFLOW_STEPS:
        return
    payload: Dict[str, Any] = {
        "finished_at": _utc_now_iso(),
        "config_path": config_path or "",
        "output_dir": output_dir or "",
    }
    if extra:
        payload.update(extra)
    set_step_status(root, step, "completed", **payload)

    meta = load_project(root)
    if not meta:
        return
    last_run = dict(meta.get("last_run") or {})
    last_run[step] = payload
    meta["last_run"] = last_run
    save_project(meta)


def mark_step_failed(root: str, step: str, error: Optional[str] = None) -> None:
    """Record step failure."""
    if not root or step not in WORKFLOW_STEPS:
        return
    set_step_status(
        root,
        step,
        "failed",
        finished_at=_utc_now_iso(),
        error=(error or "")[:2000],
    )


def detect_success_from_log(log_text: str) -> bool:
    """
    Heuristic: treat a run as successful when the log shows completion markers.

    Args:
        log_text: Final console log text from the GUI job runner.

    Returns:
        bool: True when success markers are present and no fatal error prefix.
    """
    if not log_text:
        return False
    text = log_text.strip()
    if text.startswith("❌") or "Validation errors:" in text:
        return False
    markers = (
        "Task completed successfully",
        "✅",
        "completed successfully",
        "Pipeline finished",
        "Analysis complete",
    )
    return any(m in text for m in markers)


def finalize_step_from_log(root: str, step: str, log_text: str, **completed_extra: Any) -> None:
    """
    Update workflow state after a GUI job based on the final log text.

    Args:
        root: Project root (skipped when empty).
        step: Workflow step id.
        log_text: Final streamed log content.
        **completed_extra: Extra fields passed to mark_step_completed.
    """
    if not root or not str(root).strip():
        return
    if detect_success_from_log(log_text):
        mark_step_completed(root, step, **completed_extra)
    elif log_text and not log_text.strip().startswith("💾"):
        mark_step_failed(root, step, error=log_text[:500])
