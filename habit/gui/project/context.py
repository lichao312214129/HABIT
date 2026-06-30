# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License -- see the LICENSE file in the
# project root for the full text.

"""
Unified project context for the HABIT GUI.

Resolves standard folder layout and step-to-step path chaining so each workflow
tab receives correct absolute paths aligned with core pipeline conventions.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from habit.gui.project.step_hooks import migrate_project_meta
from habit.gui.project_manager import load_project, resolve_project_path, normalize_project_root
from habit.gui.utils import user_visible_path
from habit.utils.docker_path_utils import to_user_visible_path


# Logical keys stored under habit_project.json -> paths
PATH_KEY_RAW: str = "raw"
PATH_KEY_DICOM_SORTED: str = "dicom_sorted"
PATH_KEY_PREPROCESSED_ROOT: str = "preprocessed_root"
PATH_KEY_PREPROCESSED: str = "preprocessed"
PATH_KEY_HABITAT_OUT: str = "habitat_out"
PATH_KEY_FEATURES: str = "features"
PATH_KEY_ML: str = "ml"
PATH_KEY_COMPARE: str = "compare"


@dataclass
class StepPathBundle:
    """Resolved input/output paths for one workflow step (absolute runtime paths)."""

    step_id: str
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    extra_lists: Dict[str, List[str]] = field(default_factory=dict)


class ProjectContext:
    """
    Loaded project metadata with helpers for GUI path auto-fill.

    Args:
        root: Absolute project root directory.
        meta: Parsed ``habit_project.json`` document (already migrated).
    """

    def __init__(self, root: str, meta: Dict[str, Any]) -> None:
        self.root: str = normalize_project_root(root)
        self.meta: Dict[str, Any] = meta

    @classmethod
    def load(cls, root: str) -> Optional["ProjectContext"]:
        """
        Load and migrate project metadata.

        Args:
            root: Project root path.

        Returns:
            ProjectContext instance, or None when the project file is missing.
        """
        if not root or not str(root).strip():
            return None
        meta = load_project(str(root).strip())
        if not meta:
            return None
        meta = migrate_project_meta(meta)
        return cls(root=str(root).strip(), meta=meta)

    def abs_path(self, path_key: str) -> str:
        """Resolve a logical path key to an absolute filesystem path."""
        return resolve_project_path(self.root, path_key, self.meta)

    def display_path(self, path_key: str) -> str:
        """User-visible path string for a logical project key."""
        return self._display(self.abs_path(path_key))

    def _display(self, abs_path: str) -> str:
        """Convert absolute runtime path to GUI textbox value."""
        if not abs_path:
            return ""
        try:
            return to_user_visible_path(abs_path)
        except Exception:
            return user_visible_path(abs_path)

    def paths_for_step(self, step_id: str) -> StepPathBundle:
        """
        Return suggested paths for a workflow step.

        Args:
            step_id: One of dicom_sort, preprocess, habitat, extract, ml, compare.

        Returns:
            StepPathBundle with input/output absolute paths.
        """
        handlers = {
            "dicom_sort": self._paths_dicom_sort,
            "preprocess": self._paths_preprocess,
            "habitat": self._paths_habitat,
            "extract": self._paths_extract,
            "ml": self._paths_ml,
            "compare": self._paths_compare,
        }
        fn = handlers.get(step_id)
        if fn is None:
            return StepPathBundle(step_id=step_id)
        return fn()

    def _paths_dicom_sort(self) -> StepPathBundle:
        return StepPathBundle(
            step_id="dicom_sort",
            inputs={"data_dir": self.abs_path(PATH_KEY_RAW)},
            outputs={"out_dir": self.abs_path(PATH_KEY_DICOM_SORTED)},
        )

    def _paths_preprocess(self) -> StepPathBundle:
        # Preprocess out_dir is the parent; core writes out_dir/processed_images/
        return StepPathBundle(
            step_id="preprocess",
            inputs={"data_dir": self._preprocess_input_dir()},
            outputs={"out_dir": self.abs_path(PATH_KEY_PREPROCESSED_ROOT)},
        )

    def _preprocess_input_dir(self) -> str:
        """Prefer sorted DICOM folder when it exists, else raw data folder."""
        dicom_sorted = self.abs_path(PATH_KEY_DICOM_SORTED)
        if os.path.isdir(dicom_sorted) and os.listdir(dicom_sorted):
            return dicom_sorted
        return self.abs_path(PATH_KEY_RAW)

    def _paths_habitat(self) -> StepPathBundle:
        return StepPathBundle(
            step_id="habitat",
            inputs={"data_dir": self.abs_path(PATH_KEY_PREPROCESSED)},
            outputs={"out_dir": self.abs_path(PATH_KEY_HABITAT_OUT)},
        )

    def _paths_extract(self) -> StepPathBundle:
        return StepPathBundle(
            step_id="extract",
            inputs={
                "raw_img_folder": self.abs_path(PATH_KEY_PREPROCESSED),
                "habitats_map_folder": self.abs_path(PATH_KEY_HABITAT_OUT),
            },
            outputs={"out_dir": self.abs_path(PATH_KEY_FEATURES)},
        )

    def _paths_ml(self) -> StepPathBundle:
        feature_csv = self._discover_feature_csv()
        pipeline_path = self._discover_ml_pipeline()
        return StepPathBundle(
            step_id="ml",
            inputs={
                "csv_path": feature_csv or "",
                "pipeline_path": pipeline_path or "",
            },
            outputs={"output_dir": self.abs_path(PATH_KEY_ML)},
        )

    def _paths_compare(self) -> StepPathBundle:
        preds = self._discover_prediction_csvs(max_files=6)
        return StepPathBundle(
            step_id="compare",
            outputs={"output_dir": self.abs_path(PATH_KEY_COMPARE)},
            extra_lists={"prediction_csvs": preds},
        )

    def _discover_feature_csv(self) -> Optional[str]:
        """Find the newest feature CSV under the project features folder."""
        features_dir = self.abs_path(PATH_KEY_FEATURES)
        if not os.path.isdir(features_dir):
            return None
        patterns = [
            os.path.join(features_dir, "**", "*.csv"),
            os.path.join(features_dir, "*.csv"),
        ]
        candidates: List[str] = []
        for pattern in patterns:
            candidates.extend(glob.glob(pattern, recursive=True))
        if not candidates:
            return None
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    def _discover_ml_pipeline(self) -> Optional[str]:
        """Find a saved ML pipeline pickle under the project ML folder."""
        ml_dir = self.abs_path(PATH_KEY_ML)
        if not os.path.isdir(ml_dir):
            return None
        patterns = [
            os.path.join(ml_dir, "**", "*.pkl"),
            os.path.join(ml_dir, "**", "*pipeline*"),
        ]
        candidates: List[str] = []
        for pattern in patterns:
            for path in glob.glob(pattern, recursive=True):
                if os.path.isfile(path):
                    candidates.append(path)
        if not candidates:
            return None
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    def _discover_prediction_csvs(self, max_files: int = 6) -> List[str]:
        """Find prediction CSV files suitable for model comparison."""
        ml_dir = self.abs_path(PATH_KEY_ML)
        if not os.path.isdir(ml_dir):
            return []
        patterns = [
            os.path.join(ml_dir, "**", "*pred*.csv"),
            os.path.join(ml_dir, "**", "*prediction*.csv"),
            os.path.join(ml_dir, "**", "*.csv"),
        ]
        seen: set[str] = set()
        candidates: List[str] = []
        for pattern in patterns:
            for path in glob.glob(pattern, recursive=True):
                if not os.path.isfile(path):
                    continue
                norm = os.path.normpath(path)
                if norm in seen:
                    continue
                seen.add(norm)
                candidates.append(path)
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[:max_files]

    def gui_updates_for_step(self, step_id: str) -> Dict[str, str]:
        """
        Map logical field names to user-visible path strings for one step.

        Returns:
            Dict[str, str]: e.g. {"data_dir": "F:\\proj\\01_raw", ...}
        """
        bundle = self.paths_for_step(step_id)
        updates: Dict[str, str] = {}
        for key, path in {**bundle.inputs, **bundle.outputs}.items():
            if path:
                updates[key] = self._display(path)
        if step_id == "compare":
            preds = bundle.extra_lists.get("prediction_csvs", [])
            for idx, path in enumerate(preds):
                if path:
                    updates[f"path_{idx}"] = self._display(path)
        return updates

    def progress_header_html(self) -> str:
        """HTML snippet showing workflow completion for the workspace header."""
        from habit.gui.workflow_state import get_progress_summary

        summary = get_progress_summary(self.root)
        pct = summary.get("percent", 0)
        completed = summary.get("completed", 0)
        total = summary.get("total", 6)
        return (
            f"<div class='habit-progress-bar'>"
            f"Progress: {completed}/{total} steps ({pct}%)"
            f"</div>"
        )

    def workspace_header_html(self) -> str:
        """Full workspace header including project name, path, and progress."""
        name = self.meta.get("name", self.root)
        return (
            f"<div class='habit-workspace-header'>"
            f"<h2>{name}</h2>"
            f"<div class='habit-project-path'>{self._display(self.root)}</div>"
            f"{self.progress_header_html()}"
            f"</div>"
        )


def gr_updates_for_paths(field_values: Dict[str, str], ordered_fields: List[str]) -> Tuple[Any, ...]:
    """
    Build Gradio gr.update tuples for a fixed list of textbox components.

    Args:
        field_values: Mapping logical field name -> display path.
        ordered_fields: Component field names in output order.

    Returns:
        Tuple of gr.update objects for each field.
    """
    import gradio as gr

    updates: List[Any] = []
    for field_name in ordered_fields:
        val = field_values.get(field_name, "")
        if val:
            updates.append(gr.update(value=val))
        else:
            updates.append(gr.update())
    return tuple(updates)
