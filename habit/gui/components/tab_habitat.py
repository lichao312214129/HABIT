# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Habitat clustering analysis tab — radiologist-oriented wizard entry point."""

from typing import Any, Optional

from habit.gui.components.habitat_wizard import render_habitat_tab as render_habitat_wizard
from habit.gui.path_picker import PathPickerRegistry

__all__ = ["render_habitat_tab"]


def render_habitat_tab(
    demo: Optional[Any] = None,
    path_picker: PathPickerRegistry | None = None,
    project_root_state: Optional[Any] = None,
) -> None:
    """Render habitat segmentation wizard (delegates to habitat_wizard module)."""
    render_habitat_wizard(
        demo=demo,
        path_picker=path_picker,
        project_root_state=project_root_state,
    )
