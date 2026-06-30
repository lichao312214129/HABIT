# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Voxel feature type helpers for the habitat analysis wizard."""

from __future__ import annotations

from typing import Dict, List

# Canonical voxel feature constructors exposed in the GUI dropdown.
VOXEL_FEATURE_TYPES: List[str] = [
    "raw",
    "kinetic",
    "local_entropy",
    "voxel_radiomics",
]

VOXEL_FEATURE_TYPE_LABELS: Dict[str, str] = {
    "raw": "Raw intensities",
    "kinetic": "DCE kinetic curve",
    "local_entropy": "Local entropy",
    "voxel_radiomics": "Voxel radiomics (texture)",
}


def voxel_radiomics_panel_visible(feature_type: str) -> bool:
    """
    Return True when the voxel radiomics parameter row should be shown.

    Args:
        feature_type: Selected voxel feature type key from the wizard.

    Returns:
        bool: True for ``voxel_radiomics``; False otherwise.
    """
    return str(feature_type) == "voxel_radiomics"


__all__ = [
    "VOXEL_FEATURE_TYPES",
    "VOXEL_FEATURE_TYPE_LABELS",
    "voxel_radiomics_panel_visible",
]
