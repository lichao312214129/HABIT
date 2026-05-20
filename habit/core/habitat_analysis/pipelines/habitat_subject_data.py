"""
Explicit per-subject payload contract for habitat pipeline steps.

The individual-level pipeline used to pass anonymous ``dict`` objects with
string keys such as ``features`` / ``raw`` / ``mask_info``.  That made the
real Interface live across several step implementations.  ``HabitatSubjectData``
centralises that Interface: each step receives and returns one explicit object,
and helper methods validate required stage data close to where it is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class HabitatSubjectData:
    """
    Per-subject data passed between individual-level habitat pipeline steps.

    Fields are optional because the object is progressively populated by the
    pipeline.  Step code should call the ``require_*`` methods before using a
    field; this gives a clear error when a recipe is mis-ordered.
    """

    features: Optional[pd.DataFrame] = None
    raw: Optional[pd.DataFrame] = None
    mask_info: Optional[Dict[str, Any]] = None
    supervoxel_labels: Optional[np.ndarray] = None
    mean_voxel_features: Optional[pd.DataFrame] = None
    supervoxel_features: Optional[pd.DataFrame] = None
    supervoxel_df: Optional[pd.DataFrame] = None

    @classmethod
    def empty(cls) -> "HabitatSubjectData":
        """Return the empty payload used at the start of a subject pipeline."""
        return cls()

    def require_features(self, step_name: str) -> pd.DataFrame:
        return self._require("features", self.features, step_name)

    def require_raw(self, step_name: str) -> pd.DataFrame:
        return self._require("raw", self.raw, step_name)

    def require_mask_info(self, step_name: str) -> Dict[str, Any]:
        return self._require("mask_info", self.mask_info, step_name)

    def require_supervoxel_labels(self, step_name: str) -> np.ndarray:
        return self._require("supervoxel_labels", self.supervoxel_labels, step_name)

    def require_mean_voxel_features(self, step_name: str) -> pd.DataFrame:
        return self._require("mean_voxel_features", self.mean_voxel_features, step_name)

    def require_supervoxel_features(self, step_name: str) -> pd.DataFrame:
        return self._require("supervoxel_features", self.supervoxel_features, step_name)

    def require_supervoxel_df(self, step_name: str) -> pd.DataFrame:
        return self._require("supervoxel_df", self.supervoxel_df, step_name)

    @staticmethod
    def _require(field_name: str, value: Any, step_name: str) -> Any:
        if value is None:
            raise ValueError(
                f"{step_name} requires '{field_name}' in HabitatSubjectData. "
                "Check the habitat pipeline recipe order."
            )
        return value
