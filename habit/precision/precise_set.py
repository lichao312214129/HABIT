# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""The precise-feature artefact: which voxel features survived, and why.

``PreciseFeatureSet`` is to precision analysis what
:class:`~habit.contracts.habitat.HabitatModel` is to habitat definition: a
small, self-describing, serialisable object a study can publish so that
other groups restrict their habitat computation to the SAME features. It
carries the selected names, the selection criterion, and the full evidence
(the cohort-level ICC panel of every experiment), so the selection is
auditable without rerunning the analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

import pandas as pd

from habit.contracts.habitat import _provenance_from_dict, _provenance_to_dict
from habit.contracts.provenance import Provenance
from habit.exceptions import HABITAPIError

__all__ = ["PreciseFeatureSet"]

#: Serialisation schema version, bumped on incompatible format changes.
_FORMAT_VERSION = "1.0"


@dataclass(frozen=True)
class PreciseFeatureSet:
    """
    Voxel features that passed the precision screen, with the evidence.

    Attributes:
        feature_names: Selected feature names, in panel column order.
        lcl_threshold: Lower-confidence-limit cutoff every selected feature
            cleared in every experiment (unless expert-included).
        experiments: Experiment names, matching ``panels``.
        panels: Cohort-level ICC panel per experiment (index: feature;
            columns: ``value``, ``lcl``, ``ucl``, ``n_voxels``).
        provenance: How this selection was produced.
    """

    feature_names: Tuple[str, ...]
    lcl_threshold: float
    experiments: Tuple[str, ...]
    panels: Mapping[str, pd.DataFrame]
    provenance: Provenance

    def __post_init__(self) -> None:
        """Enforce the internal consistency of the artefact."""
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "experiments", tuple(self.experiments))
        object.__setattr__(self, "panels", dict(self.panels))
        if set(self.experiments) != set(self.panels):
            raise HABITAPIError(
                "PreciseFeatureSet: experiments and panels keys differ "
                f"({sorted(self.experiments)} vs {sorted(self.panels)})."
            )
        if not self.experiments:
            raise HABITAPIError(
                "PreciseFeatureSet: at least one experiment is required."
            )
        known = set(self.panels[self.experiments[0]].index)
        unknown = [f for f in self.feature_names if f not in known]
        if unknown:
            raise HABITAPIError(
                f"PreciseFeatureSet: selected features absent from the "
                f"panels: {unknown}."
            )

    def preprocessor(self) -> Any:
        """
        Return the feature whitelist restricting a habitat run to these features.

        Drop the returned component into a spec's
        ``voxel_feature_preprocessors`` chain (or use its ``.spec`` in a
        YAML-facing payload) and the habitat computation clusters exactly
        the precise features -- the Prior et al. 2024 workflow.

        Returns:
            A ``FeatureWhitelist`` over :attr:`feature_names`.
        """
        from habit.feature_preprocessing import FeatureWhitelist

        return FeatureWhitelist(list(self.feature_names))

    def to_frame(self) -> pd.DataFrame:
        """
        Return the long-format evidence table.

        Returns:
            One row per experiment per feature, with columns
            ``experiment``, ``feature``, ``value``, ``lcl``, ``ucl``,
            ``n_voxels`` and ``precise``.
        """
        frames = []
        for experiment in self.experiments:
            panel = self.panels[experiment]
            frame = panel.reset_index().rename(columns={"index": "feature"})
            if "feature" not in frame.columns:
                frame = frame.rename(columns={frame.columns[0]: "feature"})
            frame.insert(0, "experiment", experiment)
            frames.append(frame)
        evidence = pd.concat(frames, ignore_index=True)
        evidence["precise"] = evidence["feature"].isin(set(self.feature_names))
        return evidence

    def save(self, path: Union[str, Path]) -> Path:
        """
        Serialise to one self-describing JSON file.

        Args:
            path: Destination file; the parent directory must exist.

        Returns:
            The path written.
        """
        payload: Dict[str, Any] = {
            "format": "habit.PreciseFeatureSet",
            "format_version": _FORMAT_VERSION,
            "feature_names": list(self.feature_names),
            "lcl_threshold": self.lcl_threshold,
            "experiments": list(self.experiments),
            "panels": {
                experiment: self.panels[experiment]
                .reset_index()
                .to_dict(orient="records")
                for experiment in self.experiments
            },
            "provenance": _provenance_to_dict(self.provenance),
        }
        destination = Path(path)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PreciseFeatureSet":
        """
        Load a serialised precise feature set.

        Args:
            path: File written by :meth:`save`.

        Returns:
            The reconstructed artefact.

        Raises:
            HABITAPIError: If the file is not a PreciseFeatureSet serialisation.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "habit.PreciseFeatureSet":
            raise HABITAPIError(
                f"Not a PreciseFeatureSet file: {path} "
                f"(format={payload.get('format')!r})."
            )
        if payload.get("format_version") != _FORMAT_VERSION:
            raise HABITAPIError(
                f"PreciseFeatureSet format version "
                f"{payload.get('format_version')!r} is not supported by this "
                f"HABIT version (expected {_FORMAT_VERSION!r})."
            )
        panels = {}
        for experiment, records in payload["panels"].items():
            frame = pd.DataFrame.from_records(records)
            index_column = "feature" if "feature" in frame.columns else frame.columns[0]
            panels[experiment] = frame.set_index(index_column)
        return cls(
            feature_names=tuple(payload["feature_names"]),
            lcl_threshold=float(payload["lcl_threshold"]),
            experiments=tuple(payload["experiments"]),
            panels=panels,
            provenance=_provenance_from_dict(payload["provenance"]),
        )
