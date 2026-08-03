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
"""What a completed study hands back: run manifest and study result.

These are L2 data structures only. The L4 recipe layer assembles them from
executed runs, and the L4 report layer renders them; nothing here touches
the filesystem except ``RunManifest.to_json`` and ``StudyResult.save`` when
explicitly called by the user.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import pandas as pd

from habit.api.exceptions import HABITAPIError
from habit.contracts.habitat import HabitatMap, HabitatModel, _provenance_to_dict
from habit.contracts.provenance import Provenance
from habit.contracts.table import FeatureTable

__all__ = ["RunManifest", "StudyResult"]

#: Reporting standards supported by :meth:`RunManifest.checklist`.
_CHECKLIST_STANDARDS = ("IBSI", "CLEAR", "METRICS", "TRIPOD+AI")


def _collect_provenance_chain(root: Provenance) -> Tuple[Provenance, ...]:
    """Flatten a provenance DAG into a breadth-first tuple without repeats."""
    seen: list[int] = []
    ordered: list[Provenance] = []
    queue = [root]
    while queue:
        current = queue.pop(0)
        if id(current) in seen:
            continue
        seen.append(id(current))
        ordered.append(current)
        queue.extend(current.inputs)
    return tuple(ordered)


@dataclass(frozen=True)
class RunManifest:
    """
    Everything needed to describe and audit one completed analysis.

    Assembled from the :class:`Provenance` records that travelled with the
    data, so it reports what actually ran rather than what was requested.
    That distinction is the whole point: a methods paragraph derived from a
    configuration file would describe intent, while this one describes fact,
    including subjects that failed and were excluded.

    Attributes:
        spec_payload: Serialised specification of the analysis that ran.
        provenance: Root provenance of the primary result.
        subject_outcomes: Per-subject success or failure, keyed by subject
            id. Values are ``"success"`` or an error summary.
        started_at: ISO-8601 start timestamp.
        finished_at: ISO-8601 completion timestamp.
    """

    spec_payload: Mapping[str, Any]
    provenance: Provenance
    subject_outcomes: Mapping[str, str] = field(default_factory=dict)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def software_versions(self) -> Mapping[str, str]:
        """Return HABIT and dependency versions captured at execution time."""
        return dict(self.provenance.software)

    def random_seeds(self) -> Mapping[str, int]:
        """Return the seed used by each stochastic component in the DAG."""
        seeds: Dict[str, int] = {}
        for record in _collect_provenance_chain(self.provenance):
            if record.random_seed is not None:
                seeds[record.produced_by] = record.random_seed
        return seeds

    def describe_methods(self, style: str = "radiology") -> str:
        """
        Render the executed analysis as a manuscript methods paragraph.

        The text states only steps that actually executed, derived from the
        provenance DAG, plus software versions, seeds, and excluded subjects.
        Generating plausible but unexecuted methods text would make the whole
        reporting feature untrustworthy.

        Args:
            style: Target venue convention, e.g. ``"radiology"`` or
                ``"nature"``. Only affects wording and ordering, never
                content.

        Returns:
            English prose that states only steps that actually executed.
        """
        chain = _collect_provenance_chain(self.provenance)
        steps = [record.produced_by for record in chain if record.produced_by]
        versions = self.software_versions()
        seeds = self.random_seeds()
        sentences = [
            "Habitat imaging analysis was performed with HABIT "
            f"(version {versions.get('habit', 'unknown')}).",
            "The executed pipeline steps, in provenance order, were: "
            + ("; ".join(steps) if steps else "none recorded")
            + ".",
        ]
        if seeds:
            seed_text = ", ".join(f"{name}={seed}" for name, seed in seeds.items())
            sentences.append(f"Random seeds were fixed as follows: {seed_text}.")
        failed = sorted(
            subject
            for subject, outcome in self.subject_outcomes.items()
            if outcome != "success"
        )
        if failed:
            sentences.append(
                f"{len(failed)} subject(s) failed processing and were "
                f"excluded: {', '.join(failed)}."
            )
        if style not in ("radiology", "nature"):
            raise HABITAPIError(
                f"Unknown methods style {style!r}; expected 'radiology' or "
                "'nature'."
            )
        return " ".join(sentences)

    def checklist(self, standard: str) -> pd.DataFrame:
        """
        Return an item-by-item compliance table for a reporting standard.

        Args:
            standard: One of ``"IBSI"``, ``"CLEAR"``, ``"METRICS"``,
                ``"TRIPOD+AI"``.

        Returns:
            One row per checklist item with the value HABIT can evidence and,
            where it cannot, an explicit statement that the item needs a
            human answer. Silently marking unverifiable items as satisfied
            would make the whole feature untrustworthy.
        """
        if standard not in _CHECKLIST_STANDARDS:
            raise HABITAPIError(
                f"Unknown reporting standard {standard!r}; expected one of "
                f"{_CHECKLIST_STANDARDS}."
            )
        versions = self.software_versions()
        rows = [
            (
                "software_version",
                "evidenced",
                f"HABIT {versions.get('habit', 'unknown')}",
            ),
            (
                "dependency_versions",
                "evidenced",
                json.dumps(versions, sort_keys=True),
            ),
            (
                "random_seeds",
                "evidenced" if self.random_seeds() else "needs_human_answer",
                json.dumps(self.random_seeds(), sort_keys=True)
                if self.random_seeds()
                else "No stochastic components recorded",
            ),
            (
                "excluded_subjects",
                "evidenced",
                json.dumps(
                    {
                        subject: outcome
                        for subject, outcome in self.subject_outcomes.items()
                        if outcome != "success"
                    },
                    sort_keys=True,
                ),
            ),
            (
                "clinical_cohort_description",
                "needs_human_answer",
                "Cohort recruitment, eligibility and ethics cannot be derived "
                "from execution records",
            ),
            (
                "annotation_protocol",
                "needs_human_answer",
                "ROI delineation protocol must be described by the authors",
            ),
        ]
        return pd.DataFrame(rows, columns=["item", "status", "evidence"])

    def to_json(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Serialise the manifest, optionally writing it to disk.

        Args:
            path: Destination file. When ``None`` the JSON text is only
                returned.

        Returns:
            The JSON text.
        """
        payload = {
            "spec_payload": json.loads(json.dumps(self.spec_payload, default=str)),
            "provenance": _provenance_to_dict(self.provenance),
            "subject_outcomes": dict(self.subject_outcomes),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
        return text


@dataclass(frozen=True, eq=False)
class StudyResult:
    """
    What a fitted study hands back, entirely in memory.

    Nothing here has touched the filesystem. Writing is a separate, explicit
    act via :meth:`save`, which is what allows the identical code to run
    inside someone else's service where there is no output directory at all.

    Attributes:
        habitat_model: The population-level habitat definition. Named in
            full rather than ``model`` because ``model`` already means a
            trained classifier elsewhere in HABIT.
        pipeline: The subject-level procedure that applies that definition,
            so that model and procedure can be shipped together for external
            validation.
        features: Habitat-level features for the fitted cohort.
        habitat_maps: Per-subject habitat label images, in cohort order.
        manifest: Provenance and reporting for this run.
    """

    habitat_model: HabitatModel
    pipeline: Any
    features: FeatureTable
    habitat_maps: Tuple[HabitatMap, ...]
    manifest: RunManifest

    def save(self, out_dir: Union[str, Path]) -> Path:
        """
        Write every artefact of this study to a directory.

        Args:
            out_dir: Destination directory, created when missing.

        Returns:
            The directory written to.
        """
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.habitat_model.save(destination / "habitat_model.habitatmodel")
        self.features.frame.to_csv(destination / "habitat_features.csv", index=False)
        self.manifest.to_json(destination / "run_manifest.json")
        return destination
