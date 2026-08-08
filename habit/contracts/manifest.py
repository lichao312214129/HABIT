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
"""What a completed run records: the run manifest.

This is an L2 data structure only. The L4 recipe layer assembles it from
executed runs, and the L4 report layer renders it; nothing here touches the
filesystem except ``RunManifest.to_json`` when explicitly called by the user.

``StudyResult`` used to live here too. It moved to
:mod:`habit.recipes.result`, where it belongs: it is the recipe layer's
return type, no L0-L3 component produces or consumes one, and its ``save``
method needs an output directory -- a concept L2 is forbidden to know.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import pandas as pd

from habit.exceptions import HABITAPIError
from habit.contracts.habitat import _provenance_to_dict
from habit.contracts.provenance import Provenance

__all__ = ["RunManifest"]

#: Reporting standards supported by :meth:`RunManifest.checklist`.
_CHECKLIST_STANDARDS = ("IBSI", "CLEAR", "METRICS", "TRIPOD+AI")

#: Methods styles supported by ``describe_methods`` here and on HabitatSpec.
_METHODS_STYLES = ("radiology", "nature")

#: Human phrasing for the HabitatSpec component domains, in pipeline order.
#: (Deliberately duplicated in ``habit.spec.specs``: the contracts layer must
#: never import the spec layer, and the fragment is small.)
_COMPONENT_PHRASES: Tuple[Tuple[str, str], ...] = (
    ("voxel_feature_extractor", "voxel feature extraction"),
    ("supervoxelizer", "supervoxelization"),
    ("postprocess_supervoxel", "supervoxel connected-component postprocessing"),
    ("habitat_model_fitter", "habitat model fitting"),
    ("habitat_assigner", "habitat assignment"),
    ("postprocess_habitat", "habitat connected-component postprocessing"),
)


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


def _params_text(params: Any) -> str:
    """Render one component's parameter mapping as prose."""
    if not params:
        return "default parameters"
    if isinstance(params, Mapping):
        return ", ".join(f"{key}={value!r}" for key, value in params.items())
    return str(params)


def _component_phrases(payload: Mapping[str, Any]) -> Tuple[str, ...]:
    """
    Render a HabitatSpec-shaped payload as ordered prose phrases.

    Only keys actually present are rendered: the report states what the
    analysis contains, never what a template might have contained.

    Args:
        payload: Spec payload as produced by ``HabitatSpec.to_dict``.

    Returns:
        One phrase per present component, in pipeline order.
    """
    phrases: list[str] = []
    for key, phrase in _COMPONENT_PHRASES:
        if key not in payload:
            continue
        entry = payload[key]
        if entry is None:
            if key == "supervoxelizer":
                phrases.append("direct voxel clustering (no supervoxelization)")
            continue
        if isinstance(entry, Mapping) and "name" in entry:
            phrases.append(
                f"{phrase} with {entry['name']} ({_params_text(entry.get('params'))})"
            )
        else:
            phrases.append(f"{phrase} with {entry}")
    features = payload.get("habitat_features") or []
    if features:
        families = ", ".join(
            f"{entry['name']} ({_params_text(entry.get('params'))})"
            if isinstance(entry, Mapping) and "name" in entry
            else str(entry)
            for entry in features
        )
        phrases.append(f"habitat feature families: {families}")
    for chain_key, chain_phrase in (
        ("voxel_feature_preprocessors", "per-subject voxel feature preprocessing"),
        (
            "supervoxel_feature_preprocessors",
            "per-subject supervoxel feature preprocessing",
        ),
        ("cohort_feature_preprocessors", "cohort-level feature preprocessing"),
    ):
        chain = payload.get(chain_key) or []
        if chain:
            steps = ", ".join(
                entry["name"] if isinstance(entry, Mapping) and "name" in entry else str(entry)
                for entry in chain
            )
            phrases.append(f"{chain_phrase}: {steps}")
    return tuple(phrases)


def _specification_sentence(payload: Mapping[str, Any]) -> Optional[str]:
    """
    Render the analysis specification as one methods sentence.

    Args:
        payload: The manifest's recorded spec payload.

    Returns:
        The sentence, or ``None`` when no specification was recorded.
    """
    if not payload:
        return None
    phrases = _component_phrases(payload)
    if not phrases:
        return None
    name = payload.get("name")
    lead = "The analysis specification"
    if isinstance(name, str) and name:
        lead += f" {name!r}"
    return f"{lead} comprised {'; '.join(phrases)}."


#: Guidance text for checklist items no execution record can answer. Every
#: item is an honest "needs_human_answer" rather than a fabricated tick.
_HUMAN_CHECKLIST_GUIDANCE: Mapping[str, str] = {
    "study_design": (
        "Study design (prospective/retrospective, multi-centre) must be "
        "described by the authors"
    ),
    "clinical_cohort_description": (
        "Cohort recruitment, eligibility and ethics cannot be derived from "
        "execution records"
    ),
    "image_acquisition": (
        "Scanner, sequence and acquisition parameters must be described by "
        "the authors"
    ),
    "annotation_protocol": (
        "ROI delineation protocol must be described by the authors"
    ),
    "image_preprocessing": (
        "Pre-analysis image processing (resampling, denoising, "
        "normalisation) must be described by the authors"
    ),
    "outcome_definition": (
        "The clinical outcome and its assessment must be described by the "
        "authors"
    ),
    "predictor_definition": (
        "Predictor measurement and blinding must be described by the authors"
    ),
    "missing_data": "Missing-data handling must be described by the authors",
    "validation_design": (
        "Internal/external validation design must be described by the authors"
    ),
    "calibration_assessment": (
        "Calibration assessment of the outcome model must be reported by "
        "the authors"
    ),
    "fairness": "Fairness/subgroup analyses must be reported by the authors",
    "benchmark_validation": (
        "Benchmarking against reference implementations must be reported by "
        "the authors"
    ),
    "code_availability": "A code availability statement is an editorial decision",
    "data_availability": (
        "A data availability statement is an editorial/legal decision"
    ),
    "funding": "A funding statement is an editorial decision",
}

#: Ordered checklist items per reporting standard. Items present in
#: ``RunManifest._checklist_facts`` are evidenced; the rest need humans.
_CHECKLIST_LAYOUTS: Mapping[str, Tuple[str, ...]] = {
    # IBSI (Image Biomarker Standardisation Initiative) reporting items.
    "IBSI": (
        "image_acquisition",
        "annotation_protocol",
        "image_preprocessing",
        "analysis_specification",
        "feature_families",
        "software_version",
        "dependency_versions",
        "random_seeds",
        "benchmark_validation",
    ),
    # CLEAR (CheckList for EvaluAtion of Radiomics research).
    "CLEAR": (
        "clinical_cohort_description",
        "image_acquisition",
        "annotation_protocol",
        "image_preprocessing",
        "analysis_specification",
        "feature_families",
        "cohort_size",
        "excluded_subjects",
        "software_version",
        "dependency_versions",
        "random_seeds",
        "code_availability",
        "data_availability",
    ),
    # METRICS (MEthodological RadiomICs Score) topics.
    "METRICS": (
        "study_design",
        "clinical_cohort_description",
        "image_acquisition",
        "annotation_protocol",
        "image_preprocessing",
        "analysis_specification",
        "feature_families",
        "cohort_size",
        "validation_design",
        "software_version",
        "dependency_versions",
        "random_seeds",
        "code_availability",
        "data_availability",
    ),
    # TRIPOD+AI prediction-model reporting items.
    "TRIPOD+AI": (
        "study_design",
        "clinical_cohort_description",
        "outcome_definition",
        "predictor_definition",
        "missing_data",
        "analysis_specification",
        "cohort_size",
        "excluded_subjects",
        "validation_design",
        "calibration_assessment",
        "fairness",
        "software_version",
        "random_seeds",
        "data_availability",
        "funding",
    ),
}


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
        provenance DAG, plus the recorded specification, software versions,
        seeds, and excluded subjects. Generating plausible but unexecuted
        methods text would make the whole reporting feature untrustworthy.

        Args:
            style: Target venue convention. ``"radiology"`` opens with the
                software sentence; ``"nature"`` closes with it. Ordering and
                wording only -- the stated facts are identical.

        Returns:
            English prose that states only steps that actually executed.

        Raises:
            HABITAPIError: On an unknown style.
        """
        if style not in _METHODS_STYLES:
            raise HABITAPIError(
                f"Unknown methods style {style!r}; expected one of "
                f"{_METHODS_STYLES}."
            )
        versions = self.software_versions()
        seeds = self.random_seeds()
        body: list[str] = []
        specification = _specification_sentence(self.spec_payload)
        if specification is not None:
            body.append(specification)
        chain = _collect_provenance_chain(self.provenance)
        steps = [record.produced_by for record in chain if record.produced_by]
        body.append(
            "The executed pipeline steps, in provenance order, were: "
            + ("; ".join(steps) if steps else "none recorded")
            + "."
        )
        if seeds:
            seed_text = ", ".join(f"{name}={seed}" for name, seed in seeds.items())
            body.append(f"Random seeds were fixed as follows: {seed_text}.")
        failed = sorted(
            subject
            for subject, outcome in self.subject_outcomes.items()
            if outcome != "success"
        )
        if failed:
            body.append(
                f"{len(failed)} subject(s) failed processing and were "
                f"excluded: {', '.join(failed)}."
            )
        if style == "nature":
            closing = (
                "All analyses were performed with HABIT "
                f"(version {versions.get('habit', 'unknown')})."
            )
            return " ".join([*body, closing])
        opening = (
            "Habitat imaging analysis was performed with HABIT "
            f"(version {versions.get('habit', 'unknown')})."
        )
        return " ".join([opening, *body])

    def _checklist_facts(self) -> Mapping[str, Tuple[str, str]]:
        """
        Compute the checklist items HABIT can evidence from execution records.

        Returns:
            Item key -> ``(status, evidence)`` for every machine-evidencable
            item. Anything absent from this mapping is a human question, and
            :data:`_HUMAN_CHECKLIST_GUIDANCE` explains why.
        """
        versions = self.software_versions()
        seeds = self.random_seeds()
        failed = {
            subject: outcome
            for subject, outcome in self.subject_outcomes.items()
            if outcome != "success"
        }
        facts: Dict[str, Tuple[str, str]] = {
            "software_version": (
                "evidenced",
                f"HABIT {versions.get('habit', 'unknown')}",
            ),
            "dependency_versions": (
                "evidenced",
                json.dumps(versions, sort_keys=True),
            ),
            "random_seeds": (
                "evidenced" if seeds else "needs_human_answer",
                json.dumps(seeds, sort_keys=True)
                if seeds
                else "No stochastic components recorded",
            ),
            "excluded_subjects": (
                "evidenced",
                json.dumps(failed, sort_keys=True),
            ),
            "analysis_specification": (
                "evidenced" if self.spec_payload else "needs_human_answer",
                json.dumps(self.spec_payload, sort_keys=True, default=str)
                if self.spec_payload
                else "No analysis specification was recorded for this run",
            ),
            "cohort_size": (
                "evidenced" if self.subject_outcomes else "needs_human_answer",
                f"{len(self.subject_outcomes)} subjects processed, "
                f"{len(failed)} excluded"
                if self.subject_outcomes
                else "Subject counts were not recorded for this run",
            ),
        }
        features = self.spec_payload.get("habitat_features") or []
        if features:
            families = [
                entry["name"]
                for entry in features
                if isinstance(entry, Mapping) and "name" in entry
            ]
            facts["feature_families"] = (
                "evidenced",
                ", ".join(families) if families else json.dumps(features, default=str),
            )
        else:
            facts["feature_families"] = (
                "needs_human_answer",
                "Habitat feature families were not recorded for this run",
            )
        return facts

    def checklist(self, standard: str) -> pd.DataFrame:
        """
        Return an item-by-item compliance table for a reporting standard.

        Args:
            standard: One of ``"IBSI"``, ``"CLEAR"``, ``"METRICS"``,
                ``"TRIPOD+AI"``.

        Returns:
            One row per checklist item of that standard with the value HABIT
            can evidence and, where it cannot, an explicit statement that the
            item needs a human answer. Silently marking unverifiable items as
            satisfied would make the whole feature untrustworthy.

        Raises:
            HABITAPIError: On an unknown standard.
        """
        if standard not in _CHECKLIST_STANDARDS:
            raise HABITAPIError(
                f"Unknown reporting standard {standard!r}; expected one of "
                f"{_CHECKLIST_STANDARDS}."
            )
        facts = self._checklist_facts()
        rows = []
        for item in _CHECKLIST_LAYOUTS[standard]:
            if item in facts:
                rows.append((item, *facts[item]))
            else:
                rows.append(
                    (item, "needs_human_answer", _HUMAN_CHECKLIST_GUIDANCE[item])
                )
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
