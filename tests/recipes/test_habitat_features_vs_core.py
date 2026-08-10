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
"""
Feature-by-feature comparison of the v1 habitat families against v0.1.

The seven ``habit/domain/habitat_features/`` extractors were rewritten onto
the v1 contracts (arrays and geometry in memory instead of file paths). Nothing
had ever checked that the rewrite preserved the numbers, and a habitat feature
that silently shifts is the worst kind of regression: every model trained on it
still fits, still validates, and is no longer the published one.

The v0.1 side of the comparison is the frozen ``habitat_features`` baseline --
the CSVs ``habit extract`` wrote, value by value. The v1 side runs the domain
extractors over the habitat maps the ``two_step`` recipe produces, which
``test_recipes_golden_parity`` separately proves are voxel-identical to the
maps those CSVs were computed from. Any difference found here is therefore in
the feature code, not upstream.

Failures print a per-feature table (feature, v0 value, v1 value, relative
difference) rather than the first mismatch, because deciding what a divergence
means needs its shape: one feature off by 1e-9 and every feature off by 30%
are different findings.

Run with::

    pytest tests/recipes -m slow
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pytest

from tests.recipes.conftest import (
    demo_data_available,
    load_baseline,
    load_demo_cohort,
    spec_and_data_root,
)

#: Config whose habitat maps the v0.1 feature CSVs were extracted from.
TRAIN_CONFIG = "config/habitat/config_habitat_two_step.yaml"

#: Golden case holding the v0.1 feature CSVs.
FEATURES_CASE = "habitat_features"

#: Tolerance for the comparison, matching the baseline's recorded tolerance.
RTOL = 1e-6
ATOL = 1e-9


class FamilyCase(NamedTuple):
    """One v1 feature family paired with its v0.1 CSV."""

    #: Registry name of the v1 extractor.
    name: str
    #: Constructor parameters for the v1 extractor.
    params: Dict[str, Any]
    #: v0.1 artefacts to compare against, as ``(csv name, column mapper)``.
    #: The mapper turns a v0.1 column name into the v1 column expected to
    #: hold the same number, or ``None`` when v1 does not emit it.
    sources: Tuple[Tuple[str, Callable[[str], Optional[str]]], ...]


def _identity(column: str) -> Optional[str]:
    """v1 keeps the v0.1 column name unchanged."""
    return column


def _ith_column(column: str) -> Optional[str]:
    """Map the v0.1 ITH summary columns onto their prefixed v1 names."""
    return {"num_habitats": "ith_num_habitats", "total_area": "ith_total_area"}.get(
        column, column
    )


def _each_habitat_column(habitat_id: int) -> Callable[[str], Optional[str]]:
    """Return a mapper prefixing v0.1 per-habitat radiomics with the habitat id."""

    def mapper(column: str) -> Optional[str]:
        return f"habitat_{habitat_id}_{column}"

    return mapper


def _volume_column(column: str) -> Optional[str]:
    """
    Map the v0.1 basic-feature volume ratios onto the v1 volume family.

    The v1 ``volume`` family has no single v0.1 counterpart: v0.1 reported
    the same quantity as ``{id}_volume_ratio`` inside the basic features and
    never exported the voxel counts, so only the fractions are comparable.
    """
    if column.endswith("_volume_ratio"):
        return f"habitat_{column.split('_')[0]}_volume_fraction"
    return None


def _feature_cases() -> List[FamilyCase]:
    """Build the family cases, resolving the PyRadiomics presets v0.1 used."""
    from habit.utils.radiomics_preset_utils import resolve_params_file

    roi_preset = resolve_params_file(None, preset="roi")
    habitat_preset = resolve_params_file(None, preset="habitat")
    modalities = ["delay2", "delay3", "delay5"]
    return [
        FamilyCase("msi", {}, (("msi_features.csv", _identity),)),
        FamilyCase("ith_score", {}, (("ith_scores.csv", _ith_column),)),
        FamilyCase("non_radiomics", {}, (("habitat_basic_features.csv", _identity),)),
        FamilyCase("volume", {}, (("habitat_basic_features.csv", _volume_column),)),
        FamilyCase(
            "traditional",
            {"params_file": roi_preset, "modalities": modalities},
            (("raw_image_radiomics.csv", _identity),),
        ),
        FamilyCase(
            "whole_habitat",
            {"params_file": habitat_preset},
            (("whole_habitat_radiomics.csv", _identity),),
        ),
        FamilyCase(
            "each_habitat",
            {"params_file": roi_preset, "modalities": modalities},
            (
                ("habitat_count.csv", _identity),
                ("habitat_1_radiomics.csv", _each_habitat_column(1)),
                ("habitat_2_radiomics.csv", _each_habitat_column(2)),
                ("habitat_3_radiomics.csv", _each_habitat_column(3)),
                ("habitat_4_radiomics.csv", _each_habitat_column(4)),
            ),
        ),
    ]


#: Cohort, habitat maps and reference CSVs, shared by every family (fitting
#: the study once per family would multiply a 30 s run by seven).
_FIXTURE: Dict[str, Any] = {}


def _study() -> Dict[str, Any]:
    """Fit the two-step study once and index its habitat maps by subject."""
    if not _FIXTURE:
        from habit.recipes.study import Study

        spec, root = spec_and_data_root(TRAIN_CONFIG)
        cohort = load_demo_cohort(spec, root)
        result = Study(spec=spec, design="two_step").fit_predict(cohort)
        _FIXTURE["subjects"] = {s.subject_id: s for s in cohort}
        _FIXTURE["maps"] = {m.subject_id: m for m in result.habitat_maps}
    return _FIXTURE


def _v0_table(csv_name: str) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """
    Read one v0.1 CSV out of the frozen baseline.

    Args:
        csv_name: Artefact name inside the ``habitat_features`` baseline.

    Returns:
        Tuple of the subject ids in file order and a
        ``{subject: {column: value}}`` mapping.
    """
    baseline = load_baseline(FEATURES_CASE)
    record = baseline["fingerprints"].get(csv_name)
    if record is None:
        pytest.skip(f"{csv_name} is absent from the {FEATURES_CASE} baseline")
    # The subject id is the CSV's unnamed index column; the fingerprint keeps
    # non-numeric columns verbatim, so it survives as a categorical entry.
    subjects = next(iter(record["categorical"].values()))
    values: Dict[str, Dict[str, float]] = {subject: {} for subject in subjects}
    for column, column_values in record["values"].items():
        for subject, value in zip(subjects, column_values):
            values[subject][column] = value
    return list(subjects), values


def _relative_difference(v0: Optional[float], v1: Optional[float]) -> float:
    """Return the relative difference used to rank divergences."""
    if v0 is None or v1 is None:
        return float("inf")
    if np.isnan(v0) and np.isnan(v1):
        return 0.0
    denominator = max(abs(v0), abs(v1))
    return abs(v0 - v1) / denominator if denominator else 0.0


def _agrees(v0: Optional[float], v1: Optional[float]) -> bool:
    """Return whether two values agree within the baseline's tolerance."""
    if v0 is None and v1 is None:
        return True
    if v0 is None or v1 is None:
        return False
    if np.isnan(v0) and np.isnan(v1):
        return True
    return bool(np.isclose(v0, v1, rtol=RTOL, atol=ATOL, equal_nan=True))


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("case", _feature_cases(), ids=lambda case: case.name)
def test_v1_family_matches_core_values(case: FamilyCase) -> None:
    """
    A v1 habitat feature family reproduces the v0.1 numbers.

    Args:
        case: Feature family under test.
    """
    if not demo_data_available():
        pytest.skip("demo_data/ is not present; the feature comparison needs imaging data")

    from habit.domain.habitat_features import HabitatFeatureExtractorRegistry

    fixture = _study()
    extractor = HabitatFeatureExtractorRegistry.create(case.name, **case.params)

    rows: Dict[str, Dict[str, float]] = {}
    for subject_id, subject in fixture["subjects"].items():
        table = extractor(subject, fixture["maps"][subject_id])
        frame = table.frame
        rows[subject_id] = {
            column: float(frame.iloc[0][column]) for column in table.feature_columns
        }

    divergences: List[Tuple[str, str, Any, Any, float]] = []
    compared = 0
    for csv_name, mapper in case.sources:
        subjects, v0_values = _v0_table(csv_name)
        for subject_id in subjects:
            for column, v0_value in v0_values[subject_id].items():
                v1_column = mapper(column)
                if v1_column is None:
                    continue
                compared += 1
                v1_value = rows.get(subject_id, {}).get(v1_column)
                if not _agrees(v0_value, v1_value):
                    divergences.append(
                        (
                            subject_id,
                            f"{column} -> {v1_column}",
                            v0_value,
                            v1_value,
                            _relative_difference(v0_value, v1_value),
                        )
                    )

    assert compared, f"{case.name}: nothing was compared; the column mapping is wrong"
    if divergences:
        divergences.sort(key=lambda row: row[-1], reverse=True)
        header = f"{'subject':10} {'feature':60} {'v0':>18} {'v1':>18} {'rel.diff':>12}"
        lines = [header, "-" * len(header)]
        for subject_id, feature, v0_value, v1_value, delta in divergences[:40]:
            lines.append(
                f"{subject_id:10} {feature:60} {str(v0_value):>18} "
                f"{str(v1_value):>18} {delta:12.3e}"
            )
        pytest.fail(
            f"{case.name}: {len(divergences)} of {compared} values diverge from v0.1\n"
            + "\n".join(lines)
        )
