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

"""Phase-4a hard gate: v0.1 templates vs the v1 domain registries.

``tests/spec/test_v0_capability_coverage.py`` already guards the habitat
templates; this gate widens the net to EVERY workflow template shipped under
``config/`` and to the domains the habitat gate does not look at
(``habitat_features``, ML, image preprocessing, feature extraction, ICC,
test-retest, model comparison).

For each template the test runs ``LegacyConfigAdapter`` and asserts that
every component name the translation produces is present in the matching
domain registry's ``available()`` list. Parsing and lookup only -- no
pipeline runs -- so the whole file finishes in seconds.

Failures are expected while v1 domains are incomplete: the precise gap list
(template path, component domain, missing name) IS the phase-4 progress
metric. Gaps that are understood are pinned in ``_KNOWN_GAPS`` so the gate
stays green while making the debt visible; an unknown gap, or a known gap
that silently stops being exercised by any template, fails loudly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import pytest
import yaml

from habit.spec.legacy import LegacyConfigAdapter

_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"

#: One entry per (config path, domain, component name) the translation
#: produces.
Gap = Tuple[Path, str, str]

#: v0.1 capabilities with no v1 component/domain yet. Keys are
#: ``(domain, name)``; values state WHY there is no v1 home. Entries are
#: asserted to STILL be missing, so the list can only shrink: closing a gap
#: without deleting its entry fails ``test_known_gaps_are_still_gaps``.
_KNOWN_GAPS: Dict[Tuple[str, str], str] = {
    # The ICC workflow's agreement statistics have no v1 domain at all (the
    # v1 ``metric`` domain holds classification evaluation metrics; the v1
    # ``feature_selector`` domain's ``icc`` is a selector, not an agreement
    # statistic). A dedicated agreement-metric domain is a later-phase
    # decision.
    ("icc_metric", "icc2"): "no v1 agreement-metric domain yet",
    ("icc_metric", "icc3"): "no v1 agreement-metric domain yet",
    ("icc_metric", "cohen"): "no v1 agreement-metric domain yet",
    ("icc_metric", "fleiss"): "no v1 agreement-metric domain yet",
    ("icc_metric", "krippendorff"): "no v1 agreement-metric domain yet",
    # Test-retest habitat matching correlates unit features between two
    # tables; there is no v1 similarity domain (habit.kernels carries no
    # correlation routine either).
    ("retest_similarity", "pearson"): "no v1 similarity-metric domain yet",
    # Class-imbalance resampling (imblearn in v0.1) has no v1 resampler
    # domain; only templates that ENABLE resampling exercise this gap.
    ("resampler", "smote"): "no v1 resampler domain yet",
    # DICOM conversion writes NIfTI to disk; it is not an in-memory
    # Subject preprocessor. The batch YAML pipeline still runs it via
    # the v0.1 compat factory.
    ("preprocessor", "dcm2nii"): "IO conversion, not an in-memory Subject step",
}

#: config/ subdirectory -> workflow alias, mirroring the directory rules of
#: ``habit.commands.cmd_check_config._guess_workflow`` so the gate covers
#: exactly the files the CLI would validate.
_DIR_WORKFLOWS: Mapping[str, str] = {
    "preprocessing": "preprocess",
    "dicom_sort": "sort-dicom",
    "feature_extraction": "extract",
    "machine_learning": "model",
    "model_comparison": "compare",
    "habitat": "habitat",
    "radiomics": "radiomics",
    "auxiliary": "icc",
}

#: Basenames under config/ that are not runnable workflow configs: input
#: manifests (referenced via ``data_dir``) and PyRadiomics parameter presets
#: (referenced via ``radiomics_params_file``). ``check-config`` treats these
#: as syntax-only files; translating them as workflows would be meaningless.
_NON_WORKFLOW_PREFIXES = ("file_", "files_", "params_", "parameter")
_NON_WORKFLOW_NAMES = {"image_files.yaml"}



def _registry_for(domain: str) -> Any:
    """
    Return the component registry serving one v1 domain.

    Args:
        domain: Registry domain name.

    Returns:
        The registry class.
    """
    from habit.api.plugins import _registry_for_domain

    return _registry_for_domain(domain)


def _workflow_for(path: Path) -> Optional[str]:
    """
    Return the canonical workflow alias for one template, or ``None`` for
    files that are not runnable workflow configs.

    Args:
        path: Template path under ``config/``.
    """
    name = path.name.lower()
    if name.startswith(".") or name in _NON_WORKFLOW_NAMES:
        return None
    if name.startswith(_NON_WORKFLOW_PREFIXES):
        return None
    directory = path.parent.name.lower()
    alias = _DIR_WORKFLOWS.get(directory)
    if alias is None:
        return None
    # K-fold configs live under machine_learning/ but run through habit cv;
    # test-retest templates live under auxiliary/ (check-config rules).
    if alias == "model" and "kfold" in name:
        return "cv"
    if alias == "icc" and "retest" in name:
        return "retest"
    return alias


def _workflow_templates() -> List[Tuple[Path, str]]:
    """Return every runnable workflow template as ``(path, workflow)``."""
    templates: List[Tuple[Path, str]] = []
    for path in sorted(_CONFIG_ROOT.rglob("*.yaml")):
        workflow = _workflow_for(path)
        if workflow is not None:
            templates.append((path, workflow))
    return templates


def _habitat_components(spec: Mapping[str, Any]) -> Iterator[Tuple[str, str]]:
    """
    Yield ``(domain, name)`` for every component a translated habitat spec
    names, including the preprocessing chains and habitat feature families.
    """
    for domain in (
        "voxel_feature_extractor",
        "supervoxelizer",
        "supervoxel_feature_extractor",
        "habitat_model_fitter",
        "habitat_assigner",
    ):
        entry = spec.get(domain)
        if entry is not None:
            yield domain, str(entry["name"])
    for entry in spec.get("habitat_features") or []:
        yield "habitat_feature_extractor", str(entry["name"])
    for chain_field in (
        "voxel_feature_preprocessors",
        "supervoxel_feature_preprocessors",
        "cohort_feature_preprocessors",
    ):
        for entry in spec.get(chain_field) or []:
            yield "feature_preprocessing_method", str(entry["name"])


def _ml_components(
    spec: Optional[Mapping[str, Any]], legacy: Mapping[str, Any]
) -> Iterator[Tuple[str, str]]:
    """
    Yield ``(domain, name)`` for every component a deep-translated ML
    document names.

    The typed spec names the preprocessing chain, the selection chain and
    the classifier directly. Components v0.1 would ALSO have run but the v1
    spec has no slot for -- the extra models of a multi-model sweep and an
    enabled resampling block -- ride under ``legacy`` and are extracted from
    there, so the coverage gate keeps seeing them. Disabled blocks (e.g.
    ``resampling.enabled: false``) name nothing, mirroring what v0.1 would
    actually run.
    """
    if spec is not None:
        for entry in spec.get("table_preprocessors") or []:
            yield "table_preprocessor", str(entry["name"])
        for entry in spec.get("feature_selectors") or []:
            yield "feature_selector", str(entry["name"])
        classifier = spec.get("classifier")
        if isinstance(classifier, Mapping) and classifier.get("name"):
            yield "classifier", str(classifier["name"])
    named = set()
    if spec is not None and isinstance(spec.get("classifier"), Mapping):
        named.add(str(spec["classifier"]["name"]))
    for name in (legacy.get("models") or {}):
        if str(name) not in named:
            yield "classifier", str(name)
    resampling = legacy.get("resampling") or {}
    if (
        isinstance(resampling, Mapping)
        and resampling.get("enabled")
        and resampling.get("method")
    ):
        yield "resampler", str(resampling["method"])


def _generic_components(
    workflow: str, params: Mapping[str, Any]
) -> Iterator[Tuple[str, str]]:
    """
    Yield ``(domain, name)`` for every component a generically translated
    workflow names inside ``spec.params``.

    Only algorithmic names are extracted; data/output/reporting keys that
    ride along in ``spec.params`` are not components. Disabled blocks (e.g.
    ``resampling.enabled: false``) name nothing, mirroring what v0.1 would
    actually run.
    """
    if workflow == "preprocess":
        steps = params.get("preprocessing") or {}
        for name in steps:
            yield "preprocessor", str(name)
    elif workflow == "extract":
        for name in params.get("feature_types") or []:
            yield "habitat_feature_extractor", str(name)
    elif workflow == "icc":
        for name in params.get("metrics") or []:
            yield "icc_metric", str(name)
    elif workflow == "retest":
        method = params.get("similarity_method")
        if method:
            yield "retest_similarity", str(method)
    elif workflow == "compare":
        delong = params.get("delong_test") or {}
        if isinstance(delong, Mapping) and delong.get("enabled"):
            # Not a registry component: the DeLong AUC comparison is an L0
            # kernel routine in v1, checked separately below.
            yield "kernels", "delong_roc_test"
    # radiomics / sort-dicom name no registry components.


def _translate(path: Path, workflow: str) -> Mapping[str, Any]:
    """Translate one template and return the v1 document."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{path.name} is not a mapping payload."
    return LegacyConfigAdapter().translate(payload, workflow).document


def _components_of(path: Path, workflow: str) -> Iterator[Tuple[str, str]]:
    """Yield every ``(domain, name)`` one template's translation names."""
    document = _translate(path, workflow)
    spec = document.get("spec")
    if workflow in ("model", "cv"):
        # Deep-translated: the typed spec names the modelling components;
        # predict stubs (spec=None) may still name a legacy resampler.
        yield from _ml_components(spec, document.get("legacy") or {})
        return
    if spec is None:
        # Predict stubs take their definition from the fitted model
        # artefact and name no components.
        return
    if workflow == "habitat":
        yield from _habitat_components(spec)
    else:
        yield from _generic_components(workflow, spec.get("params") or {})


def _is_available(domain: str, name: str) -> bool:
    """
    Return whether ``name`` resolves in ``domain``'s v1 home.

    Pseudo-domains without any v1 home (``_KNOWN_GAPS`` territory) resolve
    nothing by construction; the ``kernels`` pseudo-domain checks for an L0
    kernel attribute instead of a registry entry.
    """
    if domain == "kernels":
        import habit.kernels

        return hasattr(habit.kernels, name)
    if domain in ("icc_metric", "retest_similarity", "resampler"):
        return False
    return name in _registry_for(domain).available()


def _collect_gaps() -> List[Gap]:
    """
    Run the whole gate logic and return every unresolved component as
    ``(config path, domain, name)`` -- the phase-4 progress metric.
    """
    gaps: List[Gap] = []
    for path, workflow in _workflow_templates():
        for domain, name in _components_of(path, workflow):
            if not _is_available(domain, name):
                gaps.append((path, domain, name))
    return gaps


@pytest.mark.unit
def test_workflow_templates_are_discoverable() -> None:
    """Guard the guard: an empty glob would make every check below vacuous."""
    assert len(_workflow_templates()) >= 40


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_path,workflow",
    _workflow_templates(),
    ids=lambda item: item.stem if isinstance(item, Path) else str(item),
)
def test_translated_component_names_are_registered(
    config_path: Path, workflow: str
) -> None:
    """Every component a translated template names must exist in v1."""
    missing: List[str] = []
    for domain, name in _components_of(config_path, workflow):
        if (domain, name) in _KNOWN_GAPS:
            continue
        if not _is_available(domain, name):
            missing.append(f"{domain}={name!r}")
    assert not missing, (
        f"{config_path.relative_to(_CONFIG_ROOT)} (workflow={workflow}) "
        f"translates to components v1 cannot build: {missing}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "gap", sorted(_KNOWN_GAPS), ids=lambda g: f"{g[0]}.{g[1]}"
)
def test_known_gaps_are_still_gaps(gap: Tuple[str, str]) -> None:
    """A closed gap must be deleted from the list, not left to rot."""
    domain, name = gap
    assert not _is_available(domain, name), (
        f"{domain}.{name} now resolves in v1: remove it from _KNOWN_GAPS so "
        f"the coverage gate starts enforcing it. (Recorded reason: "
        f"{_KNOWN_GAPS[gap]})"
    )


@pytest.mark.unit
def test_every_known_gap_is_exercised_by_a_template() -> None:
    """The pinned debt list must match reality exactly, in both directions.

    A known gap no template exercises anymore would silently shrink the
    metric; an unknown gap means a template started needing a capability v1
    lacks. Either way the phase-4 progress list is wrong, so fail.
    """
    exercised = {(domain, name) for _, domain, name in _collect_gaps()}
    assert exercised == set(_KNOWN_GAPS), (
        f"Gap list drifted. Exercised but unknown: "
        f"{sorted(exercised - set(_KNOWN_GAPS))}; pinned but no longer "
        f"exercised: {sorted(set(_KNOWN_GAPS) - exercised)}."
    )


@pytest.mark.unit
def test_gap_report(capsys: pytest.CaptureFixture[str]) -> None:
    """Print the phase-4 progress table (visible with ``pytest -s``).

    The report itself is the deliverable: one row per
    (template, domain, missing name), plus pass/fail counts over all
    workflow templates.
    """
    templates = _workflow_templates()
    gaps = _collect_gaps()
    gap_configs = {path for path, _, _ in gaps}
    lines = [
        "",
        "=" * 72,
        "PHASE-4a TEMPLATE x DOMAIN COVERAGE REPORT",
        "=" * 72,
        f"workflow templates checked : {len(templates)}",
        f"templates fully covered    : {len(templates) - len(gap_configs)}",
        f"templates with gaps        : {len(gap_configs)}",
        f"missing component names    : {len(gaps)}",
        "-" * 72,
    ]
    for path, domain, name in gaps:
        lines.append(
            f"GAP  {path.relative_to(_CONFIG_ROOT)}  domain={domain}  name={name}"
        )
    lines.append("=" * 72)
    report = "\n".join(lines)
    print(report)
    captured = capsys.readouterr()
    assert "PHASE-4a" in captured.out
