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
"""Tests for the v0 -> v1 legacy config translation (habit.spec.legacy)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import pytest
import yaml

from habit.api.exceptions import HABITAPIError
from habit.spec.legacy import (
    LegacyConfigAdapter,
    default_migrated_path,
    detect_yaml_version,
    migrate_yaml,
    validate_v1_document,
)
from habit.spec.specs import HabitatSpec
from habit.spec.policy import RunPolicy

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
HABITAT_CONFIG_DIR: Path = PROJECT_ROOT / "config" / "habitat"

#: Bundled habitat YAMLs that are data manifests, not HabitatAnalysisConfig.
_MANIFEST_PREFIXES: Tuple[str, ...] = ("file_habitat",)


def _load_config(name: str) -> Dict[str, Any]:
    """Load one bundled habitat config as a plain mapping."""
    path = HABITAT_CONFIG_DIR / name
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _translate(name: str) -> Any:
    """Translate one bundled habitat config with the adapter."""
    return LegacyConfigAdapter().translate(_load_config(name), "habitat")


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_detect_yaml_version_v0_for_flat_config() -> None:
    """A classic v0 habitat config classifies as v0."""
    assert detect_yaml_version(_load_config("config_habitat_two_step.yaml")) == "v0"


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    (
        {"version": "1.0", "workflow": "habitat"},
        {"spec": {}, "policy": {}},  # characteristic section pair, no tag
        {"spec": {}, "data": {}},
    ),
)
def test_detect_yaml_version_v1(payload: Mapping[str, Any]) -> None:
    """v1 is recognised by tag or by the spec+policy/data section pair."""
    assert detect_yaml_version(payload) == "v1"


@pytest.mark.unit
def test_detect_yaml_version_rejects_non_mapping() -> None:
    """Non-mapping payloads cannot be classified."""
    with pytest.raises(HABITAPIError):
        detect_yaml_version(["not", "a", "mapping"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Habitat deep translation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_two_step_translation_full_document() -> None:
    """The canonical two-step config lands in every typed v1 slot."""
    translation = _translate("config_habitat_two_step.yaml")
    document = translation.document

    assert document["version"] == "1.0"
    assert document["workflow"] == "habitat"
    assert document["mode"] == "train"

    spec = document["spec"]
    assert spec["voxel_feature_extractor"] == {
        "name": "raw",
        "params": {"modalities": ["delay2", "delay3", "delay5"]},
    }
    supervoxelizer = spec["supervoxelizer"]
    assert supervoxelizer["name"] == "kmeans"
    assert supervoxelizer["params"]["n_supervoxels"] == 50

    fitter = spec["habitat_model_fitter"]
    assert fitter["name"] == "kmeans"
    assert fitter["params"]["validation"] == "elbow"
    assert fitter["params"]["n_habitats"] is None
    assert fitter["params"]["min_habitats"] == 2
    assert fitter["params"]["max_habitats"] == 10

    assert spec["habitat_assigner"] == {"name": "nearest_centroid", "params": {}}
    assert [s["name"] for s in spec["subject_table_preprocessors"]] == [
        "winsorize",
        "minmax",
    ]
    assert [s["name"] for s in spec["group_table_preprocessors"]] == ["binning"]
    winsorize = spec["subject_table_preprocessors"][0]
    assert winsorize["params"]["winsor_limits"] == [0.05, 0.05]
    assert spec["random_seed"] == 42

    # The spec section must parse into the typed v1 model unchanged.
    habitat_spec = HabitatSpec.from_dict(spec)
    assert habitat_spec.random_seed == 42

    policy = document["policy"]
    assert policy["workers"] == 2
    assert policy["backend"] == "process"
    assert policy["subject_timeout_sec"] == 900
    assert policy["parallel_mode"] == "persistent"
    assert policy["auto_retry_rounds"] == 2
    # Keys absent from the v0 file stay absent (specs record deviations
    # only); the typed model fills its defaults, matching v0 semantics.
    run_policy = RunPolicy.from_dict(policy)
    assert run_policy.on_subject_failure == "continue"

    assert document["data"]["source"].endswith("processed_images")
    assert document["output"]["out_dir"].endswith("habitat_two_step")
    assert document["output"]["save_results_csv"] is True

    # Worker-lifecycle knobs without a RunPolicy field survive verbatim.
    assert translation.unmapped["persistent_worker_max_consecutive_failures"] == 1
    assert translation.unmapped["persistent_worker_recycle_after_tasks"] == 0
    assert translation.warnings  # preservation was reported, not silent

    validate_v1_document(document)


@pytest.mark.unit
def test_one_step_translation_uses_one_step_settings() -> None:
    """one_step: no supervoxelizer; fitter params come from one_step_settings."""
    translation = _translate("config_habitat_one_step_voxel_radiomics_train.yaml")
    spec = translation.document["spec"]

    assert spec["supervoxelizer"] is None
    fitter = spec["habitat_model_fitter"]
    assert fitter["name"] == "kmeans"
    assert fitter["params"]["min_habitats"] == 2
    assert fitter["params"]["max_habitats"] == 10
    assert fitter["params"]["validation"] == "elbow"

    assert spec["voxel_feature_extractor"]["name"] == "voxel_radiomics"
    assert spec["voxel_feature_extractor"]["params"]["modalities"] == ["delay2"]

    # The spec parses even with supervoxelizer=None (one-step assembly).
    habitat_spec = HabitatSpec.from_dict(spec)
    assert habitat_spec.supervoxelizer is None
    validate_v1_document(translation.document)


@pytest.mark.unit
def test_direct_pooling_translation_uses_habitat_block() -> None:
    """direct_pooling: no supervoxelizer; cohort fitter from the habitat block."""
    translation = _translate("config_habitat_direct_pooling_gmm_aic.yaml")
    spec = translation.document["spec"]

    assert spec["supervoxelizer"] is None
    fitter = spec["habitat_model_fitter"]
    assert fitter["name"] == "gmm"
    assert fitter["params"]["validation"] == "aic"

    # concat(voxel_radiomics(A), voxel_radiomics(B), ...) is homogeneous.
    extractor = spec["voxel_feature_extractor"]
    assert extractor["name"] == "voxel_radiomics"
    assert extractor["params"]["modalities"] == ["delay2", "delay3", "delay5"]
    validate_v1_document(translation.document)


@pytest.mark.unit
def test_predict_mode_translation_has_no_spec() -> None:
    """Predict configs omit the spec; the pipeline path carries the definition."""
    translation = _translate("config_habitat_two_step_predict.yaml")
    document = translation.document

    assert document["mode"] == "predict"
    assert document["spec"] is None
    assert document["pipeline"].endswith("habitat_pipeline.pkl")
    validate_v1_document(document)


@pytest.mark.unit
def test_supervoxel_radiomics_aggregator_both_spellings() -> None:
    """supervoxel_radiomics is recognised bare and inside a concat() wrapper."""
    translation = _translate("config_habitat_two_step_supervoxel_radiomics_train.yaml")
    supervoxelizer = translation.document["spec"]["supervoxelizer"]

    assert supervoxelizer["name"] == "supervoxel_radiomics"
    assert supervoxelizer["params"]["modalities"] == ["delay2"]
    # The label-growing half of the fused v0 blocks is preserved.
    assert supervoxelizer["params"]["label_algorithm"] == "kmeans"
    assert supervoxelizer["params"]["n_supervoxels"] == 50
    # Aggregator params ride along verbatim.
    assert supervoxelizer["params"]["use_supervoxel_cext"] == "auto"
    validate_v1_document(translation.document)


@pytest.mark.unit
def test_slic_supervoxelizer_params() -> None:
    """SLIC keeps its own knob set (compactness/sigma/connectivity)."""
    translation = _translate("config_habitat_two_step_supervoxel_slic.yaml")
    supervoxelizer = translation.document["spec"]["supervoxelizer"]

    assert supervoxelizer["name"] == "slic"
    params = supervoxelizer["params"]
    assert params["n_supervoxels"] == 50
    assert "compactness" in params
    assert "sigma" in params
    assert "enforce_connectivity" in params


@pytest.mark.unit
def test_fixed_cluster_count_maps_to_n_habitats() -> None:
    """fixed_n_clusters becomes the v1 n_habitats shortcut."""
    translation = _translate("config_habitat_two_step_fixed_n_train.yaml")
    fitter = translation.document["spec"]["habitat_model_fitter"]
    assert fitter["params"]["n_habitats"] is not None
    assert fitter["params"]["n_habitats"] >= 2


@pytest.mark.unit
def test_per_component_seed_preserved_with_warning() -> None:
    """A component-level seed has no v1 slot; it is preserved and reported."""
    payload = _load_config("config_habitat_two_step.yaml")
    payload["habitat_segmentation"]["supervoxel"]["random_state"] = 7
    translation = LegacyConfigAdapter().translate(payload, "habitat")

    assert translation.document["spec"]["random_seed"] == 42
    seeds = translation.unmapped["habitat_segmentation"]["supervoxel"]
    assert seeds["random_state"] == 7
    assert any("random_state" in note for note in translation.warnings)


@pytest.mark.unit
def test_multiple_selection_methods_first_wins() -> None:
    """A selection-method list keeps its first entry as validation."""
    payload = _load_config("config_habitat_two_step.yaml")
    payload["habitat_segmentation"]["habitat"][
        "habitat_cluster_selection_method"
    ] = ["elbow", "silhouette"]
    translation = LegacyConfigAdapter().translate(payload, "habitat")

    params = translation.document["spec"]["habitat_model_fitter"]["params"]
    assert params["validation"] == "elbow"
    assert params["selection_methods"] == ["elbow", "silhouette"]
    assert any("selection" in note for note in translation.warnings)


# ---------------------------------------------------------------------------
# Expression mini-parser
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "expression, expected_name, expected_modalities",
    (
        ("raw(delay2)", "raw", ["delay2"]),
        ("concat(raw(A), raw(B))", "raw", ["A", "B"]),
        ("voxel_radiomics(T1)", "voxel_radiomics", ["T1"]),
        ("concat(voxel_radiomics(T1), voxel_radiomics(T2))",
         "voxel_radiomics", ["T1", "T2"]),
        # Bare tokens inside concat default to raw, mirroring the v0 parser.
        ("concat(T1, T2)", "raw", ["T1", "T2"]),
    ),
)
def test_voxel_expression_homogeneous_forms(
    expression: str, expected_name: str, expected_modalities: list
) -> None:
    """Homogeneous expressions collapse into name + modalities."""
    adapter = LegacyConfigAdapter()
    warnings: list = []
    spec = adapter._translate_voxel_extractor(
        {"method": expression, "params": {}}, warnings
    )
    assert spec["name"] == expected_name
    assert spec["params"]["modalities"] == expected_modalities
    assert "expression" not in spec["params"]


@pytest.mark.unit
def test_voxel_expression_composite_keeps_verbatim_expression() -> None:
    """Composite expressions (kinetic) keep the original string verbatim."""
    adapter = LegacyConfigAdapter()
    warnings: list = []
    expression = "kinetic(raw(pre), raw(LAP), timestamps)"
    spec = adapter._translate_voxel_extractor(
        {"method": expression, "params": {"timestamps": "ts.xlsx"}}, warnings
    )
    assert spec["name"] == "kinetic"
    assert spec["params"]["modalities"] == ["pre", "LAP"]
    assert spec["params"]["expression"] == expression
    assert spec["params"]["timestamps"] == "ts.xlsx"
    assert warnings


@pytest.mark.unit
def test_voxel_expression_missing_method_rejected() -> None:
    """A train config without a voxel method expression fails loudly."""
    adapter = LegacyConfigAdapter()
    with pytest.raises(HABITAPIError):
        adapter._translate_voxel_extractor({"params": {}}, [])


@pytest.mark.unit
def test_expression_parser_rejects_unbalanced_parentheses() -> None:
    """Malformed expressions surface as HABITAPIError, not IndexError."""
    from habit.spec.legacy import _parse_method_expression

    with pytest.raises(HABITAPIError):
        _parse_method_expression("concat(raw(A)")


# ---------------------------------------------------------------------------
# Generic workflow translation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generic_workflow_translation_splits_sections() -> None:
    """Non-habitat configs split into data/policy/output plus a spec blob."""
    payload = {
        "run_mode": "train",
        "data_path": "features.csv",
        "out_dir": "results",
        "n_processes": 4,
        "models": {"SVM": {"params": {}}},
        "feature_selection": {"method": "anova"},
    }
    translation = LegacyConfigAdapter().translate(payload, "model")
    document = translation.document

    assert document["workflow"] == "model"
    assert document["mode"] == "train"
    assert document["data"] == {"data_path": "features.csv"}
    assert document["policy"]["workers"] == 4
    assert document["policy"]["backend"] == "process"
    assert document["output"] == {"out_dir": "results"}
    # The algorithmic remainder is preserved verbatim, lossless by design.
    assert document["spec"]["name"] == "model"
    assert document["spec"]["params"]["models"] == {"SVM": {"params": {}}}
    assert document["spec"]["params"]["feature_selection"] == {
        "method": "anova"
    }
    assert translation.warnings
    validate_v1_document(document)


@pytest.mark.unit
def test_generic_radiomics_nested_blocks_split() -> None:
    """Radiomics paths/processing blocks unfold into data/policy/output."""
    payload = {
        "paths": {"params_file": "p.yaml", "images_folder": "img", "out_dir": "out"},
        "processing": {"n_processes": 2, "target_labels": [1]},
        "export": {"export_format": "csv"},
    }
    translation = LegacyConfigAdapter().translate(payload, "radiomics")
    document = translation.document

    assert document["data"]["images_folder"] == "img"
    assert document["data"]["params_file"] == "p.yaml"
    assert document["output"]["out_dir"] == "out"
    assert document["output"]["export"] == {"export_format": "csv"}
    assert document["policy"]["workers"] == 2
    assert document["spec"]["params"]["processing"] == {"target_labels": [1]}


@pytest.mark.unit
def test_unknown_workflow_rejected() -> None:
    """Translation refuses workflows outside the frozen v0 set."""
    with pytest.raises(HABITAPIError):
        LegacyConfigAdapter().translate({}, "not-a-workflow")


# ---------------------------------------------------------------------------
# migrate_yaml / validate_v1_document
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_migrate_yaml_writes_default_path_and_roundtrips(tmp_path: Path) -> None:
    """A migrated file classifies as v1 and validates; re-migrating refuses."""
    source = tmp_path / "my_habitat.yaml"
    source.write_text(
        (HABITAT_CONFIG_DIR / "config_habitat_two_step.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    report = migrate_yaml(source)
    assert report.destination == default_migrated_path(source)
    assert report.destination is not None and report.destination.is_file()
    assert report.diff  # unified diff was produced for the report

    migrated = yaml.safe_load(report.destination.read_text(encoding="utf-8"))
    assert detect_yaml_version(migrated) == "v1"
    validate_v1_document(migrated)
    HabitatSpec.from_dict(migrated["spec"])

    # A second migration run on the v1 file is a clear no-op error.
    with pytest.raises(HABITAPIError):
        migrate_yaml(report.destination)


@pytest.mark.unit
def test_migrate_yaml_dry_run_writes_nothing(tmp_path: Path) -> None:
    """--dry-run reports the diff without touching the filesystem."""
    source = tmp_path / "cfg.yaml"
    source.write_text(
        (HABITAT_CONFIG_DIR / "config_habitat_two_step_minimal.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    report = migrate_yaml(source, dry_run=True, workflow="habitat")
    assert report.destination is None
    assert report.diff
    assert not default_migrated_path(source).exists()


@pytest.mark.unit
def test_migrate_yaml_requires_workflow_when_unguessable(tmp_path: Path) -> None:
    """An unguessable path without an explicit workflow alias fails clearly."""
    source = tmp_path / "opaque_name.yaml"
    source.write_text("data_dir: x\nout_dir: y\n", encoding="utf-8")
    with pytest.raises(HABITAPIError):
        migrate_yaml(source)
    report = migrate_yaml(source, workflow="habitat")
    assert report.workflow == "habitat"


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload, message_fragment",
    (
        ({"workflow": "habitat"}, "version"),
        ({"version": "1.0"}, "workflow"),
        (
            {"version": "1.0", "workflow": "habitat", "mode": "train"},
            "spec",
        ),
        (
            {"version": "1.0", "workflow": "habitat", "mode": "predict"},
            "pipeline",
        ),
        (
            {"version": "1.0", "workflow": "model", "policy": {"workers": 0}},
            "workers",
        ),
        (
            {"version": "1.0", "workflow": "model", "policy": {"bogus_key": 1}},
            "Unknown RunPolicy field",
        ),
    ),
)
def test_validate_v1_document_rejects_bad_structure(
    payload: Mapping[str, Any], message_fragment: str
) -> None:
    """Structural violations fail with a message naming the offender."""
    with pytest.raises(HABITAPIError, match=message_fragment):
        validate_v1_document(payload)


@pytest.mark.unit
def test_validate_v1_document_workflow_mismatch() -> None:
    """An explicit -w alias must agree with the document's workflow tag."""
    payload = {"version": "1.0", "workflow": "habitat", "mode": "predict",
               "pipeline": "m.pkl"}
    with pytest.raises(HABITAPIError, match="workflow"):
        validate_v1_document(payload, workflow="model")


# ---------------------------------------------------------------------------
# Bundled-config sweep: every shipped habitat config translates losslessly
# ---------------------------------------------------------------------------


def _discover_habitat_configs() -> list:
    """Return every bundled habitat pipeline config (manifests excluded)."""
    return sorted(
        path
        for path in HABITAT_CONFIG_DIR.glob("*.yaml")
        if not path.name.startswith(_MANIFEST_PREFIXES)
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "config_path",
    _discover_habitat_configs(),
    ids=[p.stem for p in _discover_habitat_configs()],
)
def test_bundled_habitat_config_translates_and_validates(config_path: Path) -> None:
    """Every bundled habitat config yields a valid v1 document."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert detect_yaml_version(payload) == "v0"

    translation = LegacyConfigAdapter().translate(payload, "habitat")
    document = translation.document
    validate_v1_document(document)

    if document["mode"] == "predict" and document["spec"] is None:
        assert document["pipeline"]
        return

    # Train documents parse into the typed v1 models unchanged.
    HabitatSpec.from_dict(document["spec"])
    RunPolicy.from_dict(document["policy"])


# ---------------------------------------------------------------------------
# Cross-workflow sweep: every bundled CLI config translates losslessly
# ---------------------------------------------------------------------------

#: config/ subdirectory -> workflow alias, mirroring the CLI's guessing.
_CROSS_WORKFLOW_DIR_RULES: Mapping[str, Optional[str]] = {
    "preprocessing": "preprocess",
    "dicom_sort": "sort-dicom",
    "feature_extraction": "extract",
    "machine_learning": "model",
    "model_comparison": "compare",
    "radiomics": "radiomics",
    "auxiliary": None,  # resolved per filename below
}

#: Helper YAMLs that are not pipeline entry configs.
_CROSS_SKIP_PREFIXES: Tuple[str, ...] = ("files_", "file_habitat", "image_files")


def _discover_cli_configs() -> list:
    """Return ``(path, workflow)`` for every bundled CLI entry config."""
    discovered = []
    for path in sorted((PROJECT_ROOT / "config").rglob("*.yaml")):
        if any(path.name.startswith(p) for p in _CROSS_SKIP_PREFIXES):
            continue
        subdir = path.parent.name
        if subdir == "habitat":
            continue  # covered by the dedicated habitat sweep
        if subdir not in _CROSS_WORKFLOW_DIR_RULES:
            continue
        workflow = _CROSS_WORKFLOW_DIR_RULES[subdir]
        name = path.name.lower()
        if subdir == "auxiliary":
            workflow = (
                "retest" if ("retest" in name or "test_retest" in name) else "icc"
            )
        if subdir == "radiomics" and not path.name.startswith("config_traditional"):
            continue  # radiomics parameter presets, not CLI configs
        if workflow == "model" and "kfold" in name:
            workflow = "cv"
        discovered.append((path, workflow))
    return discovered


_CROSS_CONFIGS: list = _discover_cli_configs()


@pytest.mark.integration
@pytest.mark.parametrize(
    "config_and_workflow",
    _CROSS_CONFIGS,
    ids=[entry[0].stem for entry in _CROSS_CONFIGS],
)
def test_bundled_cli_config_translates_and_validates(
    config_and_workflow: Tuple[Path, str]
) -> None:
    """Every bundled non-habitat CLI config yields a valid v1 document."""
    config_path, workflow = config_and_workflow
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert detect_yaml_version(payload) == "v0"

    translation = LegacyConfigAdapter().translate(payload, workflow)
    document = translation.document
    validate_v1_document(document)

    # Lossless by construction: every algorithmic key survives inside the
    # document (spec blob or a typed section).
    assert document["spec"] is not None
    assert document["workflow"] == workflow
