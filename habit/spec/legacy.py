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
"""Legacy v0 YAML translation: the bridge into the v1 document layout.

The v0.1 "god configuration object" mixes four concerns in one flat mapping:
where the data lives, what the algorithm does, how execution is scheduled,
and where results are written. The v1 document keeps them apart:

.. code-block:: yaml

    version: "1.0"
    workflow: habitat
    mode: train
    spec:        # HabitatSpec payload -- changes the scientific result
    data:        # DataSource description -- where the cohort comes from
    pipeline:    # predict-mode model path (habitat/model workflows only)
    policy:      # RunPolicy payload -- scheduling only, never scientific
    output:      # out_dir plus output switches -- a ResultWriter concern
    legacy:      # v0 settings with no v1 slot yet, preserved verbatim

The translation is LOSSLESS BY CONSTRUCTION: every v0 key either lands in a
typed v1 slot or is copied verbatim into ``legacy`` with a human-readable
warning, so ``habit migrate-config`` never silently drops a setting.

This module deliberately performs PURE MAPPING translation: it never imports
the v0 ``habit.core.*`` schema chain, keeping ``habit.spec`` light and
satisfying the layer contract (tests/test_architecture_contracts.py). The
v0 schemas remain the validation authority for v0 files inside
``habit check-config``; here we only need to understand the YAML shape.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

from habit.exceptions import HABITAPIError
from habit.spec.expressions import parse_feature_expression
from habit.spec.policy import RunPolicy
from habit.spec.specs import HabitatSpec, MLSpec
from habit.spec.yaml_io import _read_yaml, _write_yaml

__all__ = [
    "LegacyConfigAdapter",
    "LegacyTranslation",
    "MigrationReport",
    "detect_yaml_version",
    "migrate_yaml",
    "validate_v1_document",
]

#: Version tag written at the top of every migrated document.
V1_SCHEMA_VERSION = "1.0"

#: Top-level sections of a v1 document, in canonical emission order.
V1_DOCUMENT_SECTIONS: Tuple[str, ...] = (
    "version",
    "workflow",
    "mode",
    "spec",
    "data",
    "pipeline",
    "policy",
    "output",
    "legacy",
)

#: Workflows ``habit check-config`` knows how to validate; kept in sync with
#: ``habit/commands/cmd_check_config.py::_WORKFLOW_LOADERS``.
_KNOWN_WORKFLOWS: Tuple[str, ...] = (
    "preprocess",
    "habitat",
    "extract",
    "radiomics",
    "model",
    "cv",
    "compare",
    "icc",
    "retest",
    "sort-dicom",
)

# ---------------------------------------------------------------------------
# v0 habitat field classification
# ---------------------------------------------------------------------------

#: v0 habitat execution keys -> RunPolicy field names. Renames follow the
#: v1 naming decisions (``processes`` -> ``workers`` and friends).
_HABITAT_POLICY_KEY_MAP: Mapping[str, str] = {
    "processes": "workers",
    "individual_subject_timeout_sec": "subject_timeout_sec",
    "individual_subject_spawn_timeout_sec": "subject_spawn_timeout_sec",
    "individual_subject_graceful_shutdown_sec": "graceful_shutdown_sec",
    "on_subject_failure": "on_subject_failure",
    "oom_backoff": "oom_backoff",
    "oom_reduce_workers_by": "oom_reduce_workers_by",
    "cap_processes_to_gpu_pool": "cap_workers_to_gpu_pool",
    "resume": "resume",
    "checkpoint_dir": "checkpoint_dir",
    "individual_subject_parallel_mode": "parallel_mode",
    "individual_subject_auto_retry_rounds": "auto_retry_rounds",
    "retry_failed_subjects": "retry_failed_subjects",
    "force_rerun_subjects": "force_rerun_subjects",
    "clear_checkpoint_on_success": "clear_checkpoint_on_success",
    "strict_checkpoint_hash": "strict_checkpoint_hash",
    "persistent_worker_max_consecutive_failures": (
        "persistent_worker_max_consecutive_failures"
    ),
    "persistent_worker_recycle_after_tasks": (
        "persistent_worker_recycle_after_tasks"
    ),
}

#: v0 habitat output-switch keys moved under the document ``output`` section.
_HABITAT_OUTPUT_KEYS: Tuple[str, ...] = (
    "plot_curves",
    "save_images",
    "save_results_csv",
    "habitats_results_format",
    "verbose",
    "debug",
)

#: v0 habitat top-level keys consumed by dedicated v1 slots; everything else
#: found at the top level is preserved under ``legacy``.
_HABITAT_CONSUMED_TOP_KEYS: frozenset = frozenset(
    {
        "run_mode",
        "data_dir",
        "out_dir",
        "config_file",
        "pipeline_path",
        "feature_construction",
        "habitat_segmentation",
        "random_state",
        # Optional Stage-1 geometry policy (HabitatSpec.on_geometry_mismatch).
        "on_geometry_mismatch",
    }
    | set(_HABITAT_POLICY_KEY_MAP)
    | set(_HABITAT_OUTPUT_KEYS)
)

#: v0 preprocessing-method parameter keys forwarded into the feature
#: preprocessing method Spec params; ``method`` becomes the Spec name.
_PREPROCESSING_NAME_KEY = "method"

#: Renamed preprocessing parameter: v0's ``global_normalize`` described
#: pooling across feature COLUMNS, not across subjects. See
#: ``_translate_preprocessing_chain``.
_V0_SCOPE_KEY = "global_normalize"
_V1_SCOPE_KEY = "across_features"

#: v0.1 preprocessing methods that actually consumed ``global_normalize``
#: (the scalers and binning, which pool statistics across feature columns).
#: The filters (``variance_filter`` / ``correlation_filter``) accepted the
#: key in their config block but never read it, so forwarding it would state
#: a parameter the v1 component does not declare -- it goes to the legacy
#: sink instead. The set is a property of the frozen v0.1 vocabulary.
_V0_SCOPE_AWARE_METHODS: frozenset = frozenset(
    {"minmax", "zscore", "robust", "binning", "winsorize", "log"}
)

#: v0.1 selector params that switch plotting, not the algorithm
#: (``visualize`` on lasso/rfecv/vif/correlation, ``plot_variances`` on
#: variance). v1 keeps visualisation in the reporting layer, so these keys
#: move to the legacy sink with a warning instead of becoming Spec params.
_V0_SELECTOR_PLOT_KEYS: frozenset = frozenset({"visualize", "plot_variances"})

# ---------------------------------------------------------------------------
# Generic (non-habitat) workflow field classification
# ---------------------------------------------------------------------------

#: Top-level keys of any v0 workflow that describe where data comes from.
_GENERIC_DATA_KEYS: frozenset = frozenset(
    {
        "data_dir",
        "images_folder",
        "raw_img_folder",
        "habitats_map_folder",
        "habitat_pattern",
        "params_file",
        "params_file_of_non_habitat",
        "params_file_of_habitat",
        "files_config",
        "feature_file",
        "data_path",
    }
)

#: Top-level keys of any v0 workflow that describe result writing.
_GENERIC_OUTPUT_KEYS: frozenset = frozenset(
    {
        "out_dir",
        "output_dir",
        "export",
        "logging",
        "save_results_csv",
        "plot_curves",
        "save_images",
        "verbose",
        "debug",
    }
)

#: Top-level execution keys of any v0 workflow -> RunPolicy field names.
_GENERIC_POLICY_KEY_MAP: Mapping[str, str] = {
    "processes": "workers",
    "n_processes": "workers",
    "num_workers": "workers",
    "n_jobs": "workers",
}

#: Mode/entry keys of any v0 workflow, mapped onto document-level slots.
_GENERIC_MODE_KEYS: Mapping[str, str] = {
    "run_mode": "mode",
    "pipeline_path": "pipeline",
    "model_path": "pipeline",
}

# ---------------------------------------------------------------------------
# v0 machine-learning (model/cv) field classification
# ---------------------------------------------------------------------------

#: v0 ``normalization.method`` names -> v1 ``table_preprocessor`` registry
#: names. Methods without a v1 component are preserved under ``legacy``.
_ML_NORMALIZATION_NAME_MAP: Mapping[str, str] = {
    "z_score": "zscore",
    "min_max": "minmax",
    "robust": "robust",
}

#: v0 validation-design keys. They shape the SPLIT, not the model
#: definition, and the v1 MLSpec deliberately has no slot for them (the
#: calling recipe owns the validation design), so they ride under ``legacy``.
_ML_VALIDATION_KEYS: Tuple[str, ...] = (
    "n_splits",
    "stratified",
    "test_size",
    "split_method",
    "train_ids_file",
    "test_ids_file",
)

#: v0 predict-mode output-schema keys, collected under the document
#: ``output`` section (a ResultWriter concern).
_ML_PREDICT_OUTPUT_KEYS: Tuple[str, ...] = (
    "evaluate",
    "output_label_col",
    "output_prob_col",
    "probability_class_index",
    "binary_positive_class_index",
)

#: v0 reporting switches collected under ``output``.
_ML_OUTPUT_SWITCH_KEYS: Tuple[str, ...] = (
    "is_visualize",
    "is_save_model",
    "visualization",
)

#: v0 selector-registry defaults of ``default_before_z_score``, i.e. which
#: stage a selector runs in when the config omits the parameter. The v0
#: registry (``habit.compat.engines.machine_learning.feature_selectors``)
#: defaults every selector to False and only ``variance`` opts into True --
#: after z-scoring every variance is 1.0, so variance selection is vacuous
#: post-normalisation. ``habit.spec`` sits below ``habit.compat`` in the
#: layering and must not import the registry, so the (frozen, v0.1) mapping
#: is mirrored here; ``tests/spec/test_legacy.py`` cross-checks it against
#: the registry so a v0-side change cannot silently drift.
_V0_SELECTOR_DEFAULT_BEFORE_Z_SCORE: Dict[str, bool] = {"variance": True}

#: v0 ML top-level keys consumed by dedicated v1 slots; everything else
#: found at the top level is preserved under ``legacy``.
_ML_CONSUMED_TOP_KEYS: frozenset = frozenset(
    {
        "run_mode",
        "config_file",
        "random_state",
        "input",
        "output",
        "pipeline_path",
        "model_path",
        "normalization",
        "feature_selection_methods",
        "models",
        "resampling",
        "sampling",
        "bootstrap",
        "strict_model_params",
    }
    | set(_ML_VALIDATION_KEYS)
    | set(_ML_PREDICT_OUTPUT_KEYS)
    | set(_ML_OUTPUT_SWITCH_KEYS)
    | _GENERIC_DATA_KEYS
    | set(_GENERIC_POLICY_KEY_MAP)
    | _GENERIC_OUTPUT_KEYS
)


@dataclass(frozen=True)
class LegacyTranslation:
    """
    Result of translating one v0 payload into the v1 document layout.

    Attributes:
        document: The v1 YAML-isomorphic payload (plain dict).
        workflow: Workflow alias the payload was translated as.
        unmapped: v0 settings preserved verbatim under ``document["legacy"]``.
        warnings: Human-readable notes about lossy or unmapped decisions,
            intended for the CLI migration report.
    """

    document: Dict[str, Any]
    workflow: str
    unmapped: Mapping[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MigrationReport:
    """
    Outcome of one ``migrate_yaml`` call.

    Attributes:
        source: The v0 file read.
        destination: The v1 file written; ``None`` for a dry run.
        workflow: Workflow alias detected or supplied.
        document: The translated v1 document payload.
        diff: Unified diff between the source text and the v1 text.
        warnings: Translation warnings, see :class:`LegacyTranslation`.
    """

    source: Path
    destination: Optional[Path]
    workflow: str
    document: Dict[str, Any]
    diff: str
    warnings: Tuple[str, ...] = ()


def detect_yaml_version(payload: Mapping[str, Any]) -> str:
    """
    Detect whether a YAML payload follows the v0 or v1 layout.

    A payload is v1 when it carries an explicit ``version: "1.x"`` tag or the
    characteristic ``spec`` + (``policy`` | ``data``) section pair; every
    other mapping is treated as v0. Detection never raises on shape -- it is
    a classifier, not a validator.

    Args:
        payload: Top-level YAML mapping.

    Returns:
        ``"v0"`` or ``"v1"``.
    """
    if not isinstance(payload, Mapping):
        raise HABITAPIError(
            "Cannot detect config version: the YAML root must be a mapping; "
            f"got {type(payload).__name__}."
        )
    version = str(payload.get("version", "")).strip()
    if version.startswith("1."):
        return "v1"
    if "spec" in payload and ("policy" in payload or "data" in payload):
        return "v1"
    return "v0"


class LegacyConfigAdapter:
    """
    Translate frozen v0 YAML mappings into the v1 document model.

    The adapter is stateless: one instance serves every workflow. The v0
    schema is frozen, so the translation table is a fixed, reviewable
    mapping rather than open-ended introspection.
    """

    #: Version tag emitted into translated documents.
    SCHEMA_VERSION = V1_SCHEMA_VERSION

    def translate(
        self, payload: Mapping[str, Any], workflow: str
    ) -> LegacyTranslation:
        """
        Translate one v0 payload into the v1 document layout.

        Args:
            payload: Top-level v0 YAML mapping.
            workflow: Workflow alias, e.g. ``"habitat"`` or ``"model"``.

        Returns:
            The translation result holding the v1 document, the verbatim
            ``legacy`` remainder, and warnings.

        Raises:
            HABITAPIError: On a non-mapping payload or an unknown workflow.
        """
        if not isinstance(payload, Mapping):
            raise HABITAPIError(
                "Legacy config translation expects a mapping payload; "
                f"got {type(payload).__name__}."
            )
        alias = str(workflow).strip().lower()
        if alias not in _KNOWN_WORKFLOWS:
            raise HABITAPIError(
                f"Unknown workflow for legacy translation: {workflow!r}. "
                f"Known workflows: {', '.join(_KNOWN_WORKFLOWS)}."
            )
        if alias == "habitat":
            return self._translate_habitat(payload)
        if alias in ("model", "cv"):
            return self._translate_ml(payload, alias)
        return self._translate_generic(payload, alias)

    # ------------------------------------------------------------------
    # Habitat workflow: deep translation into HabitatSpec slots
    # ------------------------------------------------------------------

    def _translate_habitat(self, payload: Mapping[str, Any]) -> LegacyTranslation:
        """
        Deep-translate a v0 ``HabitatAnalysisConfig`` mapping.

        Every feature-construction and habitat-segmentation block lands in
        its typed :class:`HabitatSpec` slot; execution keys become a
        :class:`RunPolicy` payload; data/output keys form their own
        sections. Enabled connected-component post-processing maps into
        ``HabitatSpec.postprocess_*``. Remaining settings with no v1 slot
        yet (per-component seeds, plotting switches inside clustering
        blocks) are preserved under ``legacy``.

        Args:
            payload: v0 habitat YAML mapping.

        Returns:
            The translation result.
        """
        warnings: List[str] = []
        unmapped: Dict[str, Any] = {}

        mode = str(payload.get("run_mode", "train")).strip().lower()
        feature_construction = payload.get("feature_construction") or {}
        segmentation = payload.get("habitat_segmentation") or {}
        clustering_mode = str(
            segmentation.get("clustering_mode", "two_step")
        ).strip().lower()

        document: Dict[str, Any] = {
            "version": self.SCHEMA_VERSION,
            "workflow": "habitat",
            "mode": mode,
            "spec": self._translate_habitat_spec(
                payload,
                feature_construction,
                segmentation,
                clustering_mode,
                warnings,
                unmapped,
            ),
            "data": {"source": payload.get("data_dir")},
            "pipeline": payload.get("pipeline_path"),
            "policy": self._translate_habitat_policy(payload, warnings, unmapped),
            "output": self._translate_habitat_output(payload),
            "legacy": {},
        }

        # Preserve every unrecognised top-level key verbatim.
        for key, value in payload.items():
            if key not in _HABITAT_CONSUMED_TOP_KEYS:
                unmapped[key] = value
                warnings.append(
                    f"Top-level key '{key}' has no v1 slot; preserved under 'legacy'."
                )

        document["legacy"] = unmapped
        return LegacyTranslation(
            document=document,
            workflow="habitat",
            unmapped=unmapped,
            warnings=tuple(warnings),
        )

    def _translate_habitat_spec(
        self,
        payload: Mapping[str, Any],
        feature_construction: Mapping[str, Any],
        segmentation: Mapping[str, Any],
        clustering_mode: str,
        warnings: List[str],
        unmapped: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Build the ``spec`` section of a habitat document.

        Predict-mode configs legitimately omit ``feature_construction`` --
        the fitted :class:`HabitatModel` carries the definition -- so the
        section is ``None`` there, which the v1 reader treats as
        "definition comes from the pipeline artefact".

        Args:
            payload: Full v0 payload (for the global seed).
            feature_construction: v0 ``feature_construction`` block.
            segmentation: v0 ``habitat_segmentation`` block.
            clustering_mode: ``two_step`` / ``one_step`` / ``direct_pooling``.
            warnings: Warning sink.
            unmapped: Legacy sink for per-component seeds.

        Returns:
            A ``HabitatSpec.to_dict()`` payload, or ``None`` for predict
            configs without a feature-construction block.
        """
        if not feature_construction:
            if clustering_mode and clustering_mode != "two_step":
                # A predict stub still declares its assembly strategy; keep it
                # visible so the reader can pick the right recipe family.
                unmapped.setdefault("habitat_segmentation", {})[
                    "clustering_mode"
                ] = clustering_mode
            return None

        voxel_extractor = self._translate_voxel_extractor(
            feature_construction.get("voxel_level") or {}, warnings
        )
        supervoxelizer = self._translate_supervoxelizer(
            feature_construction, segmentation, clustering_mode, warnings, unmapped
        )
        supervoxel_features = self._translate_supervoxel_feature_extractor(
            feature_construction, segmentation, clustering_mode, warnings, unmapped
        )
        fitter = self._translate_habitat_fitter(
            segmentation, clustering_mode, warnings, unmapped
        )

        spec_payload: Dict[str, Any] = {
            "name": f"habitat_{clustering_mode}",
            "version": self.SCHEMA_VERSION,
            "voxel_feature_extractor": voxel_extractor,
            "supervoxelizer": supervoxelizer,
            "supervoxel_feature_extractor": supervoxel_features,
            "habitat_model_fitter": fitter,
            # v0.1 always assigns by nearest centroid; it never had a knob.
            "habitat_assigner": {"name": "nearest_centroid", "params": {}},
            "habitat_features": [],
            # v0.1's two blocks map onto the stateless voxel chain and the
            # stateful cohort chain. The third v1 chain (stateless, on
            # supervoxel features) has no v0.1 source: that version could only
            # preprocess supervoxel features at cohort level.
            "voxel_feature_preprocessors": self._translate_preprocessing_chain(
                feature_construction.get("preprocessing_for_subject_level"),
                warnings,
                unmapped,
            ),
            "supervoxel_feature_preprocessors": [],
            "cohort_feature_preprocessors": self._translate_cohort_chain(
                feature_construction, clustering_mode, warnings, unmapped
            ),
            "random_seed": payload.get("random_state"),
            # The v0 ``clustering_mode`` knob becomes an explicit dataflow
            # declaration: one_step never pools across subjects, the other
            # two designs fit one cohort-level definition. ``"cohort"`` is
            # fingerprint-neutral (``to_dict`` omits it, matching an
            # undeclared dataflow); ``"none"`` is recorded, which is what
            # moves the one-step spec fingerprint forward.
            "pooling": "none" if clustering_mode == "one_step" else "cohort",
            "definition_level": (
                "subject" if clustering_mode == "one_step" else "cohort"
            ),
        }
        # Optional top-level key; omit when unset so the HabitatSpec default
        # (resample_mask) applies without changing historical fingerprints.
        if payload.get("on_geometry_mismatch") is not None:
            spec_payload["on_geometry_mismatch"] = payload.get("on_geometry_mismatch")
        for slot, block_name in (
            ("postprocess_supervoxel", "postprocess_supervoxel"),
            ("postprocess_habitat", "postprocess_habitat"),
        ):
            translated = self._translate_connected_component_postprocess(
                segmentation.get(block_name),
                block_name=block_name,
                clustering_mode=clustering_mode,
                warnings=warnings,
            )
            if translated is not None:
                spec_payload[slot] = translated
        return spec_payload

    def _translate_connected_component_postprocess(
        self,
        block: Any,
        *,
        block_name: str,
        clustering_mode: str,
        warnings: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Translate a v0 connected-component postprocess block into a Spec.

        ``enabled: false`` / missing / empty become ``None`` (slot omitted).
        ``enabled: true`` becomes ``Spec(name="connected_components", ...)``.

        Args:
            block: v0 ``postprocess_supervoxel`` or ``postprocess_habitat``
                mapping, or ``None``.
            block_name: YAML key name, for warnings.
            clustering_mode: Assembly strategy; supervoxel cleanup is ignored
                outside ``two_step``.
            warnings: Warning sink.

        Returns:
            A ``Spec.to_dict()`` payload, or ``None`` when cleanup is off.
        """
        if not isinstance(block, Mapping):
            return None
        if not bool(block.get("enabled", False)):
            return None
        if (
            block_name == "postprocess_supervoxel"
            and clustering_mode != "two_step"
        ):
            warnings.append(
                f"habitat_segmentation.{block_name} is enabled but "
                f"clustering_mode '{clustering_mode}' has no supervoxel "
                "stage; the setting is ignored (habitat cleanup still applies "
                "via postprocess_habitat when enabled)."
            )
            return None
        params: Dict[str, Any] = {
            "min_component_size": int(block.get("min_component_size", 30)),
            "connectivity": int(block.get("connectivity", 1)),
            "reassign_method": str(block.get("reassign_method", "neighbor_vote")),
            "max_iterations": int(block.get("max_iterations", 3)),
        }
        return {"name": "connected_components", "params": params}

    def _translate_voxel_extractor(
        self, voxel_level: Mapping[str, Any], warnings: List[str]
    ) -> Dict[str, Any]:
        """
        Translate the v0 ``voxel_level`` expression into an extractor Spec.

        The v0 mini-language (``concat(raw(A), raw(B))``,
        ``voxel_radiomics(A)``, ``kinetic(...)``) collapses into a v1
        component name plus a ``modalities`` list whenever the expression is
        homogeneous; anything more complex keeps its original expression
        under ``params.expression`` so no information is lost.

        Args:
            voxel_level: v0 ``voxel_level`` block with ``method``/``params``.
            warnings: Warning sink.

        Returns:
            A ``Spec.to_dict()`` payload for domain ``voxel_feature_extractor``.
        """
        expression = str(voxel_level.get("method", "")).strip()
        v0_params = dict(voxel_level.get("params") or {})
        if not expression:
            raise HABITAPIError(
                "feature_construction.voxel_level.method is required in a "
                "habitat train config."
            )

        if '"' in expression or "'" in expression:
            # The v1 strict spelling (quoted modality names, key=value
            # params, nested children) parses into the canonical Spec
            # tree. The block's params mapping merges into the ROOT node,
            # with inline kwargs winning; unquoted expressions keep the
            # legacy translation below byte-identically.
            tree = parse_feature_expression(expression)
            return {
                "name": tree.name,
                "params": {**v0_params, **dict(tree.params)},
            }

        outer, steps = _parse_method_expression(
            expression, frozenset(v0_params)
        )
        inner_methods = {step[0] for step in steps}
        modalities = [step[1] for step in steps]

        if outer == "concat" and len(inner_methods) == 1:
            # Homogeneous concat: the inner method IS the extractor.
            name = inner_methods.pop()
            params: Dict[str, Any] = {"modalities": modalities, **v0_params}
        elif outer != "concat" and not steps:
            # Bare method name without parentheses, e.g. "raw".
            name = outer
            params = {**v0_params}
        elif outer != "concat" and inner_methods == {outer}:
            # Single-method form, e.g. ``voxel_radiomics(delay2)``: every
            # step belongs to the outer method itself.
            name = outer
            params = {"modalities": modalities, **v0_params}
        else:
            # Heterogeneous or nested composition (kinetic, mixed concat):
            # keep the outer method as the component and the full expression
            # verbatim so the future v1 component can replay it exactly.
            name = outer
            params = {"modalities": modalities, **v0_params}
            params["expression"] = expression
            warnings.append(
                f"voxel_level method '{expression}' is a composite expression; "
                "kept verbatim under spec params 'expression'."
            )
        return {"name": name, "params": params}

    def _translate_supervoxelizer(
        self,
        feature_construction: Mapping[str, Any],
        segmentation: Mapping[str, Any],
        clustering_mode: str,
        warnings: List[str],
        unmapped: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Translate ``habitat_segmentation.supervoxel`` into a Spec.

        This block answers ONE question -- how the supervoxel labels are
        grown -- and maps to the ``supervoxelizer`` domain alone. How those
        supervoxels are then described is the separate
        ``feature_construction.supervoxel_level`` block, translated by
        :meth:`_translate_supervoxel_feature_extractor`. Fusing the two (as
        an earlier version of this adapter did) loses one of two orthogonal
        choices and produces spec names no registry can instantiate.

        ``one_step``/``direct_pooling`` have no supervoxel stage and yield
        ``None``.

        Args:
            feature_construction: v0 ``feature_construction`` block. Unused
                here; kept so both supervoxel translators share a signature.
            segmentation: v0 ``habitat_segmentation`` block.
            clustering_mode: Assembly strategy.
            warnings: Warning sink.
            unmapped: Legacy sink for the per-block seed.

        Returns:
            A ``Spec.to_dict()`` payload, or ``None``.
        """
        if clustering_mode != "two_step":
            return None

        supervoxel_block = segmentation.get("supervoxel") or {}
        algorithm = str(supervoxel_block.get("algorithm", "kmeans")).strip()
        seed = supervoxel_block.get("random_state")
        if seed is not None:
            unmapped.setdefault("habitat_segmentation", {}).setdefault(
                "supervoxel", {}
            )["random_state"] = seed
            warnings.append(
                "Per-component seed supervoxel.random_state has no v1 slot; "
                "v1 applies HabitatSpec.random_seed to every Seedable. The "
                "original value is preserved under 'legacy'."
            )

        params: Dict[str, Any] = {
            "n_supervoxels": supervoxel_block.get("n_clusters", 50),
        }
        if algorithm == "slic":
            params.update(
                {
                    "compactness": supervoxel_block.get("compactness", 0.1),
                    "enforce_connectivity": supervoxel_block.get(
                        "enforce_connectivity", True
                    ),
                }
            )
            # v0 forwarded ``max_iter``/``sigma`` to skimage's slic verbatim
            # (``max_iter`` maps to ``max_num_iter``). Keep them alive through
            # the reserved passthrough key instead of dropping them silently.
            # ``max_num_iter`` is always emitted because the v0 schema
            # materialised its default (300), which differs from the skimage
            # default; ``sigma`` is omitted at its skimage default (0.0).
            passthrough: Dict[str, Any] = {
                "max_num_iter": int(supervoxel_block.get("max_iter", 300)),
            }
            sigma = float(supervoxel_block.get("sigma", 0.0))
            if sigma != 0.0:
                passthrough["sigma"] = sigma
            params["estimator_params"] = passthrough
        else:
            params.update(
                {
                    "max_iter": supervoxel_block.get("max_iter", 300),
                    "n_init": supervoxel_block.get("n_init", 10),
                }
            )
        return {"name": algorithm, "params": params}

    def _translate_supervoxel_feature_extractor(
        self,
        feature_construction: Mapping[str, Any],
        segmentation: Mapping[str, Any],
        clustering_mode: str,
        warnings: List[str],
        unmapped: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Translate ``feature_construction.supervoxel_level`` into a Spec.

        ``mean_voxel_features()`` -- the v0 default -- maps to ``None``:
        every v1 supervoxelizer already attaches feature means, so naming the
        default would add a redundant step rather than describe one.
        ``supervoxel_radiomics(...)``, bare or wrapped in a homogeneous
        ``concat()``, maps to the extractor of the same name.

        v0.1 ignores this block outside ``two_step`` (the one-step design
        clusters voxels straight into habitats, leaving no supervoxel to
        describe); the same is done here, with the same warning.

        Args:
            feature_construction: v0 ``feature_construction`` block.
            segmentation: v0 ``habitat_segmentation`` block. Unused here;
                kept so both supervoxel translators share a signature.
            clustering_mode: Assembly strategy.
            warnings: Warning sink.
            unmapped: Legacy sink for expressions with no v1 component.

        Returns:
            A ``Spec.to_dict()`` payload, or ``None`` for the default.
        """
        supervoxel_level = feature_construction.get("supervoxel_level") or {}
        expression = str(
            supervoxel_level.get("method", "mean_voxel_features()")
        ).strip()
        v0_params = dict(supervoxel_level.get("params") or {})
        # ``supervoxel_file_keyword`` locates a previously written label map:
        # a data-source concern in v1, never an algorithm parameter.
        v0_params.pop("supervoxel_file_keyword", None)

        if '"' in expression or "'" in expression:
            # The v1 strict spelling (quoted modality names, key=value
            # params, nested children) parses into the canonical Spec
            # tree; unquoted expressions keep the legacy translation
            # below byte-identically.
            if clustering_mode != "two_step":
                warnings.append(
                    f"clustering_mode '{clustering_mode}' ignores "
                    f"feature_construction.supervoxel_level.method "
                    f"('{expression}'): there is no supervoxel stage to "
                    "describe. This matches v0.1 behaviour."
                )
                return None
            tree = parse_feature_expression(expression)
            return {
                "name": tree.name,
                "params": {**v0_params, **dict(tree.params)},
            }

        outer, steps = _parse_method_expression(
            expression, frozenset(supervoxel_level.get("params") or {})
        )
        inner_methods = {step[0] for step in steps}
        is_default = outer in ("mean_voxel_features", "")
        is_radiomics = outer == "supervoxel_radiomics" or (
            outer == "concat" and inner_methods == {"supervoxel_radiomics"}
        )

        if clustering_mode != "two_step":
            if not is_default:
                warnings.append(
                    f"clustering_mode '{clustering_mode}' ignores "
                    f"feature_construction.supervoxel_level.method "
                    f"('{expression}'): there is no supervoxel stage to "
                    "describe. This matches v0.1 behaviour."
                )
            return None

        if is_default:
            return None

        if is_radiomics:
            return {
                "name": "supervoxel_radiomics",
                "params": {
                    "modalities": [step[1] for step in steps],
                    **v0_params,
                },
            }

        unmapped.setdefault("feature_construction", {}).setdefault(
            "supervoxel_level", {}
        )["method"] = expression
        warnings.append(
            f"supervoxel_level method '{expression}' has no v1 supervoxel "
            "feature extractor; the expression is preserved under 'legacy'."
        )
        return None

    def _translate_habitat_fitter(
        self,
        segmentation: Mapping[str, Any],
        clustering_mode: str,
        warnings: List[str],
        unmapped: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Translate the clustering blocks into a ``habitat_model_fitter`` Spec.

        ``two_step``/``direct_pooling`` fit at cohort level from the
        ``habitat`` block; ``one_step`` fits per subject from
        ``supervoxel.algorithm`` plus ``one_step_settings``. v0 allows a LIST
        of selection criteria where v1 keeps one ``validation`` string -- the
        first entry wins and the full list rides along as
        ``selection_methods``.

        Args:
            segmentation: v0 ``habitat_segmentation`` block.
            clustering_mode: Assembly strategy.
            warnings: Warning sink.
            unmapped: Legacy sink for seeds and search-parallelism knobs.

        Returns:
            A ``Spec.to_dict()`` payload for domain ``habitat_model_fitter``.
        """
        habitat_block = segmentation.get("habitat") or {}
        supervoxel_block = segmentation.get("supervoxel") or {}

        for block_name, block in (("habitat", habitat_block),):
            seed = block.get("random_state")
            if seed is not None:
                unmapped.setdefault("habitat_segmentation", {}).setdefault(
                    block_name, {}
                )["random_state"] = seed
                warnings.append(
                    f"Per-component seed {block_name}.random_state has no v1 "
                    "slot; preserved under 'legacy'."
                )

        if clustering_mode == "one_step":
            settings = supervoxel_block.get("one_step_settings") or {}
            name = str(supervoxel_block.get("algorithm", "kmeans")).strip()
            params: Dict[str, Any] = {
                "n_habitats": settings.get("fixed_n_clusters"),
                "min_habitats": settings.get("min_clusters", 2),
                "max_habitats": settings.get("max_clusters", 10),
                "validation": settings.get("selection_method", "elbow"),
                "max_iter": supervoxel_block.get("max_iter", 300),
                "n_init": supervoxel_block.get("n_init", 10),
            }
            if settings.get("plot_validation_curves") is not None:
                unmapped.setdefault("habitat_segmentation", {}).setdefault(
                    "supervoxel", {}
                ).setdefault("one_step_settings", {})[
                    "plot_validation_curves"
                ] = settings.get("plot_validation_curves")
            return {"name": name, "params": params}

        name = str(habitat_block.get("algorithm", "kmeans")).strip()
        selection = habitat_block.get("habitat_cluster_selection_method", "elbow")
        methods = [selection] if isinstance(selection, str) else list(selection)
        params = {
            "n_habitats": habitat_block.get("fixed_n_clusters"),
            "min_habitats": habitat_block.get("min_clusters", 2),
            "max_habitats": habitat_block.get("max_clusters", 10),
            # The v1 fitters accept a criterion list and vote, matching the
            # v0 engine, which kept every valid method. A single method stays
            # a scalar so specs (and fingerprints) equal pre-voting runs.
            "validation": (
                methods[0] if len(methods) == 1 else methods
            ) if methods else "elbow",
            "max_iter": habitat_block.get("max_iter", 300),
            "n_init": habitat_block.get("n_init", 10),
        }
        for key in ("parallel_cluster_search", "cluster_search_workers"):
            if key in habitat_block:
                unmapped.setdefault("habitat_segmentation", {}).setdefault(
                    "habitat", {}
                )[key] = habitat_block[key]
        return {"name": name, "params": params}

    def _translate_preprocessing_chain(
        self,
        block: Optional[Mapping[str, Any]],
        warnings: List[str],
        unmapped: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Translate one v0 preprocessing block into method Specs.

        The v0 ``method`` field becomes the Spec name; every remaining key
        (``winsor_limits``, ``n_bins``, ...) is a Spec param, so the YAML
        order -- which is scientifically meaningful -- is preserved exactly.

        ``global_normalize`` is renamed to ``across_features``. The old name
        read as "use cohort-wide statistics", which it never meant: the flag
        selects whether statistics are pooled across feature COLUMNS. Whose
        rows they are pooled over is decided by which chain holds the method,
        so keeping the old name next to the new chain names would actively
        mislead. The value carries over unchanged -- but only for the methods
        that consumed it in v0.1 (:data:`_V0_SCOPE_AWARE_METHODS`); the
        filters ignored the key, so for them it lands in the legacy sink
        rather than in a params mapping the v1 component does not declare.

        Args:
            block: v0 ``preprocessing_for_*_level`` block or ``None``.
            warnings: Warning sink.
            unmapped: Legacy sink for scope flags a method never consumed.

        Returns:
            Ordered list of ``Spec.to_dict()`` payloads.
        """
        if not block:
            return []
        chain: List[Dict[str, Any]] = []
        for entry in block.get("methods") or ():
            entry = dict(entry)
            name = entry.pop(_PREPROCESSING_NAME_KEY, None)
            if not name:
                raise HABITAPIError(
                    "Every preprocessing method entry needs a 'method' key; "
                    f"got {entry!r}."
                )
            if _V0_SCOPE_KEY in entry:
                scope_value = bool(entry.pop(_V0_SCOPE_KEY))
                if str(name) in _V0_SCOPE_AWARE_METHODS:
                    entry[_V1_SCOPE_KEY] = scope_value
                else:
                    unmapped.setdefault("preprocessing_scope_flags", {})[
                        str(name)
                    ] = scope_value
                    warnings.append(
                        f"Preprocessing method {name!r} accepted but never "
                        f"consumed '{_V0_SCOPE_KEY}' in v0.1; the value "
                        "is preserved under 'legacy' and not forwarded."
                    )
            # The v0 ``PreprocessingMethod`` schema is a single wide model:
            # when the CLI feeds a ``model_dump()`` payload, every knob the
            # user never wrote materialises as an explicit None (e.g. a
            # ``winsorize`` entry arrives carrying ``n_bins: None``). The
            # v0.1 engine treated None as "not set" (handlers guard with
            # ``if value is not None``), so stripping it here is behaviour-
            # preserving -- and required, since the v1 domain params are
            # ``extra="forbid"`` and would reject the foreign None keys.
            params = {
                key: value for key, value in entry.items() if value is not None
            }
            chain.append({"name": str(name), "params": params})
        return chain

    def _translate_cohort_chain(
        self,
        feature_construction: Mapping[str, Any],
        clustering_mode: str,
        warnings: List[str],
        unmapped: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Translate ``preprocessing_for_group_level`` for the given mode.

        v0.1 wired this block into the ``two_step`` and ``direct_pooling``
        pipelines only: ``_build_one_step_steps`` has no group-preprocessing
        step at all, because one-step defines habitats inside each subject
        and nothing is ever pooled. Translating it into a cohort chain
        regardless of mode would state that a step ran which v0.1 never ran
        -- and the v1 recipe, taking the spec at its word, WOULD run it and
        return different habitats.

        Args:
            feature_construction: v0 ``feature_construction`` block.
            clustering_mode: ``two_step`` / ``one_step`` / ``direct_pooling``.
            warnings: Warning sink.
            unmapped: Legacy sink for the dropped block.

        Returns:
            Ordered method Specs, empty for ``one_step``.
        """
        block = feature_construction.get("preprocessing_for_group_level")
        if clustering_mode != "one_step":
            return self._translate_preprocessing_chain(block, warnings, unmapped)
        if block and (block.get("methods") or ()):
            unmapped.setdefault("feature_construction", {})[
                "preprocessing_for_group_level"
            ] = block
            warnings.append(
                "feature_construction.preprocessing_for_group_level is ignored "
                "in clustering_mode 'one_step' (v0.1 never applied it there); "
                "preserved verbatim under 'legacy'."
            )
        return []

    def _translate_habitat_policy(
        self,
        payload: Mapping[str, Any],
        warnings: List[str],
        unmapped: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Translate v0 habitat execution keys into a ``RunPolicy`` payload.

        Keys are renamed onto the v1 surface. ``workers > 1`` **or** a
        positive ``subject_timeout_sec`` selects the ``process`` backend,
        matching v0.1 ``_should_use_spawn_workers`` (timeout isolation needs
        a child even when ``workers == 1``).

        Args:
            payload: Full v0 payload.
            warnings: Warning sink.
            unmapped: Legacy sink.

        Returns:
            A ``RunPolicy.to_dict()``-shaped payload (omitted keys stay at
            v1 defaults).
        """
        policy: Dict[str, Any] = {}
        for v0_key, v1_key in _HABITAT_POLICY_KEY_MAP.items():
            if v0_key in payload:
                policy[v1_key] = payload[v0_key]
        workers = policy.get("workers", 1)
        # Prefer an explicit timeout from the payload; when the key is absent
        # the typed RunPolicy default (900s) still applies after from_dict, so
        # treat "absent" like the v0.1 schema default of 900.
        if "subject_timeout_sec" in policy:
            timeout = policy["subject_timeout_sec"]
        else:
            timeout = 900.0
        timeout_armed = timeout is not None and float(timeout) > 0
        workers_parallel = isinstance(workers, int) and workers > 1
        policy["backend"] = "process" if (workers_parallel or timeout_armed) else "serial"
        return policy

    def _translate_habitat_output(
        self, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """
        Collect the habitat output switches into the ``output`` section.

        Args:
            payload: Full v0 payload.

        Returns:
            ``out_dir`` plus every output switch present in the source.
        """
        output: Dict[str, Any] = {"out_dir": payload.get("out_dir")}
        for key in _HABITAT_OUTPUT_KEYS:
            if key in payload:
                output[key] = payload[key]
        return output

    # ------------------------------------------------------------------
    # Machine-learning workflows (model/cv): deep translation into MLSpec
    # ------------------------------------------------------------------

    def _translate_ml(
        self, payload: Mapping[str, Any], workflow: str
    ) -> LegacyTranslation:
        """
        Deep-translate a v0 ``MLConfig`` mapping (``model`` / ``cv``).

        The algorithmic core lands in typed :class:`MLSpec` slots:
        ``normalization`` becomes the table-preprocessing chain,
        ``feature_selection_methods`` the selection chain, and the FIRST
        ``models`` entry the classifier. What the v1 spec deliberately has
        no slot for -- the validation design (``n_splits``, ``test_size``,
        id files), class-imbalance resampling, bootstrap CIs, multi-model
        sweeps, per-component seeds -- is preserved verbatim under
        ``legacy`` with a warning each, so the translation stays lossless.

        Args:
            payload: v0 ML YAML mapping.
            workflow: ``"model"`` or ``"cv"``.

        Returns:
            The translation result.
        """
        warnings: List[str] = []
        unmapped: Dict[str, Any] = {}

        mode = str(payload.get("run_mode", "train")).strip().lower()
        document: Dict[str, Any] = {
            "version": self.SCHEMA_VERSION,
            "workflow": workflow,
            "mode": mode,
            "spec": self._translate_ml_spec(payload, workflow, warnings, unmapped),
            "data": self._translate_ml_data(payload),
            "pipeline": payload.get("pipeline_path") or payload.get("model_path"),
            "policy": self._translate_ml_policy(payload),
            "output": self._translate_ml_output(payload),
            "legacy": {},
        }

        preserved_validation = {
            key: payload[key] for key in _ML_VALIDATION_KEYS if key in payload
        }
        if preserved_validation:
            unmapped.update(preserved_validation)
            warnings.append(
                "Validation-design keys "
                f"{sorted(preserved_validation)} shape the split, not the "
                "model definition; the v1 MLSpec has no slot for them (the "
                "calling recipe owns the validation design). Preserved "
                "verbatim under 'legacy'."
            )

        for key in ("resampling", "sampling"):
            block = payload.get(key)
            if block:
                unmapped[key] = block
                warnings.append(
                    f"'{key}' (class-imbalance resampling) has no v1 slot "
                    "yet; preserved verbatim under 'legacy'."
                )
        for key in ("bootstrap", "strict_model_params"):
            if key in payload and payload[key]:
                unmapped[key] = payload[key]
                warnings.append(
                    f"'{key}' has no v1 slot yet; preserved under 'legacy'."
                )

        # Preserve every unrecognised top-level key verbatim.
        for key, value in payload.items():
            if key not in _ML_CONSUMED_TOP_KEYS:
                unmapped[key] = value
                warnings.append(
                    f"Top-level key '{key}' has no v1 slot; preserved under 'legacy'."
                )

        document["legacy"] = unmapped
        return LegacyTranslation(
            document=document,
            workflow=workflow,
            unmapped=unmapped,
            warnings=tuple(warnings),
        )

    def _translate_ml_spec(
        self,
        payload: Mapping[str, Any],
        workflow: str,
        warnings: List[str],
        unmapped: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Build the ``spec`` section of a model/cv document.

        Predict-mode configs legitimately omit ``models`` -- the fitted
        pipeline artefact carries the definition -- so the section is
        ``None`` there, which the v1 reader treats as "definition comes
        from the pipeline artefact".

        Args:
            payload: Full v0 payload (for the global seed).
            workflow: ``"model"`` or ``"cv"`` (names the spec).
            warnings: Warning sink.
            unmapped: Legacy sink.

        Returns:
            An ``MLSpec.to_dict()`` payload, or ``None`` for predict
            configs without a ``models`` block.
        """
        models = payload.get("models") or {}
        if not models:
            warnings.append(
                "No 'models' block: the spec section is None. In predict "
                "mode the model definition comes from the pipeline artefact; "
                "a train-mode v0 config without models is rejected by the "
                "v0 schema itself."
            )
            return None
        pre_selectors, post_selectors = self._translate_ml_selectors(
            payload.get("feature_selection_methods"), warnings, unmapped
        )
        return {
            "name": f"ml_{workflow}",
            "version": self.SCHEMA_VERSION,
            # Key order mirrors execution order: pre-preprocessing selection,
            # preprocessing, post-preprocessing selection.
            #
            # These are the DEPRECATED MLSpec chain keys, and the translation
            # keeps emitting them on purpose. ``MLSpec`` folds them into its
            # ordered ``steps`` list in exactly this order, so ``before_z_score``
            # does now decide a POSITION rather than a bucket -- but the
            # translated document itself stays byte-identical, and so does the
            # MLSpec fingerprint every published v0.1 result was recorded
            # under. Emitting ``steps`` here instead would move that
            # fingerprint for every legacy config in existence, which the
            # golden baselines correctly treat as a regression.
            "pre_preprocessing_feature_selectors": pre_selectors,
            "table_preprocessors": self._translate_ml_normalization(
                payload.get("normalization"), warnings, unmapped
            ),
            "feature_selectors": post_selectors,
            "classifier": self._translate_ml_classifier(models, warnings, unmapped),
            # v0.1 reported a fixed metric panel chosen by the workflow, not
            # by the config; the v1 default panel lives in the recipe.
            "metrics": [],
            "random_seed": payload.get("random_state"),
        }

    def _translate_ml_normalization(
        self,
        normalization: Optional[Mapping[str, Any]],
        warnings: List[str],
        unmapped: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Translate the v0 ``normalization`` block into a one-step chain.

        Args:
            normalization: v0 ``normalization`` block or ``None``.
            warnings: Warning sink.
            unmapped: Legacy sink for methods with no v1 component.

        Returns:
            Zero or one ``Spec.to_dict()`` payloads.
        """
        block = normalization or {}
        method = block.get("method")
        if not method:
            return []
        name = _ML_NORMALIZATION_NAME_MAP.get(str(method).strip())
        if name is None:
            unmapped["normalization"] = block
            warnings.append(
                f"normalization.method {method!r} has no v1 table "
                "preprocessor; preserved verbatim under 'legacy'."
            )
            return []
        return [{"name": name, "params": dict(block.get("params") or {})}]

    def _translate_ml_selectors(
        self,
        methods: Optional[Sequence[Mapping[str, Any]]],
        warnings: List[str],
        unmapped: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Translate v0 ``feature_selection_methods`` into staged selector Specs.

        Each entry lands in one of two chains: the PRE-preprocessing chain
        (fitted on the raw training table) or the post-preprocessing chain.
        The stage rule reproduces v0's two-stage pipeline exactly: an
        explicit ``before_z_score`` param wins; when it is absent the v0
        registry default applies (``_V0_SELECTOR_DEFAULT_BEFORE_Z_SCORE`` --
        the gap the old translation had, silently reordering selectors like
        ``variance`` whose default is True). The YAML order -- which is
        scientifically meaningful -- is preserved WITHIN each stage; v0
        itself grouped execution by stage, so the split loses no ordering
        information. ``before_z_score`` leaves the params because the stage
        is a chain-level fact in v1, not an algorithm parameter; the
        ``images`` runtime key and per-component ``random_state`` seeds are
        stripped as before (v1 seeds every Seedable from
        ``MLSpec.random_seed``).

        Args:
            methods: v0 ``feature_selection_methods`` list or ``None``.
            warnings: Warning sink.
            unmapped: Legacy sink for stripped per-component seeds.

        Returns:
            Tuple ``(pre_preprocessing, post_preprocessing)`` of ordered
            ``Spec.to_dict()`` payload lists.
        """
        pre_selectors: List[Dict[str, Any]] = []
        post_selectors: List[Dict[str, Any]] = []
        seeds: Dict[str, Any] = {}
        for entry in methods or ():
            entry = dict(entry)
            name = entry.get("method")
            if not name:
                raise HABITAPIError(
                    "Every feature_selection_methods entry needs a 'method' "
                    f"key; got {entry!r}."
                )
            v0_name = str(name)
            params = dict(entry.get("params") or {})
            before_z_score = params.pop("before_z_score", None)
            params.pop("images", None)
            seed = params.pop("random_state", None)
            if seed is not None:
                seeds[v0_name] = seed
            plot_flags = {
                key: params.pop(key)
                for key in _V0_SELECTOR_PLOT_KEYS
                if key in params
            }
            if plot_flags:
                unmapped.setdefault("feature_selection_plot_flags", {})[
                    v0_name
                ] = plot_flags
                warnings.append(
                    f"Selector {v0_name!r} plotting switch(es) "
                    f"{sorted(plot_flags)} have no v1 params slot; v1 keeps "
                    "visualisation in the reporting layer. The values are "
                    "preserved under 'legacy'."
                )
            if v0_name == "icc":
                # v0.1's ``icc`` selector ONLY read a precomputed results
                # JSON; the v1 registry names that contract ``icc_precomputed``
                # (``icc`` is the repeat-tables recomputing selector).
                name, params = self._translate_icc_selector(params)
            if before_z_score is None:
                run_pre = _V0_SELECTOR_DEFAULT_BEFORE_Z_SCORE.get(v0_name, False)
            else:
                run_pre = bool(before_z_score)
            target = pre_selectors if run_pre else post_selectors
            target.append({"name": str(name), "params": params})
        if seeds:
            unmapped.setdefault("feature_selection_seeds", {}).update(seeds)
            warnings.append(
                "Per-component seed(s) in feature_selection_methods params "
                "(random_state) have no v1 slot; v1 applies "
                "MLSpec.random_seed to every Seedable. The original values "
                "are preserved under 'legacy'."
            )
        return pre_selectors, post_selectors

    @staticmethod
    def _translate_icc_selector(
        params: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Map a v0.1 ``icc`` selector block onto the v1 registry name.

        With an ``icc_results``/``icc_results_path`` in the params the v0.1
        contract is the precomputed-JSON selector, which the v1 registry
        carries as ``icc_precomputed``; the aliases collapse into
        ``icc_results_path``/``groups``. Without a results path the block
        stays ``icc`` -- the v1 repeat-tables selector -- which fails loudly
        at fit time exactly as the v0.1 selector failed on a missing path.

        Args:
            params: The selector's v0 params (runtime keys and seeds already
                stripped).

        Returns:
            ``(registry_name, params)`` for the v1 Spec payload.
        """
        results_path = params.pop("icc_results", None) or params.pop(
            "icc_results_path", None
        )
        params.pop("icc_results", None)
        params.pop("icc_results_path", None)
        if results_path is None:
            return "icc", params
        groups = params.pop("groups", None) or params.pop("keys", None)
        params.pop("groups", None)
        params.pop("keys", None)
        translated: Dict[str, Any] = {"icc_results_path": str(results_path)}
        if groups:
            translated["groups"] = [str(group) for group in groups]
        translated.update(params)
        return "icc_precomputed", translated

    def _translate_ml_classifier(
        self,
        models: Mapping[str, Any],
        warnings: List[str],
        unmapped: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Translate the v0 ``models`` mapping into ONE classifier Spec.

        v0 trained every entry of the mapping in one run; the v1 MLSpec
        describes ONE modelling definition, so the first entry (in YAML
        order) wins and the full block -- per-component seeds included --
        is preserved verbatim under ``legacy``. For a single-entry mapping
        only a stripped ``random_state`` param is preserved, mirroring the
        habitat translator's per-component seed rule.

        Args:
            models: v0 ``models`` mapping (non-empty).
            warnings: Warning sink.
            unmapped: Legacy sink.

        Returns:
            A ``Spec.to_dict()`` payload for domain ``classifier``.
        """
        entries = list(models.items())
        first_name, first_config = entries[0]
        params = dict((first_config or {}).get("params") or {})
        seed = params.pop("random_state", None)
        if len(entries) > 1:
            unmapped["models"] = models
            warnings.append(
                f"v0 trained {len(entries)} models in one run; the v1 "
                f"MLSpec describes ONE classifier -- the first entry "
                f"({first_name!r}) wins and the full block is preserved "
                "verbatim under 'legacy'."
            )
        elif seed is not None:
            unmapped.setdefault("models", {})[first_name] = {
                "params": {"random_state": seed}
            }
            warnings.append(
                f"Per-component seed models.{first_name}.params.random_state "
                "has no v1 slot; v1 applies MLSpec.random_seed to every "
                "Seedable. The original value is preserved under 'legacy'."
            )
        return {"name": str(first_name), "params": params}

    def _translate_ml_data(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Collect the ML data keys into the ``data`` section.

        The v0 ``input`` list (per-table path / id / label descriptors) is
        kept verbatim: it IS the data-source description.

        Args:
            payload: Full v0 payload.

        Returns:
            The ``data`` section payload.
        """
        data: Dict[str, Any] = {}
        if "input" in payload:
            data["input"] = payload["input"]
        for key in _GENERIC_DATA_KEYS:
            if key in payload:
                data[key] = payload[key]
        return data

    def _translate_ml_policy(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Collect the ML execution keys into a ``RunPolicy`` payload.

        Args:
            payload: Full v0 payload.

        Returns:
            A ``RunPolicy.to_dict()``-shaped payload (empty when the v0
            config sets no execution keys).
        """
        policy: Dict[str, Any] = {}
        for v0_key, v1_key in _GENERIC_POLICY_KEY_MAP.items():
            if v0_key in payload:
                policy[v1_key] = payload[v0_key]
        if policy:
            workers = policy.get("workers", 1)
            policy["backend"] = (
                "process" if isinstance(workers, int) and workers > 1 else "serial"
            )
        return policy

    def _translate_ml_output(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Collect the ML output keys into the ``output`` section.

        The v0 ``output`` path string becomes ``out_dir``; reporting
        switches and the predict-mode output schema ride alongside.

        Args:
            payload: Full v0 payload.

        Returns:
            The ``output`` section payload.
        """
        output: Dict[str, Any] = {"out_dir": payload.get("output")}
        for key in _GENERIC_OUTPUT_KEYS:
            if key in payload:
                output[key] = payload[key]
        for key in (*_ML_OUTPUT_SWITCH_KEYS, *_ML_PREDICT_OUTPUT_KEYS):
            if key in payload:
                output[key] = payload[key]
        return output

    # ------------------------------------------------------------------
    # Generic workflows: section split with the algorithm kept verbatim
    # ------------------------------------------------------------------

    def _translate_generic(
        self, payload: Mapping[str, Any], workflow: str
    ) -> LegacyTranslation:
        """
        Translate a non-habitat v0 workflow by section classification.

        The v1 spec models for these workflows are designed in a later
        phase, so the translation is deliberately structural: data keys form
        the ``data`` section, execution keys the ``policy`` section, output
        keys the ``output`` section, and the remaining algorithmic mapping
        rides verbatim as ``spec.params`` under a workflow-named Spec. No
        key is dropped, so the future deep translation can replay the v0
        file exactly from the v1 document alone.

        Args:
            payload: v0 workflow YAML mapping.
            workflow: Workflow alias.

        Returns:
            The translation result.
        """
        warnings: List[str] = []
        data: Dict[str, Any] = {}
        policy: Dict[str, Any] = {}
        output: Dict[str, Any] = {}
        spec_params: Dict[str, Any] = {}
        mode: Optional[str] = None
        pipeline: Optional[str] = None

        for key, value in payload.items():
            if key in _GENERIC_MODE_KEYS:
                slot = _GENERIC_MODE_KEYS[key]
                if slot == "mode":
                    mode = value
                else:
                    pipeline = value
            elif key == "config_file":
                continue  # bookkeeping metadata, not part of the run
            elif key in _GENERIC_DATA_KEYS:
                data[key] = value
            elif key in _GENERIC_POLICY_KEY_MAP:
                policy[_GENERIC_POLICY_KEY_MAP[key]] = value
            elif key in _GENERIC_OUTPUT_KEYS:
                output[key] = value
            elif key == "paths" and isinstance(value, Mapping):
                # Radiomics-style nested path block: split by purpose.
                for sub_key, sub_value in value.items():
                    if sub_key in _GENERIC_DATA_KEYS:
                        data[sub_key] = sub_value
                    else:
                        output[sub_key] = sub_value
            elif key == "processing" and isinstance(value, Mapping):
                for sub_key, sub_value in value.items():
                    if sub_key in _GENERIC_POLICY_KEY_MAP:
                        policy[_GENERIC_POLICY_KEY_MAP[sub_key]] = sub_value
                    else:
                        spec_params.setdefault(key, {})[sub_key] = sub_value
            else:
                spec_params[key] = value

        if policy:
            workers = policy.get("workers", 1)
            policy["backend"] = (
                "process" if isinstance(workers, int) and workers > 1 else "serial"
            )
        if spec_params:
            warnings.append(
                f"Workflow '{workflow}' has no deep v1 spec model yet; its "
                "algorithmic configuration is preserved verbatim under "
                "spec.params."
            )

        document: Dict[str, Any] = {
            "version": self.SCHEMA_VERSION,
            "workflow": workflow,
            "mode": mode,
            "spec": {"name": workflow, "params": spec_params},
            "data": data,
            "pipeline": pipeline,
            "policy": policy,
            "output": output,
            "legacy": {},
        }
        return LegacyTranslation(
            document=document,
            workflow=workflow,
            unmapped={},
            warnings=tuple(warnings),
        )


# ---------------------------------------------------------------------------
# Method-expression mini-parser (v0 voxel/supervoxel level syntax)
# ---------------------------------------------------------------------------


def _split_top_level(text: str) -> List[str]:
    """
    Split a comma-separated expression at nesting depth zero.

    Args:
        text: Expression body, e.g. ``"raw(A), raw(B)"``.

    Returns:
        Trimmed parts; empty parts are dropped.

    Raises:
        HABITAPIError: On unbalanced parentheses.
    """
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for char in text:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise HABITAPIError(f"Unbalanced ')' in expression: {text!r}.")
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if depth != 0:
        raise HABITAPIError(f"Unbalanced '(' in expression: {text!r}.")
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return [part for part in parts if part]


def _parse_method_expression(
    expression: str, param_names: Optional[frozenset] = None
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Parse one v0 method expression into its outer method and image steps.

    Only the shape needed for translation is extracted: the outermost method
    name and, for every nested one-argument call, the ``(method, image)``
    pair. The semantics mirror the v0 ``FeatureExpressionParser``:

    * a body WITHOUT nested calls is the outer method's own argument list
      (``voxel_radiomics(T1)`` means method ``voxel_radiomics`` on image
      ``T1``), so the first token pairs with the outer method;
    * a bare token inside a nested composition is a parameter reference
      when it names a key of the block's ``params`` mapping
      (``kinetic(raw(A), timestamps)``), otherwise it defaults to a raw
      image name (``concat(T1, T2)``).

    Args:
        expression: Method expression, e.g. ``"concat(raw(A), raw(B))"``.
        param_names: Keys of the block's ``params`` mapping, used to tell
            parameter references from image names; ``None`` treats every
            bare token as an image.

    Returns:
        ``(outer_method, [(method, image), ...])``; ``steps`` is empty for
        a bare method name without parentheses.

    Raises:
        HABITAPIError: On a malformed expression.
    """
    known_params = param_names or frozenset()
    text = expression.strip()
    if not text:
        raise HABITAPIError("Empty method expression.")
    if "(" not in text:
        return text, []
    open_at = text.find("(")
    outer = text[:open_at].strip()
    if not outer.isidentifier():
        raise HABITAPIError(f"Malformed method expression: {expression!r}.")
    if not text.endswith(")"):
        raise HABITAPIError(f"Malformed method expression: {expression!r}.")
    body = text[open_at + 1 : -1]

    if "(" not in body and outer != "concat":
        # Single-method form: the body is the outer method's own argument
        # list; its first token is the image, later tokens are parameter
        # references resolved from the block's params mapping. concat() is
        # excluded: its bare tokens are raw image lists by the v0 rule.
        tokens = _split_top_level(body)
        if tokens and tokens[0] not in known_params:
            return outer, [(outer, tokens[0])]
        return outer, []

    steps: List[Tuple[str, str]] = []
    for part in _split_top_level(body):
        if "(" in part and part.endswith(")"):
            inner_open = part.find("(")
            inner = part[:inner_open].strip()
            inner_body = part[inner_open + 1 : -1]
            image = _split_top_level(inner_body)[0] if inner_body.strip() else ""
            if inner.isidentifier() and image:
                steps.append((inner, image.strip()))
        elif part and part not in known_params:
            # A bare token that is not a parameter reference is a raw image,
            # mirroring the v0 parser's default-raw rule.
            steps.append(("raw", part))
    return outer, steps


# ---------------------------------------------------------------------------
# File-level entry points
# ---------------------------------------------------------------------------


def _guess_workflow_from_path(config_path: Path) -> Optional[str]:
    """
    Guess the workflow alias from path fragments.

    Mirrors ``habit.commands.cmd_check_config._guess_workflow`` (kept in
    sync manually because the spec layer must not import the commands
    layer); the CLI passes its own guess explicitly, so this fallback only
    serves direct Python callers of :func:`migrate_yaml`.

    Args:
        config_path: Path to the v0 YAML file.

    Returns:
        Workflow alias, or ``None`` when no fragment matches.
    """
    parts = [part.lower() for part in config_path.parts]
    name = config_path.name.lower()
    dir_rules = (
        ("preprocessing", "preprocess"),
        ("dicom_sort", "sort-dicom"),
        ("feature_extraction", "extract"),
        ("machine_learning", "model"),
        ("model_comparison", "compare"),
        ("habitat", "habitat"),
        ("radiomics", "radiomics"),
        ("auxiliary", "icc"),
    )
    for needle, alias in dir_rules:
        if needle in parts:
            if alias == "model" and "kfold" in name:
                return "cv"
            if alias == "icc" and ("retest" in name or "test_retest" in name):
                return "retest"
            return alias
    name_rules = (
        ("preprocess", "preprocess"),
        ("kfold", "cv"),
        ("habitat", "habitat"),
        ("extract_features", "extract"),
        ("radiomics", "radiomics"),
        ("machine_learning", "model"),
        ("model_comparison", "compare"),
        ("icc", "icc"),
        ("retest", "retest"),
        ("sort_dicom", "sort-dicom"),
    )
    for needle, alias in name_rules:
        if needle in name:
            return alias
    return None


def default_migrated_path(config_path: Union[str, Path]) -> Path:
    """
    Return the default v1 output path for a v0 config: ``foo.v1.yaml``.

    Args:
        config_path: Source v0 YAML path.

    Returns:
        Sibling path with ``.v1`` inserted before the suffix.
    """
    source = Path(config_path)
    suffix = source.suffix or ".yaml"
    return source.with_name(f"{source.stem}.v1{suffix}")


def migrate_yaml(
    config_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    dry_run: bool = False,
    workflow: Optional[str] = None,
) -> MigrationReport:
    """
    Migrate one v0 YAML config into the v1 document layout.

    Args:
        config_path: Source v0 YAML file.
        output_path: Destination v1 file; defaults to ``<name>.v1.yaml``
            next to the source.
        dry_run: When ``True``, compute the diff without writing anything.
        workflow: Workflow alias; guessed from the path when omitted.

    Returns:
        The migration report with the translated document, unified diff and
        warnings.

    Raises:
        HABITAPIError: When the file is already v1, the workflow cannot be
            determined, or translation fails.
    """
    source = Path(config_path)
    payload = _read_yaml(source)
    if detect_yaml_version(payload) == "v1":
        raise HABITAPIError(
            f"{source} already follows the v1 layout; nothing to migrate."
        )

    alias = (workflow or "").strip().lower() or _guess_workflow_from_path(source)
    if not alias:
        raise HABITAPIError(
            f"Cannot guess the workflow from {source}; pass workflow= "
            f"explicitly (one of: {', '.join(_KNOWN_WORKFLOWS)})."
        )

    translation = LegacyConfigAdapter().translate(payload, alias)
    v1_text = yaml.safe_dump(
        translation.document, sort_keys=False, allow_unicode=True
    )
    source_text = source.read_text(encoding="utf-8")
    diff = "".join(
        difflib.unified_diff(
            source_text.splitlines(keepends=True),
            v1_text.splitlines(keepends=True),
            fromfile=str(source),
            tofile=str(output_path or default_migrated_path(source)),
        )
    )

    destination: Optional[Path] = None
    if not dry_run:
        destination = _write_yaml(
            translation.document, output_path or default_migrated_path(source)
        )
    return MigrationReport(
        source=source,
        destination=destination,
        workflow=alias,
        document=translation.document,
        diff=diff,
        warnings=translation.warnings,
    )


def validate_v1_document(
    payload: Mapping[str, Any], workflow: Optional[str] = None
) -> None:
    """
    Validate the structure of a v1 document payload.

    Structural checks: section types, the ``version`` tag, and -- for
    habitat -- that the ``spec``/``policy`` sections parse into
    :class:`HabitatSpec` / :class:`RunPolicy` and that every declared
    habitat component name is registered. ML documents still validate
    structure only; registry availability for ML components is checked
    when the modelling recipe assembles its pipeline.

    Args:
        payload: Top-level YAML mapping detected as v1.
        workflow: Expected workflow alias; when given, the document's
            ``workflow`` tag must match.

    Raises:
        HABITAPIError: On any structural violation.
        ComponentNotFoundError: When a habitat spec names an unknown component.
    """
    if not isinstance(payload, Mapping):
        raise HABITAPIError(
            "A v1 config document must be a mapping at the top level; "
            f"got {type(payload).__name__}."
        )
    version = str(payload.get("version", "")).strip()
    if not version.startswith("1."):
        raise HABITAPIError(
            "A v1 config document needs a version tag like \"1.0\"; "
            f"got {payload.get('version')!r}."
        )
    doc_workflow = payload.get("workflow")
    if not isinstance(doc_workflow, str) or not doc_workflow.strip():
        raise HABITAPIError("A v1 config document needs a 'workflow' string.")
    if workflow and doc_workflow.strip().lower() != workflow.strip().lower():
        raise HABITAPIError(
            f"Document workflow is {doc_workflow!r}, not {workflow!r}."
        )

    for section in ("data", "output", "legacy"):
        value = payload.get(section)
        if value is not None and not isinstance(value, Mapping):
            raise HABITAPIError(
                f"v1 section '{section}' must be a mapping; "
                f"got {type(value).__name__}."
            )

    mode = payload.get("mode")
    if mode is not None and not isinstance(mode, str):
        raise HABITAPIError(
            f"v1 section 'mode' must be a string; got {type(mode).__name__}."
        )

    if doc_workflow.strip().lower() == "habitat":
        _validate_habitat_v1(payload, mode)

    policy_payload = payload.get("policy")
    if policy_payload:
        if not isinstance(policy_payload, Mapping):
            raise HABITAPIError(
                "v1 section 'policy' must be a mapping; "
                f"got {type(policy_payload).__name__}."
            )
        RunPolicy.from_dict(policy_payload)

    # The ML check runs after the policy check so a policy violation keeps
    # surfacing as a policy error even when the spec section is also absent.
    if doc_workflow.strip().lower() in ("model", "cv"):
        _validate_ml_v1(payload, mode)


def _validate_habitat_v1(payload: Mapping[str, Any], mode: Optional[str]) -> None:
    """
    Validate the habitat-specific sections of a v1 document.

    A train document must carry a parseable ``HabitatSpec``; a predict
    document may omit ``spec`` (the definition then lives in the loaded
    pipeline artefact) but must name its ``pipeline`` path.

    Args:
        payload: Full v1 document mapping.
        mode: Document ``mode`` value, already type-checked.

    Raises:
        HABITAPIError: On a missing/invalid spec or pipeline reference.
    """
    spec_payload = payload.get("spec")
    if spec_payload is None:
        if mode != "predict":
            raise HABITAPIError(
                "A habitat v1 document needs a 'spec' section unless "
                "mode: predict."
            )
        if not payload.get("pipeline"):
            raise HABITAPIError(
                "A predict-mode habitat v1 document without a 'spec' section "
                "must name its 'pipeline' path."
            )
        return
    if not isinstance(spec_payload, Mapping):
        raise HABITAPIError(
            f"v1 section 'spec' must be a mapping; got {type(spec_payload).__name__}."
        )
    spec = HabitatSpec.from_dict(spec_payload)
    spec.validate_dataflow()
    from habit.domain.assembly import validate_habitat_spec_registry

    validate_habitat_spec_registry(spec)


def _validate_ml_v1(payload: Mapping[str, Any], mode: Optional[str]) -> None:
    """
    Validate the ML-specific sections of a v1 document.

    Same contract as the habitat counterpart: a train document must carry a
    parseable :class:`MLSpec`; a predict document may omit ``spec`` (the
    definition then lives in the loaded pipeline artefact) but must name
    its ``pipeline`` path.

    Args:
        payload: Full v1 document mapping.
        mode: Document ``mode`` value, already type-checked.

    Raises:
        HABITAPIError: On a missing/invalid spec or pipeline reference.
    """
    spec_payload = payload.get("spec")
    if spec_payload is None:
        if mode != "predict":
            raise HABITAPIError(
                "A model/cv v1 document needs a 'spec' section unless "
                "mode: predict."
            )
        if not payload.get("pipeline"):
            raise HABITAPIError(
                "A predict-mode model/cv v1 document without a 'spec' "
                "section must name its 'pipeline' path."
            )
        return
    if not isinstance(spec_payload, Mapping):
        raise HABITAPIError(
            f"v1 section 'spec' must be a mapping; got {type(spec_payload).__name__}."
        )
    MLSpec.from_dict(spec_payload)
