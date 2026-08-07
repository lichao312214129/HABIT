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
"""L4 YAML entry point: translate v0.1 configs and dispatch recipes.

This module is the notebook-friendly counterpart to the CLI: load a legacy
YAML, translate it with :class:`~habit.spec.legacy.LegacyConfigAdapter`, and
run the matching L4 recipe. It deliberately stays thin -- no new algorithms,
only the same assembly the command layer performs.

Known limitations (stage-5 debt, mirrored from the CLI):

* **v0.1 YAML only.** v1 documents should call recipes directly with an
  in-memory :class:`~habit.spec.specs.HabitatSpec` / :class:`~habit.spec.specs.MLSpec`.
* **Full check-config coverage.** All ten ``habit check-config`` workflows
  (including ``sort-dicom``) are routed here through the matching L4 recipes.
* **Legacy artefact paths.** Habitat predict on raw v0.1 pickles and ML
  predict / precomputed-ICC train still delegate to the v0.1 engines through
  ``habit.api.*`` (logged, same as CLI).
* **Disk output is opt-in.** Pass ``save=True`` to persist artefacts the way
  the CLI does; the default returns the in-memory result only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import yaml

from habit.adapters.directory import DirectoryDataSource
from habit.adapters.image_refs import FileImageRef
from habit.contracts.habitat import HabitatModel
from habit.contracts.subject import Cohort, Subject
from habit.contracts.table import FeatureTable
from habit.execution.checkpoint import CheckpointStore
from habit.execution.selection import backend_from_policy
from habit.exceptions import HABITAPIError
from habit.recipes.comparison import compare_models
from habit.recipes.features import extract_habitat_features, traditional_radiomics
from habit.recipes.habitat import apply_habitat_model, direct_pooling, one_step, two_step
from habit.recipes.icc import icc_analysis
from habit.recipes.modeling import CVResult, ModelResult, cross_validate, train_model
from habit.recipes.preprocess import preprocess_images
from habit.recipes.result import StudyResult
from habit.recipes.sort_dicom import sort_dicom
from habit.recipes.test_retest import test_retest_analysis
from habit.spec.legacy import LegacyConfigAdapter, _guess_workflow_from_path, detect_yaml_version, validate_v1_document
from habit.spec.policy import RunPolicy
from habit.spec.specs import HabitatSpec, MLSpec

__all__ = ["run_from_yaml"]

#: Workflows ``run_from_yaml`` can dispatch (full ``habit check-config`` set).
_SUPPORTED_WORKFLOWS: Tuple[str, ...] = (
    "habitat",
    "model",
    "cv",
    "compare",
    "preprocess",
    "icc",
    "retest",
    "extract",
    "radiomics",
    "sort-dicom",
)

#: clustering_mode -> L4 habitat recipe.
_RECIPE_BY_MODE: Mapping[str, Callable[..., StudyResult]] = {
    "two_step": two_step,
    "one_step": one_step,
    "direct_pooling": direct_pooling,
}

_TRAIN_CHECKPOINT_DIRNAME = ".habitat_checkpoint"
_PREDICT_CHECKPOINT_DIRNAME = ".habitat_predict_checkpoint"
_ZIP_MAGIC = b"PK\x03\x04"
_V1_MODEL_NAME = "habitat_model.habitatmodel"
_LEGACY_PICKLE_MESSAGE = (
    "Legacy v0.1 pickle pipelines are not supported in HABIT v1.0. "
    f"Train a model to produce {_V1_MODEL_NAME!r}, then run predict with "
    "pipeline_path pointing at that archive or call "
    "habit.recipes.apply_habitat_model in Python."
)


def run_from_yaml(
    config_path: Union[str, Path],
    *,
    workflow: Optional[str] = None,
    save: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """
    Run a YAML configuration document through the matching recipe.

    This is the programmatic twin of the CLI. The document version is
    detected automatically: **v1** documents (``version: '1.0'``) are read
    directly for the habitat train/predict and ML train/cv workflows, while
    **v0.1** documents are first translated through
    :class:`~habit.spec.legacy.LegacyConfigAdapter`.

    Args:
        config_path: Path to the YAML file. In v0.1 documents, relative
            paths inside the file resolve against the file's directory; v1
            documents use paths as written (current working directory).
        workflow: Workflow alias (``habitat``, ``model``, ``cv``, ``compare``,
            ``preprocess``, ``icc``, ``retest``, ``extract``, ``radiomics``,
            ``sort-dicom``).
            When omitted, guessed from the path/name the same way as
            :func:`~habit.spec.legacy.migrate_yaml`.
        save: When ``True``, persist outputs under the config's ``out_dir`` /
            ``output`` / ``output_dir`` the way the CLI does for habitat and ML
            workflows. Default is ``False`` so callers keep results in memory.
            Thin-delegation workflows (``preprocess``, ``icc``, ``retest``,
            ``extract``, ``radiomics``, ``compare``) always write through the
            v0.1 engines regardless of this flag.
        logger: Optional logger forwarded to delegated v0.1 API paths.

    Returns:
        :class:`~habit.recipes.result.StudyResult` for habitat train/predict,
        :class:`~habit.recipes.modeling.ModelResult` or
        :class:`~habit.recipes.modeling.CVResult` for ML train/K-fold,
        or :class:`~habit.api.contracts.WorkflowResult` for comparison and
        thin-delegation workflows.

    Raises:
        HABITAPIError: When the workflow cannot be determined or translation fails.
        NotImplementedError: When the workflow is not routed by ``run_from_yaml``.
        FileNotFoundError: When ``config_path`` does not exist.
        ValueError: When the translated document or on-disk inputs are invalid.

    Examples:
        Run the shipped v1 two-step demo exactly as the CLI would, writing
        outputs under the document's ``output.out_dir``:

        >>> import habit.recipes as recipes
        >>> result = recipes.run_from_yaml(  # doctest: +SKIP
        ...     "config/habitat/config_habitat_two_step_v1.yaml",
        ...     workflow="habitat",  # optional when the path names the workflow
        ...     save=True,
        ... )
        >>> result.habitat_model.n_habitats >= 2  # doctest: +SKIP
        True

        A v0.1 document runs through the same call; it is translated to a
        v1 spec first:

        >>> result = recipes.run_from_yaml(  # doctest: +SKIP
        ...     "config/habitat/config_habitat_two_step.yaml",
        ...     save=False,  # keep the StudyResult in memory
        ... )
    """
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    alias = (workflow or "").strip().lower() or (_guess_workflow_from_path(path) or "")
    if not alias:
        raise HABITAPIError(
            f"Cannot guess workflow from {path}; pass workflow= explicitly "
            f"(supported by run_from_yaml: {', '.join(_SUPPORTED_WORKFLOWS)})."
        )
    if alias not in _SUPPORTED_WORKFLOWS:
        raise NotImplementedError(
            f"run_from_yaml does not route workflow {alias!r} yet. "
            f"Supported workflows: {', '.join(_SUPPORTED_WORKFLOWS)}. "
            "Use the matching CLI command or public API helper instead."
        )

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise HABITAPIError(
            f"Configuration file {path} must contain a YAML mapping at the top level."
        )
    if detect_yaml_version(payload) == "v1":
        return _run_v1_yaml(path, payload, workflow=alias, save=save, logger=logger)

    if alias == "habitat":
        return _run_habitat_yaml(path, save=save, logger=logger)
    if alias == "compare":
        return _run_compare_yaml(path, save=save, logger=logger)
    if alias == "cv":
        return _run_ml_yaml(path, payload, workflow="cv", save=save, logger=logger)
    if alias == "model":
        return _run_ml_yaml(path, payload, workflow="model", save=save, logger=logger)
    if alias == "preprocess":
        return _run_preprocess_yaml(path, logger=logger)
    if alias == "icc":
        return _run_icc_yaml(path, logger=logger)
    if alias == "retest":
        return _run_retest_yaml(path, logger=logger)
    if alias == "extract":
        return _run_extract_yaml(path, logger=logger)
    if alias == "radiomics":
        return _run_radiomics_yaml(path, logger=logger)
    if alias == "sort-dicom":
        return _run_sort_dicom_yaml(path, logger=logger)
    raise NotImplementedError(
        f"run_from_yaml does not route workflow {alias!r} yet. "
        f"Supported workflows: {', '.join(_SUPPORTED_WORKFLOWS)}."
    )


def _run_v1_yaml(
    path: Path,
    document: Mapping[str, Any],
    *,
    workflow: str,
    save: bool,
    logger: Optional[logging.Logger],
) -> Any:
    """
    Dispatch a native v1 document through the same recipes as translated v0 YAML.

    Args:
        path: Resolved config path (for error messages and relative data paths).
        document: Parsed v1 YAML mapping.
        workflow: Workflow alias passed by the caller or inferred from the path.
        save: Whether to persist outputs for habitat / ML workflows.
        logger: Optional logger forwarded to delegated paths.

    Returns:
        The in-memory recipe result, optionally written when ``save=True``.
    """
    alias = str(document.get("workflow", workflow)).strip().lower()
    validate_v1_document(document, workflow=workflow if workflow else None)

    if alias == "habitat":
        return _run_v1_habitat_yaml(path, document, save=save, logger=logger)
    if alias in ("model", "cv"):
        return _run_v1_ml_yaml(path, document, workflow=alias, save=save, logger=logger)
    raise NotImplementedError(
        f"run_from_yaml v1 direct read routes habitat and ML workflows only; "
        f"got {alias!r}. Translate other workflows with migrate-config or call "
        "the matching recipe/API helper."
    )


def _run_v1_habitat_yaml(
    path: Path,
    document: Mapping[str, Any],
    *,
    save: bool,
    logger: Optional[logging.Logger],
) -> StudyResult:
    """Run a habitat v1 document without round-tripping through v0 schemas."""
    config = _habitat_config_from_v1_document(document)
    mode = str(document.get("mode", "train")).strip().lower()
    if mode == "predict":
        pipeline_file = Path(config.pipeline_path) if config.pipeline_path else None
        if pipeline_file is None or not pipeline_file.is_file():
            raise ValueError(
                "A predict-mode habitat v1 document needs a readable 'pipeline' path."
            )
        if not _is_v1_model_archive(pipeline_file):
            raise HABITAPIError(
                "Legacy v0.1 pickle pipelines are not supported by v1 direct read; "
                "keep the source file as v0 YAML or point 'pipeline' at a v1 "
                ".habitatmodel archive."
            )
        result = _habitat_predict_v1(document, config, logger=logger)
    else:
        result = _habitat_train_v1(document, config, logger=logger)

    if save:
        _save_habitat_result(result, config)
    return result


def _habitat_train_v1(
    document: Mapping[str, Any],
    config: Any,
    *,
    logger: Optional[logging.Logger],
) -> StudyResult:
    """Fit through the v1 recipe named by the v1 document spec."""
    spec_payload = document.get("spec")
    if spec_payload is None:
        raise ValueError(
            "A train-mode habitat v1 document must carry a non-empty 'spec' section."
        )
    spec = HabitatSpec.from_dict(spec_payload)
    backend = backend_from_policy(RunPolicy.from_dict(document.get("policy") or {}))
    cohort = _load_habitat_cohort(config, spec, logger=logger)
    checkpoint = _checkpoint_store_for(config, spec=spec, predict=False)
    mode = _clustering_mode_from_spec(spec)
    recipe = _RECIPE_BY_MODE.get(mode)
    if recipe is None:
        raise ValueError(
            f"Unknown clustering_mode {mode!r}; expected one of {sorted(_RECIPE_BY_MODE)}."
        )
    return recipe(cohort, spec, backend=backend, checkpoint=checkpoint)


def _habitat_predict_v1(
    document: Mapping[str, Any],
    config: Any,
    *,
    logger: Optional[logging.Logger],
) -> StudyResult:
    """Apply a fitted v1 habitat model archive described by a v1 document."""
    model = HabitatModel.load(Path(config.pipeline_path))
    spec_payload = document.get("spec")
    if spec_payload is None:
        spec_payload = model.spec_payload
    spec = HabitatSpec.from_dict(spec_payload)
    backend = backend_from_policy(RunPolicy.from_dict(document.get("policy") or {}))
    cohort = _load_habitat_cohort(config, spec, logger=logger)
    checkpoint = _checkpoint_store_for(config, spec=spec, predict=True)
    return apply_habitat_model(
        cohort, spec, model, backend=backend, checkpoint=checkpoint
    )


def _run_v1_ml_yaml(
    path: Path,
    document: Mapping[str, Any],
    *,
    workflow: str,
    save: bool,
    logger: Optional[logging.Logger],
) -> Union[ModelResult, CVResult, Any]:
    """Run an ML v1 document through the v1 recipes."""
    mode = str(document.get("mode", "train")).strip().lower()
    if mode == "predict":
        from habit.api.machine_learning import MLConfig, run_ml

        config = MLConfig.model_validate(_ml_v0_payload_from_v1(document, path))
        return run_ml(config, logger=logger)

    spec = _ml_spec_from_document(document)
    table = _load_feature_table_from_v1(document, path, logger=logger)

    if workflow == "cv":
        legacy = document.get("legacy") or {}
        n_splits = int(legacy.get("n_splits", 5))
        seed = spec.random_seed
        result: Union[ModelResult, CVResult] = cross_validate(
            table,
            spec,
            n_splits=n_splits,
            seed=seed,
        )
        if save:
            config = _ml_config_shim_from_v1(document, path)
            _save_cv_result(result, config, table=table, logger=logger)
        return result

    result = train_model(table, spec, seed=spec.random_seed)
    if save:
        config = _ml_config_shim_from_v1(document, path)
        _save_model_result(result, table, config, logger=logger)
    return result


def _habitat_config_from_v1_document(document: Mapping[str, Any]) -> Any:
    """Build a v0-shaped habitat config shim from a v1 document."""
    output = dict(document.get("output") or {})
    policy = RunPolicy.from_dict(document.get("policy") or {})
    data = dict(document.get("data") or {})
    legacy = dict(document.get("legacy") or {})

    class _V1HabitatConfig:
        """Minimal attribute bag shared by habitat YAML helpers."""

        pass

    config = _V1HabitatConfig()
    config.data_dir = data.get("source")
    config.out_dir = output.get("out_dir", ".")
    config.plot_curves = output.get("plot_curves", True)
    config.save_images = output.get("save_images", True)
    config.save_results_csv = output.get("save_results_csv", True)
    config.habitats_results_format = output.get("habitats_results_format", "parquet")
    config.pipeline_path = document.get("pipeline")
    config.checkpoint_dir = policy.checkpoint_dir
    config.resume = policy.resume
    config.retry_failed_subjects = policy.retry_failed_subjects
    config.force_rerun_subjects = policy.force_rerun_subjects
    config.clear_checkpoint_on_success = policy.clear_checkpoint_on_success
    config.strict_checkpoint_hash = policy.strict_checkpoint_hash
    config.run_mode = document.get("mode", "train")

    spec_payload = document.get("spec")
    clustering_mode = legacy.get("habitat_segmentation", {}).get("clustering_mode")
    if clustering_mode is None and isinstance(spec_payload, Mapping):
        name = str(spec_payload.get("name", ""))
        if name.startswith("habitat_"):
            clustering_mode = name[len("habitat_") :]
    if clustering_mode is None:
        clustering_mode = "two_step"

    class _Segmentation:
        clustering_mode: str

    segmentation = _Segmentation()
    segmentation.clustering_mode = str(clustering_mode)
    config.habitat_segmentation = segmentation
    config.model_dump = lambda: dict(document)  # type: ignore[method-assign]
    return config


def _ml_v0_payload_from_v1(document: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    """Reconstruct a v0 MLConfig-shaped mapping from a v1 document."""
    data = dict(document.get("data") or {})
    output = dict(document.get("output") or {})
    legacy = dict(document.get("legacy") or {})
    payload: Dict[str, Any] = {
        "run_mode": document.get("mode", "train"),
        "input": data.get("input") or [],
        "output": output.get("out_dir", "."),
        "random_state": (document.get("spec") or {}).get("random_seed", 0),
    }
    payload.update(legacy)
    spec = document.get("spec") or {}
    if spec.get("classifier"):
        payload["models"] = {
            spec["classifier"]["name"]: {"params": spec["classifier"].get("params") or {}}
        }
    payload["feature_selection_methods"] = _v0_selection_methods_from_spec(spec)
    return payload


def _v0_selection_methods_from_spec(
    spec: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """
    Reverse-translate a v1 spec section into v0 ``feature_selection_methods``.

    v0 had no ordered step list: a selector's position was a boolean,
    ``before_z_score``, meaning "before the single normalisation block".
    Both v1 layouts reduce to that boolean the same way -- a selector is
    ``before_z_score`` exactly when no table preprocessor precedes it:

    * the deprecated three-chain layout states it structurally (the
      pre-preprocessing chain runs before all preprocessing);
    * the ordered ``steps`` layout states it positionally, so the flag is
      read off the position of the first preprocessor.

    Interleaved orders that v0 cannot express (a second preprocessor after a
    later selector) collapse onto the same boolean; nothing here depends on
    the distinction, because the reconstructed v0 payload is used only to
    build the ``MLConfig`` shim that LOADS the feature table. The
    scientifically meaningful order is what ``MLSpec.steps`` already carries
    into ``build_table_pipeline``.

    Args:
        spec: The v1 document's ``spec`` section, as a plain mapping.

    Returns:
        List[Dict[str, Any]]: v0 ``feature_selection_methods`` entries, in
        execution order, each carrying ``before_z_score`` when it applies.
    """
    ordered_steps = spec.get("steps")
    if ordered_steps:
        # Lazy import: the L4 recipe layer may read L3 registries, but a
        # module-level import here would pull the whole domain package in
        # just to translate a config document.
        from habit.domain.table_preprocessing import TablePreprocessorRegistry

        preprocessor_names = set(TablePreprocessorRegistry.available())
        methods: List[Dict[str, Any]] = []
        seen_preprocessor = False
        for entry in ordered_steps:
            name = str(entry.get("name", ""))
            if name in preprocessor_names:
                seen_preprocessor = True
                continue
            params = dict(entry.get("params") or {})
            if not seen_preprocessor:
                params["before_z_score"] = True
            methods.append({"method": name, "params": params})
        return methods
    return [
        # Pre-preprocessing selectors map back to v0's ``before_z_score``
        # stage flag; stage order (pre first) matches v0 execution order.
        {"method": entry["name"], "params": {**(entry.get("params") or {}), "before_z_score": True}}
        for entry in (spec.get("pre_preprocessing_feature_selectors") or [])
    ] + [
        {"method": entry["name"], "params": entry.get("params") or {}}
        for entry in (spec.get("feature_selectors") or [])
    ]


def _ml_config_shim_from_v1(document: Mapping[str, Any], path: Path) -> Any:
    """Build a minimal config shim for ML artefact writers."""
    output = dict(document.get("output") or {})
    legacy = dict(document.get("legacy") or {})

    class _V1MLConfig:
        pass

    config = _V1MLConfig()
    config.output = output.get("out_dir", ".")
    config.is_save_model = bool(output.get("is_save_model", legacy.get("is_save_model", False)))
    config.random_state = (document.get("spec") or {}).get("random_seed", 0)
    return config


def _load_feature_table_from_v1(
    document: Mapping[str, Any],
    path: Path,
    *,
    logger: Optional[logging.Logger],
) -> FeatureTable:
    """Assemble a feature table from a v1 ML document's ``data.input`` list."""
    from habit.api.machine_learning import MLConfig

    config = MLConfig.model_validate(_ml_v0_payload_from_v1(document, path))
    return _load_feature_table(config, logger=logger)


def _clustering_mode_from_spec(spec: HabitatSpec) -> str:
    """Infer the L4 recipe key from a translated habitat spec name."""
    prefix = "habitat_"
    if spec.name.startswith(prefix):
        mode = spec.name[len(prefix) :]
        if mode in _RECIPE_BY_MODE:
            return mode
    raise ValueError(
        f"Cannot infer clustering_mode from HabitatSpec name {spec.name!r}; "
        f"expected one of {sorted(_RECIPE_BY_MODE)}."
    )


def _run_habitat_yaml(
    path: Path,
    *,
    save: bool,
    logger: Optional[logging.Logger],
) -> Any:
    """Translate and run a habitat workflow YAML."""
    from habit.api.habitat import HabitatAnalysisConfig

    config = HabitatAnalysisConfig.from_file(str(path))
    run_mode = str(config.run_mode)
    if run_mode == "predict":
        pipeline_file = Path(config.pipeline_path) if config.pipeline_path else None
        if pipeline_file is None or not pipeline_file.is_file():
            raise ValueError(
                "pipeline_path is required for predict mode (set it in the YAML)."
            )
        if not _is_v1_model_archive(pipeline_file):
            raise ValueError(
                f"{_LEGACY_PICKLE_MESSAGE} Got: {pipeline_file}"
            )
        result = _habitat_predict(config, logger=logger)
    else:
        result = _habitat_train(config, logger=logger)

    if save:
        _save_habitat_result(result, config)
    return result


def _habitat_train(
    config: Any,
    *,
    logger: Optional[logging.Logger],
) -> StudyResult:
    """Fit through the v1 recipe named by ``clustering_mode``."""
    document = _translate_habitat_document(config)
    spec_payload = document.get("spec")
    if spec_payload is None:
        raise ValueError(
            "The train config translated to no habitat spec; train configs "
            "must carry feature_construction and habitat_segmentation blocks."
        )
    spec = HabitatSpec.from_dict(spec_payload)
    backend = backend_from_policy(RunPolicy.from_dict(document.get("policy") or {}))
    cohort = _load_habitat_cohort(config, spec, logger=logger)
    checkpoint = _checkpoint_store_for(config, spec=spec, predict=False)

    mode = str(config.habitat_segmentation.clustering_mode)
    recipe = _RECIPE_BY_MODE.get(mode)
    if recipe is None:
        raise ValueError(
            f"Unknown clustering_mode {mode!r}; expected one of {sorted(_RECIPE_BY_MODE)}."
        )
    return recipe(cohort, spec, backend=backend, checkpoint=checkpoint)


def _habitat_predict(
    config: Any,
    *,
    logger: Optional[logging.Logger],
) -> StudyResult:
    """Apply a fitted v1 habitat model archive."""
    document = _translate_habitat_document(config)
    model = HabitatModel.load(Path(config.pipeline_path))
    spec_payload = document.get("spec")
    if spec_payload is None:
        # Stub predict YAMLs rely on the embedded spec inside the model archive.
        spec_payload = model.spec_payload
    spec = HabitatSpec.from_dict(spec_payload)
    backend = backend_from_policy(RunPolicy.from_dict(document.get("policy") or {}))
    cohort = _load_habitat_cohort(config, spec, logger=logger)
    checkpoint = _checkpoint_store_for(config, spec=spec, predict=True)
    return apply_habitat_model(
        cohort, spec, model, backend=backend, checkpoint=checkpoint
    )


def _run_ml_yaml(
    path: Path,
    payload: Mapping[str, Any],
    *,
    workflow: str,
    save: bool,
    logger: Optional[logging.Logger],
) -> Union[ModelResult, CVResult, Any]:
    """Translate and run an ML hold-out or K-fold YAML."""
    config = _load_ml_config(path)
    if str(config.run_mode) == "predict":
        from habit.api.machine_learning import run_ml

        return run_ml(config, logger=logger)

    document = LegacyConfigAdapter().translate(config.model_dump(), workflow).document
    spec = _ml_spec_from_document(document)
    table = _load_feature_table(config, logger=logger)

    if workflow == "cv":
        result: Union[ModelResult, CVResult] = cross_validate(
            table,
            spec,
            n_splits=int(config.n_splits),
            seed=config.random_state,
        )
        if save:
            _save_cv_result(result, config, table=table, logger=logger)
    else:
        result = train_model(
            table,
            spec,
            seed=config.random_state,
            **_resolve_ml_split_kwargs(config, table, logger=logger),
        )
        if save:
            _save_model_result(result, table, config, logger=logger)
    return result


def _run_compare_yaml(
    path: Path,
    *,
    save: bool,
    logger: Optional[logging.Logger],
) -> Any:
    """Run model comparison through the L4 recipe."""
    from habit.api.machine_learning import ModelComparisonConfig

    config = ModelComparisonConfig.from_file(str(path))
    output_dir = str(config.output_dir) if save else None
    return compare_models(config, logger=logger, output_dir=output_dir)


def _run_preprocess_yaml(
    path: Path,
    *,
    logger: Optional[logging.Logger],
) -> Any:
    """Run image preprocessing through the L4 recipe."""
    from habit.api.preprocessing import PreprocessingConfig

    config = PreprocessingConfig.from_file(str(path))
    return preprocess_images(config, logger=logger)


def _run_icc_yaml(
    path: Path,
    *,
    logger: Optional[logging.Logger],
) -> Any:
    """Run ICC reliability analysis through the L4 recipe."""
    from habit.api.analysis import ICCConfig

    config = ICCConfig.from_file(str(path))
    return icc_analysis(config)


def _run_retest_yaml(
    path: Path,
    *,
    logger: Optional[logging.Logger],
) -> Any:
    """Run test-retest label mapping through the L4 recipe."""
    from habit.api.analysis import TestRetestConfig

    config = TestRetestConfig.from_file(str(path))
    return test_retest_analysis(config, logger=logger)


def _run_extract_yaml(
    path: Path,
    *,
    logger: Optional[logging.Logger],
) -> Any:
    """Run habitat feature extraction through the L4 recipe."""
    from habit.api.habitat import load_feature_extraction_config

    config, plugin_configs = load_feature_extraction_config(str(path))
    return extract_habitat_features(
        config,
        plugin_configs=plugin_configs,
        logger=logger,
    )


def _run_radiomics_yaml(
    path: Path,
    *,
    logger: Optional[logging.Logger],
) -> Any:
    """Run standalone traditional radiomics through the L4 recipe."""
    from habit.api.habitat import RadiomicsConfig

    config = RadiomicsConfig.from_file(str(path))
    return traditional_radiomics(config, logger=logger)


def _run_sort_dicom_yaml(
    path: Path,
    *,
    logger: Optional[logging.Logger],
) -> Any:
    """Run standalone DICOM sort through the L4 recipe."""
    from habit.api.dicom_sort import DicomSortConfig

    config = DicomSortConfig.from_file(str(path))
    return sort_dicom(config, logger=logger)


def _translate_habitat_document(config: Any) -> Dict[str, Any]:
    """Translate a validated v0.1 habitat config into the v1 document model."""
    return LegacyConfigAdapter().translate(config.model_dump(), "habitat").document


def _ml_spec_from_document(document: Mapping[str, Any]) -> MLSpec:
    """Parse the ``spec`` section of a translated ML document."""
    spec_payload = document.get("spec")
    if spec_payload is None:
        raise ValueError(
            "The train config translated to no ML spec; train configs must "
            "carry a non-empty 'models' block."
        )
    return MLSpec.from_dict(spec_payload)


def _load_ml_config(path: Path) -> Any:
    """Load and validate an ML config through the public API facade."""
    from habit.api.machine_learning import MLConfig

    return MLConfig.from_file(str(path))


def _resolve_ml_split_kwargs(
    config: Any, table: FeatureTable, *, logger: Optional[logging.Logger]
) -> Dict[str, Any]:
    """
    Turn the v0.1 split settings into ``train_model`` keyword arguments.

    Mirrors ``habit.commands.cmd_ml._resolve_split_kwargs`` so
    ``run_from_yaml`` stays aligned with the CLI without importing the
    commands layer: ``custom`` follows the id files (ids the table does not
    have are dropped with a warning, the v0.1 SplitStrategy rule), while
    ``stratified``/``random`` draw a ``test_size`` hold-out.
    """
    import json

    log = logger or logging.getLogger(__name__)
    split_method = str(getattr(config, "split_method", "stratified"))
    if split_method != "custom":
        return {
            "test_size": float(getattr(config, "test_size", 0.3)),
            "stratify": split_method == "stratified",
        }

    train_ids_file = getattr(config, "train_ids_file", None)
    test_ids_file = getattr(config, "test_ids_file", None)
    if not train_ids_file or not test_ids_file:
        raise ValueError("Custom split requires train_ids_file and test_ids_file")

    def _read_ids(path: str) -> List[str]:
        """Read a v0.1 id file: JSON list, comma-separated, or one per line."""
        content = Path(path).read_text(encoding="utf-8").strip()
        if content.startswith("["):
            return [str(item) for item in json.loads(content)]
        if "," in content:
            return [item.strip() for item in content.split(",")]
        return [line.strip() for line in content.split("\n") if line.strip()]

    def _valid_ids(ids: List[str], side: str) -> List[str]:
        """Keep the ids present in the table, warning about the rest."""
        valid = [row_id for row_id in ids if row_id in row_ids]
        missing = [row_id for row_id in ids if row_id not in row_ids]
        if missing:
            log.warning(
                "Custom split: %d %s IDs not found in data index. Sample: %s",
                len(missing),
                side,
                missing[:10],
            )
        if not valid:
            raise ValueError(
                f"Custom split: no {side} IDs from the id file are present "
                "in the input table."
            )
        return valid

    id_column = table.id_columns[0]
    row_ids = {str(value) for value in table.frame[id_column]}
    return {
        "train_ids": _valid_ids(_read_ids(str(train_ids_file)), "train"),
        "test_ids": _valid_ids(_read_ids(str(test_ids_file)), "test"),
    }


def _load_feature_table(config: Any, *, logger: Optional[logging.Logger]) -> FeatureTable:
    """
    Assemble one labelled :class:`FeatureTable` from the v0.1 ``input`` list.

    Mirrors ``habit.commands.cmd_ml._load_feature_table`` so ``run_from_yaml``
    stays aligned with the CLI without importing the commands layer.
    """
    import pandas as pd

    from habit.contracts.outcome import BinaryOutcome
    from habit.contracts.provenance import Provenance

    if not config.input:
        raise ValueError("ML config 'input' must contain at least one table.")

    log = logger or logging.getLogger(__name__)
    subject_id_col: Optional[str] = None
    label_col: Optional[str] = None
    merged: Optional[pd.DataFrame] = None
    label_series: Optional[pd.Series] = None

    for index, entry in enumerate(config.input):
        table_path = Path(entry.path)
        if not table_path.is_file():
            raise ValueError(f"Input table not found: {table_path}")

        subj_col = entry.subject_id_col
        lbl_col = entry.label_col
        if not subj_col or not lbl_col:
            raise ValueError(
                f"subject_id_col and label_col are required for input table {table_path}."
            )

        frame = pd.read_csv(table_path, dtype={subj_col: str})
        if subj_col not in frame.columns or lbl_col not in frame.columns:
            raise ValueError(
                f"Input table {table_path} is missing required columns "
                f"{subj_col!r} and/or {lbl_col!r}."
            )
        frame[subj_col] = frame[subj_col].astype(str)

        if subject_id_col is None:
            subject_id_col = subj_col
            label_col = lbl_col

        features = list(entry.features or [])
        available = [column for column in frame.columns if column not in {subj_col, lbl_col}]
        selected = features if features else available
        prefix = str(entry.name or "").strip()
        rename = {
            column: f"{prefix}{column}"
            for column in selected
            if prefix and column in frame.columns
        }
        subset = frame.set_index(subj_col)[selected].rename(columns=rename)

        if merged is None:
            merged = subset
            label_series = frame.set_index(subj_col)[lbl_col]
        else:
            overlap = merged.columns.intersection(subset.columns)
            if len(overlap):
                safe_prefix = prefix if prefix.endswith("_") else f"{prefix or f'input{index}'}_"
                collision_map = {
                    column: f"{safe_prefix}{column}" for column in overlap
                }
                subset = subset.rename(columns=collision_map)
                log.warning(
                    "Detected overlapping feature columns in %s; auto-renamed %d columns.",
                    table_path,
                    len(collision_map),
                )
            merged = merged.join(subset, how="outer")
            label_series = label_series.combine_first(frame.set_index(subj_col)[lbl_col])

    if merged is None or subject_id_col is None or label_col is None or label_series is None:
        raise ValueError("Failed to assemble a feature table from config.input.")

    common_index = merged.index.intersection(label_series.index)
    merged = merged.loc[common_index].reset_index()
    merged[label_col] = label_series.loc[common_index].to_numpy()

    feature_columns = tuple(
        column
        for column in merged.columns
        if column not in {subject_id_col, label_col}
    )
    if not feature_columns:
        raise ValueError("No feature columns remain after assembling input tables.")
    return FeatureTable(
        frame=merged,
        id_columns=(subject_id_col,),
        feature_columns=feature_columns,
        outcome=BinaryOutcome(column=label_col, positive_label=1),
        provenance=Provenance.source("recipes.run_from_yaml"),
    )


def _checkpoint_root_for(config: Any, *, predict: bool) -> Path:
    """Resolve the checkpoint directory for a train or predict run."""
    if config.checkpoint_dir:
        return Path(config.checkpoint_dir)
    dirname = _PREDICT_CHECKPOINT_DIRNAME if predict else _TRAIN_CHECKPOINT_DIRNAME
    return Path(config.out_dir) / dirname


def _checkpoint_store_for(
    config: Any,
    *,
    spec: HabitatSpec,
    predict: bool,
) -> CheckpointStore:
    """Attach the checkpoint store for a train or predict run."""
    clustering_mode = (
        config.habitat_segmentation.clustering_mode
        if getattr(config, "habitat_segmentation", None) is not None
        else None
    )
    return CheckpointStore(
        _checkpoint_root_for(config, predict=predict),
        run_fingerprint=spec.fingerprint(),
        strict=bool(getattr(config, "strict_checkpoint_hash", False)),
        clustering_mode=clustering_mode,
    )


def _spec_modalities(spec: HabitatSpec) -> Tuple[str, ...]:
    """Return modality names from the translated spec."""
    raw = spec.voxel_feature_extractor.params.get("modalities") or ()
    return tuple(str(modality) for modality in raw)


def _load_habitat_cohort(
    config: Any,
    spec: HabitatSpec,
    *,
    logger: Optional[logging.Logger],
) -> Cohort:
    """Assemble the cohort from ``data_dir`` (directory layout or manifest)."""
    log = logger or logging.getLogger(__name__)
    modalities = _spec_modalities(spec)
    if not modalities:
        raise ValueError(
            "The translated spec names no modalities; cannot assemble a cohort."
        )
    roi = modalities[0]
    data_dir = Path(config.data_dir)
    if data_dir.is_dir():
        return DirectoryDataSource(data_dir, modalities=modalities, roi=roi).load()
    if data_dir.is_file() and data_dir.suffix.lower() in (".yaml", ".yml"):
        return _cohort_from_manifest(
            data_dir, modalities=modalities, roi=roi, logger=log
        )
    raise ValueError(f"Data path not found: {data_dir}")


def _cohort_from_manifest(
    manifest_path: Path,
    *,
    modalities: Tuple[str, ...],
    roi: str,
    logger: logging.Logger,
) -> Cohort:
    """Build a cohort from a v0.1 ``file_*.yaml`` input manifest."""
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Input manifest {manifest_path} is not a mapping payload.")
    images = payload.get("images") or {}
    masks = payload.get("masks") or {}
    auto_select = bool(payload.get("auto_select_first_file", False))
    base_dir = manifest_path.parent

    subjects: List[Subject] = []
    skipped: List[str] = []
    for subject_id, by_modality in images.items():
        refs, missing = _subject_image_refs(
            by_modality, modalities, base_dir, auto_select
        )
        if missing:
            skipped.append(f"{subject_id} (missing {missing})")
            continue
        mask_file = _subject_mask_file(
            masks.get(subject_id), roi, base_dir, auto_select
        )
        if mask_file is None:
            skipped.append(f"{subject_id} (missing mask for roi={roi!r})")
            continue
        mask_ref = FileImageRef(mask_file, is_mask=True, role_name=roi)
        subjects.append(
            Subject(subject_id=str(subject_id), images=refs, mask=mask_ref)
        )

    if skipped:
        logger.warning(
            "Skipped %d manifest subject(s): %s",
            len(skipped),
            "; ".join(skipped),
        )
    if not subjects:
        raise ValueError(
            f"Input manifest {manifest_path} yielded no usable subjects for "
            f"modalities {list(modalities)}."
        )
    return Cohort(subjects)


def _subject_image_refs(
    by_modality: Any,
    modalities: Tuple[str, ...],
    base_dir: Path,
    auto_select: bool,
) -> Tuple[Dict[str, FileImageRef], List[str]]:
    """Resolve one subject's image entries from a manifest."""
    refs: Dict[str, FileImageRef] = {}
    missing: List[str] = []
    source = by_modality if isinstance(by_modality, Mapping) else {}
    for modality in modalities:
        resolved = _resolve_manifest_entry(source.get(modality), base_dir, auto_select)
        if resolved is None:
            missing.append(modality)
        else:
            refs[modality] = FileImageRef(resolved, is_mask=False, role_name=modality)
    return refs, missing


def _subject_mask_file(
    mask_entry: Any, roi: str, base_dir: Path, auto_select: bool
) -> Optional[Path]:
    """Resolve one subject's ROI mask from a manifest ``masks`` entry."""
    if isinstance(mask_entry, Mapping):
        candidate = mask_entry.get(roi)
        if candidate is None and mask_entry:
            candidate = next(iter(mask_entry.values()))
    else:
        candidate = mask_entry
    return _resolve_manifest_entry(candidate, base_dir, auto_select)


def _resolve_manifest_entry(
    entry: Any, base_dir: Path, auto_select: bool
) -> Optional[Path]:
    """Resolve one manifest path entry to an existing file."""
    if entry is None:
        return None
    path = Path(str(entry))
    if not path.is_absolute():
        path = base_dir / path
    if path.is_file():
        return path
    if path.is_dir() and auto_select:
        candidates = sorted(
            item
            for item in path.iterdir()
            if item.is_file() and not item.name.startswith(".")
        )
        return candidates[0] if candidates else None
    return None


def _save_habitat_result(result: StudyResult, config: Any) -> None:
    """Persist a study result with the v0.1 reporting switches honoured."""
    result.save(
        config.out_dir,
        table_format=config.habitats_results_format,
        write_maps=config.save_images,
        write_units_table=config.save_results_csv,
        write_cluster_plots=config.save_images,
        write_cluster_plots_3d=config.plot_curves,
        write_interactive_cluster_plots=config.plot_curves,
    )


def _save_model_result(
    result: ModelResult,
    table: FeatureTable,
    config: Any,
    *,
    logger: Optional[logging.Logger],
) -> None:
    """Write hold-out train artefacts with v0.1-friendly filenames."""
    import json

    out_dir = Path(config.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = result.pipeline.model.spec.name
    if result.test_metrics is not None:
        metrics_payload: Dict[str, Any] = {
            "train": dict(result.train_metrics),
            "test": dict(result.test_metrics),
        }
        legacy_payload: Dict[str, Any] = {
            "train": {model_name: {"metrics": dict(result.train_metrics)}},
            "test": {model_name: {"metrics": dict(result.test_metrics)}},
        }
    else:
        metrics_payload = dict(result.train_metrics)
        legacy_payload = {
            "train": {model_name: {"metrics": dict(result.train_metrics)}}
        }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    result.manifest.to_json(out_dir / "run_manifest.json")
    (out_dir / "ml_standard_results.json").write_text(
        json.dumps(legacy_payload, indent=2), encoding="utf-8"
    )

    selected = list(result.pipeline.transform(table).feature_columns)
    (out_dir / "selected_features.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )

    if config.is_save_model:
        pipeline_path = result.pipeline.save(out_dir / "model.habitpipeline")
        log = logger or logging.getLogger(__name__)
        log.info("Saved fitted pipeline to %s", pipeline_path)

    _write_all_prediction_results(result, table, out_dir, logger=logger)
    _write_ml_figures(
        result, table, config, out_dir=out_dir, mode="holdout", logger=logger
    )


def _write_all_prediction_results(
    result: ModelResult,
    table: FeatureTable,
    out_dir: Path,
    *,
    logger: Optional[logging.Logger],
) -> None:
    """Write ``all_prediction_results.csv`` for compare / DeLong (v0.1 compat).

    Only emitted when a hold-out split ran; mirrors the v0.1 report writer
    column layout (subject_id, label, dataset, {Model}_prob, {Model}_pred).
    """
    if result.test_metrics is None or not result.train_row_ids:
        return
    import pandas as pd
    from habit.recipes.modeling import predict_model, _select_rows, _row_ids

    model_name = result.pipeline.model.spec.name
    row_ids = _row_ids(table)
    id_to_index = {rid: idx for idx, rid in enumerate(row_ids)}
    train_idx = [id_to_index[rid] for rid in result.train_row_ids]
    test_idx = [id_to_index[rid] for rid in result.test_row_ids]
    subj_col = table.id_columns[0]
    label_col = table.outcome.column if table.outcome is not None else "label"

    train_table = _select_rows(table, train_idx)
    test_table = _select_rows(table, test_idx)
    train_pred = predict_model(result.pipeline, train_table)
    test_pred = predict_model(result.pipeline, test_table)

    train_df = train_table.frame[[subj_col, label_col]].copy()
    train_df.columns = ["subject_id", "label"]
    train_df["dataset"] = "train"
    test_df = test_table.frame[[subj_col, label_col]].copy()
    test_df.columns = ["subject_id", "label"]
    test_df["dataset"] = "test"

    pos_idx = 1
    train_df[f"{model_name}_pred"] = train_pred.predictions.to_numpy()
    test_df[f"{model_name}_pred"] = test_pred.predictions.to_numpy()
    if (
        train_pred.probabilities is not None
        and train_pred.probabilities.shape[1] > pos_idx
    ):
        train_df[f"{model_name}_prob"] = (
            train_pred.probabilities.iloc[:, pos_idx].to_numpy()
        )
        test_df[f"{model_name}_prob"] = (
            test_pred.probabilities.iloc[:, pos_idx].to_numpy()
        )

    all_df = pd.concat([train_df, test_df], ignore_index=True)
    destination = out_dir / "all_prediction_results.csv"
    all_df.to_csv(destination, index=False)
    log = logger or logging.getLogger(__name__)
    log.info("Saved hold-out predictions to %s", destination)


def _save_cv_result(
    result: CVResult,
    config: Any,
    *,
    table: Optional[FeatureTable] = None,
    logger: Optional[logging.Logger],
) -> None:
    """Write cross-validation artefacts with v0.1-friendly filenames."""
    import json

    out_dir = Path(config.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "mean": dict(result.mean_metrics),
        "std": dict(result.std_metrics),
        "folds": [dict(panel) for panel in result.fold_metrics],
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    result.manifest.to_json(out_dir / "run_manifest.json")

    model_name = result.manifest.spec_payload["classifier"]["name"]
    legacy_payload = {
        "aggregated": {model_name: dict(result.mean_metrics)},
        "folds": [{model_name: dict(panel)} for panel in result.fold_metrics],
    }
    (out_dir / "ml_kfold_results.json").write_text(
        json.dumps(legacy_payload, indent=2), encoding="utf-8"
    )

    log = logger or logging.getLogger(__name__)
    if config.is_save_model and result.pipelines:
        model_dir = out_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        for fold_index, pipeline in enumerate(result.pipelines):
            destination = (
                model_dir / f"{model_name}_fold{fold_index}_pipeline.habitpipeline"
            )
            pipeline.save(destination)
            log.info("Saved fold %d pipeline to %s", fold_index, destination)

    if table is not None:
        _write_ml_figures(
            result, table, config, out_dir=out_dir, mode="cv", logger=log
        )


def _write_ml_figures(
    result: Union[ModelResult, CVResult],
    table: FeatureTable,
    config: Any,
    *,
    out_dir: Path,
    mode: str,
    logger: Optional[logging.Logger],
) -> None:
    """Persist v1 ML figures under ``out_dir/visualizations`` when enabled."""
    from habit.recipes.ml_reporting import write_ml_figures_from_config

    log = logger or logging.getLogger(__name__)
    figure_dir = out_dir / "visualizations"
    paths = write_ml_figures_from_config(
        result,
        table,
        config,
        destination=figure_dir,
        mode=mode,
        logger=log,
    )
    if paths:
        log.info("Wrote %d ML figure(s) under %s", len(paths), figure_dir)


def _is_v1_model_archive(path: Path) -> bool:
    """Return whether the artefact is a v1 zip archive vs a v0.1 pickle."""
    try:
        with path.open("rb") as handle:
            return handle.read(4) == _ZIP_MAGIC
    except OSError:
        return False
