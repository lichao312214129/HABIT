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

"""Habitat analysis command implementation.

L5 wiring only (phase-4 layout): the command parses the v0.1 YAML with
the v0.1 schema, translates it into the v1 document model through
``LegacyConfigAdapter``, assembles cohort/execution backend/recipe, and
hands the study result to the directory writer. No algorithm lives here;
the only module-level ``habit.core`` dependency left is the v0.1 config
*schema*, which is the YAML parsing contract this command must honour.

Checkpoint strategy (stage-5): the v1 ``CheckpointStore`` is wired into
both the train and predict paths at the v0.1 locations
(``.habitat_checkpoint`` / ``.habitat_predict_checkpoint`` under
``out_dir``, or ``checkpoint_dir`` when set), and the recipes receive it
as an assembly argument. Cache keys incorporate the spec fingerprint
(units and one-step stages) or the fitted model's ``model_id`` (label
stage), so a changed spec or definition never reads a stale entry.
A legacy v0.1 ``manifest.json`` / ``subjects/`` tree is auto-migrated on
open (logged), then resume continues. ``strict_checkpoint_hash=True``
raises :class:`~habit.exceptions.CompatibilityError` only for fingerprint
mismatch or a corrupt/unreadable legacy manifest. Reads honour
``resume``; writes always happen, matching v0.1.

Predict compatibility: models fitted by this command are saved as v1
``.habitatmodel`` archives only. Legacy v0.1 raw-pickle pipelines are
rejected with a migration message pointing at ``habitat_model.habitatmodel``
and :func:`habit.recipes.apply_habitat_model`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import click
import yaml

from habit.adapters.directory import DirectoryDataSource
from habit.adapters.image_refs import FileImageRef
from habit.api.habitat import apply_habitat_cli_overrides
from habit.commands.common import echo_error, echo_success, load_config_or_exit
from habit.contracts.habitat import HabitatModel
from habit.contracts.subject import Cohort, Subject
from habit.exceptions import (
    ComponentNotFoundError,
    ConfigurationError,
    DataFormatError,
    HABITAPIError,
)
from habit.schemas import HabitatAnalysisConfig
from habit.execution.checkpoint import CheckpointStore
from habit.execution.selection import backend_from_policy
from habit.recipes import (
    apply_habitat_model,
    direct_pooling,
    one_step,
    run_from_yaml,
    two_step,
)
from habit.recipes.result import StudyResult
from habit.spec.legacy import LegacyConfigAdapter, detect_yaml_version
from habit.spec.policy import RunPolicy
from habit.spec.specs import HabitatSpec
from habit.utils.log_utils import setup_logger, stop_queue_listener

#: clustering_mode -> the L4 recipe implementing that study design.
_RECIPE_BY_MODE: Mapping[str, Callable[..., StudyResult]] = {
    "two_step": two_step,
    "one_step": one_step,
    "direct_pooling": direct_pooling,
}

#: v0.1 checkpoint directory names, kept identical so a user's existing
#: mental model (and cleanup scripts) still apply; the payload format is
#: the v1 store's, never v0.1's manifest/joblib layout.
_TRAIN_CHECKPOINT_DIRNAME = ".habitat_checkpoint"
_PREDICT_CHECKPOINT_DIRNAME = ".habitat_predict_checkpoint"

#: v1 fitted model file name written under ``out_dir`` after train.
_V1_MODEL_NAME = "habitat_model.habitatmodel"

#: Zip local-file magic: v1 ``.habitatmodel`` artefacts are zip archives.
_ZIP_MAGIC = b"PK\x03\x04"

_LEGACY_PICKLE_MESSAGE = (
    "Legacy v0.1 pickle pipelines are not supported in HABIT v1.0. "
    f"Train a model to produce {_V1_MODEL_NAME!r}, then run predict with "
    "pipeline_path pointing at that archive or call "
    "habit.recipes.apply_habitat_model in Python."
)

#: Exceptions that reflect user input or data-layout problems rather than
#: internal defects. The CLI prints these without a Python traceback.
_USER_FACING_ERRORS: Tuple[type, ...] = (
    ValueError,
    DataFormatError,
    HABITAPIError,
    ComponentNotFoundError,
    ConfigurationError,
)


def run_habitat(
    config_file: str,
    debug_mode: bool,
    mode: Optional[str],
    pipeline_path: Optional[str],
    resume: bool = False,
    exit_on_error: bool = True,
) -> None:
    """
    Run habitat analysis pipeline in train or predict mode.

    The command layer only parses and assembles: translation produces the
    spec/policy, ``_load_cohort`` builds the cohort, and the recipe named
    by ``clustering_mode`` computes the study. Persistence goes through
    ``StudyResult.save`` (v1 layout) plus ``habitat_model.habitatmodel``
    for the fitted model.

    Native **v1** documents (``version: '1.0'``) share the same assembly as
    :func:`~habit.recipes.run_from_yaml` so exported YAML from
    :func:`~habit.spec.save_habitat_config` replays voxel-identically.

    Args:
        config_file: Path to configuration YAML file.
        debug_mode: Whether to enable debug mode.
        mode: Override run mode (``train`` or ``predict``).
        pipeline_path: Override pipeline path for prediction.
        resume: Resume train run from individual-level checkpoint.
        exit_on_error: When True (CLI default), call ``sys.exit(1)`` on failure.
            GUI callers should pass False so exceptions propagate.
    """
    config_path = Path(config_file)
    if not config_path.is_file():
        echo_error(f"Error: Configuration file not found: {config_file}")
        if exit_on_error:
            sys.exit(1)
        raise FileNotFoundError(config_file)

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        echo_error(f"Error: Failed to load configuration: {exc}")
        if exit_on_error:
            sys.exit(1)
        raise

    if isinstance(payload, Mapping) and detect_yaml_version(payload) == "v1":
        _run_habitat_v1_document(
            config_path,
            payload,
            debug_mode=debug_mode,
            mode=mode,
            pipeline_path=pipeline_path,
            resume=resume,
            exit_on_error=exit_on_error,
        )
        return

    config = load_config_or_exit(HabitatAnalysisConfig, config_file)
    click.echo(f"Loaded configuration from: {config_file}")

    apply_habitat_cli_overrides(
        config,
        mode=mode,
        pipeline_path=pipeline_path,
        debug=debug_mode,
        resume=resume,
    )

    output_path = Path(config.out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if config.debug else logging.INFO
    logger = setup_logger(
        name="cli.habitat",
        output_dir=output_path,
        log_filename="habitat_analysis.log",
        level=log_level,
    )

    logger.info("==== Starting Habitat Analysis ====")
    logger.info("Config file: %s", config_file)
    logger.info("Full configuration: %s", config.model_dump())
    logger.info("=====================================")

    click.echo("Starting habitat analysis...")
    click.echo(f"  Mode: {config.run_mode}")
    click.echo(f"  Output directory: {config.out_dir}")
    if config.run_mode == "predict":
        click.echo(f"  Pipeline path: {config.pipeline_path or 'auto'}")
    if config.resume:
        click.echo(
            "  Resume: enabled (checkpoint: "
            f"{_checkpoint_root_for(config, predict=config.run_mode == 'predict')})"
        )
    else:
        click.echo("  Resume: disabled (checkpoint is written, never read)")
    click.echo(f"  Log file at: {output_path / 'habitat_analysis.log'}")

    try:
        if config.run_mode == "predict":
            _run_predict(config, logger)
        else:
            _run_train(config, logger)
        logger.info("Habitat analysis completed successfully")
        echo_success("Habitat analysis completed successfully!")
    except _USER_FACING_ERRORS as exc:
        logger.error("Error during habitat analysis: %s", exc)
        echo_error(f"Error: {exc}")
        if exit_on_error:
            sys.exit(1)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Unexpected error during habitat analysis: %s", exc, exc_info=True
        )
        echo_error(
            "Error: An unexpected internal error occurred during habitat analysis.\n"
            f"Details: {exc}\n"
            "If this persists, please report a bug and attach habitat_analysis.log "
            "from your output directory."
        )
        if exit_on_error:
            sys.exit(1)
        raise
    finally:
        stop_queue_listener()


def _run_habitat_v1_document(
    config_path: Path,
    payload: Mapping[str, Any],
    *,
    debug_mode: bool,
    mode: Optional[str],
    pipeline_path: Optional[str],
    resume: bool,
    exit_on_error: bool,
) -> None:
    """
    Execute a native v1 habitat document through ``run_from_yaml``.

    CLI overrides are applied to an in-memory copy written beside the original
    path only when needed; scientific assembly stays identical to the YAML API.

    Args:
        config_path: Path to the source YAML file.
        payload: Parsed v1 document mapping.
        debug_mode: Enable debug logging.
        mode: Optional ``train`` / ``predict`` override.
        pipeline_path: Optional fitted-model path override.
        resume: When True, force ``policy.resume`` on.
        exit_on_error: Exit the process on user-facing failures.
    """
    document = dict(payload)
    if mode is not None:
        document["mode"] = str(mode).strip().lower()
    if pipeline_path is not None:
        document["pipeline"] = str(pipeline_path)
    if resume:
        policy = dict(document.get("policy") or {})
        policy["resume"] = True
        document["policy"] = policy

    output = dict(document.get("output") or {})
    out_dir = Path(str(output.get("out_dir") or "."))
    out_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if debug_mode else logging.INFO
    logger = setup_logger(
        name="cli.habitat",
        output_dir=out_dir,
        log_filename="habitat_analysis.log",
        level=log_level,
    )

    run_mode = str(document.get("mode", "train")).strip().lower()
    click.echo(f"Loaded v1 configuration from: {config_path}")
    click.echo("Starting habitat analysis...")
    click.echo(f"  Mode: {run_mode}")
    click.echo(f"  Output directory: {out_dir}")
    if run_mode == "predict":
        click.echo(f"  Pipeline path: {document.get('pipeline') or 'auto'}")
    click.echo(f"  Log file at: {out_dir / 'habitat_analysis.log'}")
    logger.info("==== Starting Habitat Analysis (v1 document) ====")
    logger.info("Config file: %s", config_path)
    logger.info("Document: %s", document)

    # Materialise CLI overrides (if any) so run_from_yaml reads one file and
    # keeps a single assembly path with the YAML API.
    effective_path = config_path
    if document != payload:
        effective_path = out_dir / "_cli_effective_habitat_config.yaml"
        effective_path.write_text(
            yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.info("Wrote CLI-overridden effective config: %s", effective_path)

    try:
        run_from_yaml(
            effective_path,
            workflow="habitat",
            save=True,
            logger=logger,
        )
        logger.info("Habitat analysis completed successfully")
        echo_success("Habitat analysis completed successfully!")
    except _USER_FACING_ERRORS as exc:
        logger.error("Error during habitat analysis: %s", exc)
        echo_error(f"Error: {exc}")
        if exit_on_error:
            sys.exit(1)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Unexpected error during habitat analysis: %s", exc, exc_info=True
        )
        echo_error(
            "Error: An unexpected internal error occurred during habitat analysis.\n"
            f"Details: {exc}\n"
            "If this persists, please report a bug and attach habitat_analysis.log "
            "from your output directory."
        )
        if exit_on_error:
            sys.exit(1)
        raise
    finally:
        stop_queue_listener()


def _run_train(config: HabitatAnalysisConfig, logger: logging.Logger) -> None:
    """
    Fit a habitat model through the v1 recipe named by ``clustering_mode``.

    Args:
        config: Validated v0.1 habitat configuration.
        logger: Run logger.
    """
    document = _translate_document(config)
    spec_payload = document.get("spec")
    if spec_payload is None:
        raise ValueError(
            "The train config translated to no habitat spec; train configs "
            "must carry feature_construction and habitat_segmentation blocks."
        )
    spec = HabitatSpec.from_dict(spec_payload)
    backend = backend_from_policy(RunPolicy.from_dict(document.get("policy") or {}))
    cohort = _load_cohort(config, spec, logger)
    checkpoint = _checkpoint_store_for(config, spec=spec, predict=False)
    _log_checkpoint_strategy(config, checkpoint.root, logger)

    mode = str(config.habitat_segmentation.clustering_mode)
    recipe = _RECIPE_BY_MODE.get(mode)
    if recipe is None:
        raise ValueError(
            f"Unknown clustering_mode {mode!r}; expected one of {sorted(_RECIPE_BY_MODE)}."
        )
    logger.info(
        "Running v1 recipe %s (clustering_mode=%s)",
        getattr(recipe, "__name__", mode),
        mode,
    )
    result = recipe(cohort, spec, backend=backend, checkpoint=checkpoint)
    _save_result(result, config)

    if result.habitat_model is None:
        # one_step fits one model per subject; the v1 per-subject model
        # persistence layout is deliberately undefined for now (stage-4
        # decision 5), so no pipeline artefact is written here.
        logger.info(
            "one_step produces per-subject models; v1 per-subject model "
            "persistence is not yet defined (stage-5), so no pipeline "
            "artefact was written."
        )
        return
    logger.info(
        "Saved the fitted model to %s",
        Path(config.out_dir) / _V1_MODEL_NAME,
    )


def _run_predict(config: HabitatAnalysisConfig, logger: logging.Logger) -> None:
    """
    Apply a fitted habitat model, routing by the artefact's format.

    v1 ``.habitatmodel`` archives go through
    :func:`habit.recipes.apply_habitat_model`. Legacy v0.1 raw pickles are
    rejected with a migration message.

    Args:
        config: Validated v0.1 habitat configuration.
        logger: Run logger.
    """
    if not config.pipeline_path:
        raise ValueError(
            "pipeline_path is required for predict mode (set it in the YAML "
            "or pass --pipeline-path)."
        )
    pipeline_file = Path(config.pipeline_path)
    if not pipeline_file.is_file():
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_file}")
    if not _is_v1_model_archive(pipeline_file):
        raise ValueError(
            f"{_LEGACY_PICKLE_MESSAGE} Got: {pipeline_file}"
        )

    model = HabitatModel.load(pipeline_file)
    document = _translate_document(config)
    spec_payload = document.get("spec")
    if spec_payload is None:
        # Stub predict YAMLs (pipeline_path only) carry the definition inside
        # the v1 model archive rather than duplicating feature_construction.
        spec_payload = model.spec_payload
    spec = HabitatSpec.from_dict(spec_payload)
    backend = backend_from_policy(RunPolicy.from_dict(document.get("policy") or {}))
    cohort = _load_cohort(config, spec, logger)
    checkpoint = _checkpoint_store_for(config, spec=spec, predict=True)
    _log_checkpoint_strategy(config, checkpoint.root, logger)
    logger.info(
        "Applying v1 habitat model %s (id=%s, n_habitats=%d, produced_by=%s)",
        pipeline_file,
        model.model_id,
        model.n_habitats,
        model.provenance.produced_by,
    )
    result = apply_habitat_model(cohort, spec, model, backend=backend, checkpoint=checkpoint)
    _save_result(result, config)


def _translate_document(config: HabitatAnalysisConfig) -> Dict[str, Any]:
    """
    Translate the validated v0.1 config into the v1 document model.

    Args:
        config: Validated v0.1 habitat configuration.

    Returns:
        The translated document with ``spec``/``policy``/``data`` sections.
    """
    document: Dict[str, Any] = (
        LegacyConfigAdapter().translate(config.model_dump(), "habitat").document
    )
    return document


def _checkpoint_root_for(config: HabitatAnalysisConfig, *, predict: bool) -> Path:
    """
    Resolve the checkpoint directory for a train or predict run.

    Args:
        config: Validated v0.1 habitat configuration.
        predict: Whether the run is a predict run.

    Returns:
        ``checkpoint_dir`` when set, else the v0.1 default location under
        ``out_dir``.
    """
    if config.checkpoint_dir:
        return Path(config.checkpoint_dir)
    dirname = _PREDICT_CHECKPOINT_DIRNAME if predict else _TRAIN_CHECKPOINT_DIRNAME
    return Path(config.out_dir) / dirname


def _checkpoint_store_for(
    config: HabitatAnalysisConfig,
    *,
    spec: HabitatSpec,
    predict: bool,
) -> CheckpointStore:
    """
    Attach the checkpoint store for a train or predict run.

    The store is attached unconditionally: reads honour the backend's
    ``resume`` flag, but a run always records its outcomes so a later
    resumed run can skip them (v0.1 behaviour). Bound to
    ``spec.fingerprint()`` so ``strict_checkpoint_hash`` can raise on
    incompatible resumes.

    Args:
        config: Validated v0.1 habitat configuration.
        spec: Effective habitat spec whose fingerprint scopes the store.
        predict: Whether the run is a predict run.

    Returns:
        The store rooted at :func:`_checkpoint_root_for`.

    Raises:
        CompatibilityError: When ``strict_checkpoint_hash`` is set and the
            on-disk fingerprint is incompatible, or a legacy v0.1 manifest
            is corrupt/unreadable.
    """
    clustering_mode = (
        config.habitat_segmentation.clustering_mode
        if getattr(config, "habitat_segmentation", None) is not None
        else None
    )
    return CheckpointStore(
        _checkpoint_root_for(config, predict=predict),
        run_fingerprint=spec.fingerprint(),
        strict=bool(config.strict_checkpoint_hash),
        clustering_mode=clustering_mode,
    )


def _spec_modalities(spec: HabitatSpec) -> Tuple[str, ...]:
    """
    Return the modality names a spec consumes, in spec order.

    Args:
        spec: Translated habitat spec.

    Returns:
        Modality names from the voxel feature extractor params.
    """
    raw = spec.voxel_feature_extractor.params.get("modalities") or ()
    return tuple(str(modality) for modality in raw)


def _load_cohort(
    config: HabitatAnalysisConfig, spec: HabitatSpec, logger: logging.Logger
) -> Cohort:
    """
    Assemble the cohort from ``data_dir`` (directory layout or manifest).

    Args:
        config: Validated v0.1 habitat configuration.
        spec: Translated habitat spec (provides the modality list).
        logger: Run logger for skip warnings.

    Returns:
        The ordered cohort the recipe will run on.
    """
    modalities = _spec_modalities(spec)
    if not modalities:
        raise ValueError(
            "The translated spec names no modalities "
            "(feature_construction.modalities is empty); cannot assemble a cohort."
        )
    # v0.1's feature service used the first mask of the resolved mask map
    # (``list(mask_path_map.values())[0]``), which for both supported data
    # sources is the first modality's mask; the golden baselines pin this.
    roi = modalities[0]
    data_dir = Path(config.data_dir)
    if data_dir.is_dir():
        return DirectoryDataSource(data_dir, modalities=modalities, roi=roi).load()
    if data_dir.is_file() and data_dir.suffix.lower() in (".yaml", ".yml"):
        return _cohort_from_manifest(
            data_dir, modalities=modalities, roi=roi, logger=logger
        )
    raise ValueError(f"Data path not found: {data_dir}")


def _cohort_from_manifest(
    manifest_path: Path,
    *,
    modalities: Tuple[str, ...],
    roi: str,
    logger: logging.Logger,
) -> Cohort:
    """
    Build a cohort from a v0.1 ``file_*.yaml`` input manifest.

    Manifest parsing lives here, not in ``habit.adapters``: the manifest is
    a v0.1 *configuration* artefact (relative paths, auto-select flag), and
    L1 adapters must stay free of configuration concepts.

    Args:
        manifest_path: Path to the input manifest YAML.
        modalities: Modality names required per subject, in spec order.
        roi: Mask key identifying the region of interest.
        logger: Run logger for skip warnings.

    Returns:
        The ordered cohort, manifest order preserved (v0.1 behaviour).
    """
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
        mask_file = _subject_mask_file(
            masks.get(subject_id), roi, base_dir, auto_select
        )
        if mask_file is None:
            missing.append(f"mask {roi!r}")
        if missing:
            skipped.append(f"{subject_id} (missing: {', '.join(missing)})")
            continue
        subjects.append(
            Subject(
                subject_id=str(subject_id),
                images=refs,
                masks={roi: FileImageRef(mask_file, is_mask=True, role_name=roi)},
            )
        )
    if skipped:
        logger.warning(
            "Manifest subjects skipped (incomplete inputs): %s", "; ".join(skipped)
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
    """
    Resolve one subject's image entries from a manifest.

    Args:
        by_modality: The subject's ``images`` mapping (or anything else,
            which is treated as "no entries").
        modalities: Required modality names, in spec order.
        base_dir: Directory relative paths resolve against.
        auto_select: Manifest ``auto_select_first_file`` flag.

    Returns:
        ``(refs, missing)``: resolved image refs per modality and the
        names of modalities that could not be resolved.
    """
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
    """
    Resolve one subject's ROI mask from a manifest ``masks`` entry.

    v0.1 used the first mask in the resolved mask map; prefer the entry
    keyed by ``roi`` and fall back to the first mapping value so manifests
    whose mask keys differ from modality names keep the v0.1 behaviour.

    Args:
        mask_entry: The subject's ``masks`` value (mapping or plain path).
        roi: Mask key identifying the region of interest.
        base_dir: Directory relative paths resolve against.
        auto_select: Manifest ``auto_select_first_file`` flag.

    Returns:
        The resolved mask file, or ``None`` when unresolvable.
    """
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
    """
    Resolve one manifest path entry to an existing file.

    Entries may point at a file directly or at a directory; a directory
    yields its first sorted non-hidden file only when the manifest sets
    ``auto_select_first_file: true`` (v0.1 semantics). Relative paths
    resolve against the manifest's own directory, matching
    ``habit.core.common.io_utils``.

    Args:
        entry: Raw manifest value (path string, or anything else).
        base_dir: Directory relative paths resolve against.
        auto_select: Manifest ``auto_select_first_file`` flag.

    Returns:
        The resolved file, or ``None`` when unresolvable.
    """
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


def _log_checkpoint_strategy(
    config: HabitatAnalysisConfig, checkpoint_root: Path, logger: logging.Logger
) -> None:
    """
    Log the checkpoint wiring actually in effect.

    Fingerprint binding and legacy-layout checks run inside
    :class:`~habit.execution.CheckpointStore` (raising when
    ``strict_checkpoint_hash`` is set). This helper only records the active
    resume knobs for operators.

    Args:
        config: Validated v0.1 habitat configuration.
        checkpoint_root: Resolved checkpoint directory for this run.
        logger: Run logger.
    """
    logger.info(
        "Checkpoint store: %s (v1 format; resume=%s, retry_failed=%s, "
        "force_rerun=%s, clear_on_success=%s, strict_hash=%s)",
        checkpoint_root,
        config.resume,
        config.retry_failed_subjects,
        tuple(config.force_rerun_subjects),
        config.clear_checkpoint_on_success,
        config.strict_checkpoint_hash,
    )


def _save_result(result: StudyResult, config: HabitatAnalysisConfig) -> None:
    """
    Persist a study result with the v0.1 reporting switches honoured.

    Args:
        result: The study result a recipe produced.
        config: Validated v0.1 habitat configuration.
    """
    result.save(
        config.out_dir,
        table_format=config.habitats_results_format,
        write_maps=config.save_images,
        write_units_table=config.save_results_csv,
        write_cluster_plots=config.save_images,
        write_cluster_plots_3d=config.plot_curves,
        write_interactive_cluster_plots=config.plot_curves,
    )


def _is_v1_model_archive(path: Path) -> bool:
    """
    Return whether the artefact is a v1 archive (zip) vs a v0.1 pickle.

    Args:
        path: Pipeline artefact path.

    Returns:
        True when the file starts with the zip local-file magic.
    """
    try:
        with path.open("rb") as handle:
            return handle.read(4) == _ZIP_MAGIC
    except OSError:
        return False
