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
"""Build and persist runnable v1 habitat configuration documents.

A pure-Python analysis (``HabitatSpec`` + ``RunPolicy`` + data/output paths)
must be exportable as a **complete effective** YAML document so that
:func:`~habit.recipes.run_from_yaml` and ``habit get-habitat --config`` replay
the same recipe assembly with voxel-identical habitats.

This module only assembles the document mapping; it never runs the analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from habit.exceptions import HABITAPIError
from habit.spec.legacy import V1_SCHEMA_VERSION, validate_v1_document
from habit.spec.policy import RunPolicy
from habit.spec.specs import HabitatSpec
from habit.spec.yaml_io import _read_yaml, _write_yaml

__all__ = [
    "build_habitat_document",
    "save_habitat_config",
    "load_habitat_config",
]


def build_habitat_document(
    spec: HabitatSpec,
    *,
    data_source: Union[str, Path],
    out_dir: Union[str, Path],
    policy: Optional[RunPolicy] = None,
    mode: str = "train",
    pipeline: Optional[Union[str, Path]] = None,
    save_images: bool = True,
    save_results_csv: bool = True,
    habitats_results_format: str = "parquet",
    plot_curves: bool = True,
) -> Dict[str, Any]:
    """
    Assemble a native v1 habitat document with expanded defaults.

    The ``spec`` section uses :meth:`HabitatSpec.to_effective_dict` so omitted
    fingerprint-stable defaults (geometry policy, empty postprocess slots)
    appear explicitly. The ``policy`` section always emits every
    :class:`RunPolicy` field (caller policy or library defaults).

    Args:
        spec: Analysis declaration (what to compute).
        data_source: Cohort root directory or input-manifest YAML path.
        out_dir: Destination directory for CLI / ``run_from_yaml(..., save=True)``.
        policy: Execution policy; ``None`` expands to :class:`RunPolicy` defaults.
        mode: ``"train"`` or ``"predict"``.
        pipeline: Fitted ``.habitatmodel`` path for predict mode.
        save_images: Persist NRRD habitat (and supervoxel) maps.
        save_results_csv: Persist the habitats unit table.
        habitats_results_format: ``"parquet"`` or ``"csv"``.
        plot_curves: Persist clustering visualisation artefacts.

    Returns:
        A v1 document mapping ready for YAML serialisation.

    Raises:
        HABITAPIError: When the assembled document fails structural validation.
    """
    if not isinstance(spec, HabitatSpec):
        raise HABITAPIError(
            "build_habitat_document expects a HabitatSpec; "
            f"got {type(spec).__name__}."
        )
    run_mode = str(mode).strip().lower()
    if run_mode not in ("train", "predict"):
        raise HABITAPIError(
            f"build_habitat_document mode must be 'train' or 'predict'; got {mode!r}."
        )
    effective_policy = policy if policy is not None else RunPolicy()
    if not isinstance(effective_policy, RunPolicy):
        raise HABITAPIError(
            "build_habitat_document policy must be a RunPolicy; "
            f"got {type(effective_policy).__name__}."
        )

    # Resolve stages-first specs before serialising so the written document
    # carries roles on every stage and sugar named fields (modalities, etc.).
    # Lazy import keeps document assembly free of a hard domain edge at import.
    from habit.domain.stages import ensure_habitat_spec_resolved

    effective_spec = ensure_habitat_spec_resolved(spec)

    document: Dict[str, Any] = {
        "version": V1_SCHEMA_VERSION,
        "workflow": "habitat",
        "mode": run_mode,
        "spec": effective_spec.to_effective_dict(),
        "data": {"source": str(data_source)},
        "policy": effective_policy.to_dict(),
        "output": {
            "out_dir": str(out_dir),
            "save_images": bool(save_images),
            "save_results_csv": bool(save_results_csv),
            "habitats_results_format": str(habitats_results_format),
            "plot_curves": bool(plot_curves),
        },
    }
    if pipeline is not None:
        document["pipeline"] = str(pipeline)
    elif run_mode == "predict":
        document["pipeline"] = None

    validate_v1_document(document, workflow="habitat")
    return document


def save_habitat_config(
    path: Union[str, Path],
    spec: HabitatSpec,
    *,
    data_source: Union[str, Path],
    out_dir: Union[str, Path],
    policy: Optional[RunPolicy] = None,
    mode: str = "train",
    pipeline: Optional[Union[str, Path]] = None,
    save_images: bool = True,
    save_results_csv: bool = True,
    habitats_results_format: str = "parquet",
    plot_curves: bool = True,
) -> Path:
    """
    Write a complete effective v1 habitat YAML for CLI / YAML-API replay.

    Args:
        path: Destination YAML path.
        spec: Analysis declaration.
        data_source: Cohort root or manifest path recorded under ``data.source``.
        out_dir: Output directory recorded under ``output.out_dir``.
        policy: Optional run policy (defaults expanded when omitted).
        mode: ``"train"`` or ``"predict"``.
        pipeline: Predict-mode model archive path.
        save_images: See :func:`build_habitat_document`.
        save_results_csv: See :func:`build_habitat_document`.
        habitats_results_format: See :func:`build_habitat_document`.
        plot_curves: See :func:`build_habitat_document`.

    Returns:
        The written path.
    """
    document = build_habitat_document(
        spec,
        data_source=data_source,
        out_dir=out_dir,
        policy=policy,
        mode=mode,
        pipeline=pipeline,
        save_images=save_images,
        save_results_csv=save_results_csv,
        habitats_results_format=habitats_results_format,
        plot_curves=plot_curves,
    )
    return _write_yaml(document, path)


def load_habitat_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and structurally validate a v1 habitat YAML document.

    Args:
        path: YAML file written by :func:`save_habitat_config` (or equivalent).

    Returns:
        The validated document mapping.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        HABITAPIError: When the file is not a valid habitat v1 document.
    """
    document = _read_yaml(path)
    if not isinstance(document, Mapping):
        raise HABITAPIError(
            f"{path} must contain a YAML mapping at the top level."
        )
    validate_v1_document(document, workflow="habitat")
    return dict(document)
