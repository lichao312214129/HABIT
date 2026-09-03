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
"""L4 image-preprocessing recipes (thin assembly).

Two surfaces, both free of direct ``habit.compat.engines`` imports (architecture
gate): they delegate to :mod:`habit.api.preprocessing`.

* :func:`preprocess_images` — batch directory pipeline (CLI twin).
* :func:`preprocess_subject` / :func:`preprocess_image` — atomic in-memory
  operators for embedding HABIT in a third-party notebook or pipeline.
"""

from __future__ import annotations

import logging
import multiprocessing
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

from habit.adapters.preprocessing_io import PreprocessingIOAdapter, PreprocessingInput
from habit.api.contracts import WorkflowResult, coerce_config
from habit.api.provenance import create_run_manifest, write_run_manifest
from habit.contracts.subject import Subject
from habit.exceptions import HABITAPIError
from habit.schemas.workflows.preprocessing import PreprocessingConfig
from habit.utils.progress_utils import CustomTqdm

if TYPE_CHECKING:
    from habit.contracts.subject import Subject

__all__ = ["preprocess_images", "preprocess_subject", "preprocess_image"]


def preprocess_images(
    config: Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> WorkflowResult[None] | None:
    """
    Run the batch image-preprocessing pipeline (``habit preprocess`` recipe).

    Args:
        config: Validated preprocessing configuration (v0.1 schema object or
            mapping accepted by
            :class:`~habit.schemas.workflows.preprocessing.PreprocessingConfig`).
        logger: Optional run logger forwarded to the workflow helper.

    Returns:
        :class:`~habit.api.contracts.WorkflowResult` with output directory
        metadata and a run manifest path.
    """
    validated_config = coerce_config(config, PreprocessingConfig)
    adapter = PreprocessingIOAdapter(
        source_root=str(validated_config.data_dir),
        destination_root=str(validated_config.out_dir),
        auto_select_first_file=bool(validated_config.auto_select_first_file),
    )
    inputs = adapter.discover()
    if not inputs:
        if logger is not None:
            logger.warning("No valid subjects found")
        return None
    steps = {
        str(name): _model_to_dict(step)
        for name, step in validated_config.preprocessing.items()
    }
    worker_count = min(
        int(validated_config.processes),
        max(1, multiprocessing.cpu_count() - 2),
    )
    tasks = [(item, steps) for item in inputs]
    progress = CustomTqdm(total=len(tasks), desc="Processing subjects")
    try:
        if worker_count == 1:
            results = (_run_subject(task) for task in tasks)
            for subject, snapshots in results:
                _write_subject_outputs(adapter, subject, snapshots, validated_config)
                progress.update(1)
        else:
            with multiprocessing.Pool(processes=worker_count) as pool:
                for subject, snapshots in pool.imap(_run_subject, tasks):
                    _write_subject_outputs(adapter, subject, snapshots, validated_config)
                    progress.update(1)
    finally:
        progress.close()

    manifest = create_run_manifest("preprocess", validated_config)
    manifest_path = write_run_manifest(manifest, validated_config.out_dir)
    return WorkflowResult(
        output_dir=validated_config.out_dir,
        metadata={
            "config_hash": manifest.config_hash,
            "habit_version": manifest.habit_version,
        },
        run_id=manifest.run_id,
        manifest_path=manifest_path,
    )


def _run_subject(
    task: Tuple[PreprocessingInput, Mapping[str, Mapping[str, Any]]],
) -> Tuple[Subject, List[Tuple[str, Sequence[str], Subject]]]:
    """Run configured steps for one lazy subject without touching output files."""
    item, steps = task
    subject = item.subject
    snapshots: List[Tuple[str, Sequence[str], Subject]] = []
    for index, (name, raw_params) in enumerate(steps.items(), start=1):
        params = dict(raw_params)
        modalities = list(params.get("images") or subject.images)
        if not modalities:
            continue
        if name not in {
            "n4_correction",
            "resample",
            "reorientation",
            "zscore_normalization",
        }:
            raise HABITAPIError(
                f"Batch preprocessing step {name!r} is not supported in HABIT v2. "
                "Use n4_correction, resample, reorientation, or "
                "zscore_normalization."
            )
        from habit.api.preprocessing import preprocess_subject

        # Legacy YAML did not implicitly select a mask for preprocessing.
        # Each atomic operator now receives a mask only through an explicit
        # algorithm parameter, preventing a silent change in its definition.
        subject = preprocess_subject(
            subject,
            {name: params},
            auto_select_mask=False,
        )
        snapshots.append((f"{name}_{index:02d}", modalities, subject))
    return subject, snapshots


def _write_subject_outputs(
    adapter: PreprocessingIOAdapter,
    subject: Subject,
    snapshots: Sequence[Tuple[str, Sequence[str], Subject]],
    config: Any,
) -> None:
    """Write optional step snapshots followed by the final legacy output tree."""
    save_options = config.save_options
    selected_names = set(save_options.intermediate_steps)
    if save_options.save_intermediate:
        for stage_name, modalities, snapshot in snapshots:
            step_name = stage_name.rsplit("_", 1)[0]
            if not selected_names or step_name in selected_names:
                adapter.write(snapshot, stage_name=stage_name, modalities=modalities)
    adapter.write(subject)


def _model_to_dict(model: Any) -> Dict[str, Any]:
    """Convert a Pydantic v1/v2 step object to a plain parameter dictionary."""
    if hasattr(model, "model_dump"):
        return dict(model.model_dump())
    return dict(model.dict())


def preprocess_subject(
    subject: "Subject",
    steps: Mapping[str, Mapping[str, Any]],
    *,
    mask_roi: Optional[str] = None,
    broadcast_mask: bool = True,
) -> "Subject":
    """
    Apply an ordered image-preprocessing chain to one subject in memory.

    Recipe twin of :func:`habit.api.preprocessing.preprocess_subject`. See
    that function for full argument documentation.

    Args:
        subject: One imaging subject.
        steps: Ordered ``{step_name: params}`` mapping (YAML ``preprocessing``
            block shape).
        mask_roi: Optional ROI key; auto-selected when the subject has exactly
            one mask.
        broadcast_mask: Attach the ROI under every ``mask_<modality>``.

    Returns:
        A new Subject with processed in-memory volumes.
    """
    from habit.api.preprocessing import preprocess_subject as _api_preprocess_subject

    return _api_preprocess_subject(
        subject,
        steps,
        mask_roi=mask_roi,
        broadcast_mask=broadcast_mask,
    )


def preprocess_image(
    image: "habit.api.image.ImageVolume",
    steps: Mapping[str, Mapping[str, Any]],
    *,
    mask: Optional["habit.api.image.MaskVolume"] = None,
    modality: str = "image",
) -> "habit.api.image.ImageVolume":
    """
    Apply an ordered image-preprocessing chain to one volume in memory.

    Recipe twin of :func:`habit.api.preprocessing.preprocess_image`.

    Args:
        image: Intensity volume to process.
        steps: Ordered ``{step_name: params}`` mapping.
        mask: Optional ROI mask.
        modality: Synthetic modality key used internally.

    Returns:
        The processed intensity volume.
    """
    from habit.api.preprocessing import preprocess_image as _api_preprocess_image

    return _api_preprocess_image(
        image, steps, mask=mask, modality=modality
    )
