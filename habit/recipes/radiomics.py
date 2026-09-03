"""L4 standalone, configuration-driven whole-ROI radiomics recipe."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from habit.adapters.radiomics_io import (
    RadiomicsFeatureRow,
    RadiomicsFilePair,
    discover_radiomics_file_pairs,
    write_radiomics_feature_tables,
)
from habit.api.contracts import WorkflowResult, coerce_config
from habit.api.image import GeometryPolicy
from habit.api.provenance import create_run_manifest, write_run_manifest
from habit.exceptions import ProcessingError
from habit.schemas.workflows.habitat import RadiomicsConfig
from habit.utils.progress_utils import CustomTqdm
from habit.utils.radiomics_preset_utils import resolve_params_file

__all__ = ["traditional_radiomics"]

_LOG = logging.getLogger(__name__)


def _extract_pair(pair: RadiomicsFilePair, params_file: Optional[str], label: int) -> RadiomicsFeatureRow:
    """Extract one configured label through the stable low-level public API."""
    from habit.api.radiomics import GeometryPolicy, extract_features

    result = extract_features(
        pair.image_path,
        pair.mask_path,
        params_file,
        label=label,
        geometry_policy=GeometryPolicy.HARMONIZE,
    )
    return RadiomicsFeatureRow(
        subject_id=pair.subject_id,
        modality=pair.modality,
        label=label,
        values=result.values,
    )


def traditional_radiomics(
    config: RadiomicsConfig | Mapping[str, Any],
    *,
    logger: Optional[logging.Logger] = None,
) -> WorkflowResult[None]:
    """Run standalone PyRadiomics with all declared YAML processing controls.

    ``target_labels`` are evaluated independently. With one label the historical
    filenames are retained; multiple labels add ``_label_<id>`` to prevent an
    ambiguous merged foreground table.
    """
    validated = coerce_config(config, RadiomicsConfig)
    log = logger or _LOG
    out_dir = Path(validated.out_dir or validated.paths.out_dir)
    source = validated.images_folder or validated.paths.images_folder
    params_file = resolve_params_file(
        validated.params_file or validated.paths.params_file, preset="roi"
    )
    processing = validated.processing
    pairs = discover_radiomics_file_pairs(
        source, modalities=processing.process_image_types
    )
    labels = tuple(processing.target_labels)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: List[RadiomicsFeatureRow] = []
    failures: Dict[str, str] = {}
    tasks = [(pair, label) for pair in pairs for label in labels]
    progress = CustomTqdm(total=len(tasks), desc="Extracting Radiomics Features")
    try:
        def consume(pair: RadiomicsFilePair, label: int) -> None:
            key = f"{pair.subject_id}/{pair.modality}/label_{label}"
            try:
                rows.append(_extract_pair(pair, params_file, label))
            except Exception as exc:  # retain batch progress while recording an honest failure
                failures[key] = f"{type(exc).__name__}: {exc}"
                log.error("Radiomics extraction failed for %s: %s", key, exc)
            finally:
                progress.update(1)
        if processing.n_processes == 1:
            for pair, label in tasks:
                consume(pair, label)
                if len(rows) and len(rows) % processing.save_every_n_files == 0:
                    write_radiomics_feature_tables(
                        out_dir, rows, export_by_image_type=validated.export.export_by_image_type,
                        export_combined=validated.export.export_combined, export_format=validated.export.export_format,
                        add_timestamp=validated.export.add_timestamp,
                        timestamp=run_timestamp,
                        target_labels=labels,
                        partial=True,
                    )
        else:
            # Each task only calls the low-level public API; keeping orchestration
            # in this recipe avoids reviving a compat batch engine.
            with ThreadPoolExecutor(max_workers=processing.n_processes) as pool:
                for pair, label in tasks:
                    pool.submit(consume, pair, label)
            # The final writer below commits one deterministic complete snapshot.
    finally:
        progress.close()
    if not rows:
        raise ProcessingError(
            "Standalone radiomics failed for every configured image/mask/label pair: "
            + "; ".join(f"{key}: {value}" for key, value in failures.items())
        )
    if failures:
        manifest = create_run_manifest(
            "radiomics",
            validated,
            metadata={
                "engine": "v2",
                "status": "partial_failure",
                "target_labels": list(labels),
                "completed_pairs": len(rows),
                "failed_pairs": failures,
            },
        )
        manifest_path = write_run_manifest(manifest, out_dir)
        raise ProcessingError(
            "Standalone radiomics produced partial checkpoint exports only; "
            f"inspect {manifest_path} for failed labels: {failures}."
        )
    artifacts = write_radiomics_feature_tables(
        out_dir, rows,
        export_by_image_type=validated.export.export_by_image_type,
        export_combined=validated.export.export_combined,
        export_format=validated.export.export_format,
        add_timestamp=validated.export.add_timestamp,
        timestamp=run_timestamp,
        target_labels=labels,
        partial=False,
    )
    manifest = create_run_manifest(
        "radiomics",
        validated,
        metadata={
            "engine": "v2",
            "target_labels": list(labels),
            "output_format": validated.export.export_format,
            "export_by_image_type": validated.export.export_by_image_type,
            "export_combined": validated.export.export_combined,
            "add_timestamp": validated.export.add_timestamp,
            "completed_pairs": len(rows),
            "failed_pairs": failures,
        },
    )
    manifest_path = write_run_manifest(manifest, out_dir)
    return WorkflowResult(
        output_dir=out_dir,
        artifacts=artifacts,
        metadata={"engine": "v2", "status": "complete", "config_hash": manifest.config_hash, "failed_pairs": failures},
        run_id=manifest.run_id,
        manifest_path=manifest_path,
    )
