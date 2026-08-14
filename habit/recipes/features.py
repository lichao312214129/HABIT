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
"""L4 feature-extraction recipes.

``extract_habitat_features`` is the domain-native path for ``habit extract``:
per-subject :class:`~habit.domain.protocols.HabitatFeatureExtractor` calls
over :class:`~habit.contracts.subject.Subject` +
:class:`~habit.contracts.habitat.HabitatMap`, with optional parallelism via
:class:`~habit.contracts.ops.ExecutionBackend`. Filesystem discovery and CSV
layout live in L1 adapters; this module only assembles them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from habit.api.contracts import WorkflowResult, coerce_config
from habit.api.provenance import create_run_manifest, write_run_manifest
from habit.contracts.ops import ExecutionBackend, SubjectResult
from habit.contracts.subject import Subject
from habit.contracts.table import FeatureTable
from habit.exceptions import HABITAPIError, ProcessingError
from habit.utils.log_utils import get_module_logger
from habit.utils.parallel_utils import resolve_process_count
from habit.utils.progress_utils import CustomTqdm

__all__ = ["extract_habitat_features", "traditional_radiomics"]

_LOG = get_module_logger(__name__)

def _domain_feature_type_names() -> FrozenSet[str]:
    """
    Feature family names routable to the domain extract path.

    The set is read from the
    :class:`~habit.domain.habitat_features.HabitatFeatureExtractorRegistry`
    after loading ``habit.habitat_feature_extractor`` entry points, so
    third-party domain plugins dispatch exactly like the built-in families.

    Returns:
        Registered domain extractor names (built-ins plus entry points).
    """
    import habit.domain.habitat_features  # noqa: F401  (register built-ins)
    from habit.domain.habitat_features import HabitatFeatureExtractorRegistry

    HabitatFeatureExtractorRegistry.load_entry_points()
    return frozenset(HabitatFeatureExtractorRegistry.available())


def _legacy_feature_type_names() -> FrozenSet[str]:
    """
    Feature family names the v0.1 ``HabitatFeatureFactory`` provides.

    Imported lazily so the domain path never pays for -- or triggers -- a
    ``habit.compat`` import; only a genuine compat fallback calls this.

    Returns:
        Registered v0.1 handler names (built-ins plus legacy optional packages).
    """
    from habit.compat.engines.habitat_extraction.feature_registry import (
        HabitatFeatureFactory,
    )

    return frozenset(HabitatFeatureFactory.registered_feature_names())


@dataclass(frozen=True)
class _ExtractItem:
    """Picklable per-subject payload for the execution backend."""

    subject: Subject
    habitat_path: str
    habitat_ids: Tuple[int, ...]

    @property
    def subject_id(self) -> str:
        """Expose subject id for ExecutionBackend result / progress keys."""
        return self.subject.subject_id


class _SubjectHabitatExtractOp:
    """
    Subject-level operator: load the habitat map, run all extractors, return
    a mapping of feature family name -> one-row FeatureTable.
    """

    def __init__(
        self,
        extractors: Mapping[str, Any],
    ) -> None:
        self._extractors = dict(extractors)

    def __call__(self, item: _ExtractItem) -> Dict[str, FeatureTable]:
        """
        Extract every requested family for one subject.

        Args:
            item: Subject plus habitat map path and canonical habitat ids.

        Returns:
            Mapping of feature type name to that subject's feature table.
        """
        from habit.adapters.extract_io import read_habitat_map

        habitat_map = read_habitat_map(
            item.habitat_path,
            subject_id=item.subject.subject_id,
            habitat_ids=item.habitat_ids,
        )
        tables: Dict[str, FeatureTable] = {}
        for name, extractor in self._extractors.items():
            tables[name] = extractor(item.subject, habitat_map)
        return tables


def _error_summary(error: BaseException) -> str:
    """Format one subject failure for logging."""
    return f"{type(error).__name__}: {error}"


def _backend_for_processes(n_processes: int) -> ExecutionBackend:
    """
    Build an execution backend from the v0.1 ``n_processes`` knob.

    Args:
        n_processes: Requested worker count (resolved to >= 1).

    Returns:
        Process-pool backend when ``n_processes > 1``, otherwise serial.
    """
    from habit.execution.backends import SerialBackend
    from habit.execution.process_pool import ProcessPoolBackend
    from habit.spec.policy import RunPolicy

    workers = resolve_process_count(n_processes)
    if workers <= 1:
        return SerialBackend(on_subject_failure="continue")
    policy = RunPolicy(
        workers=workers,
        backend="process",
        on_subject_failure="continue",
        # Feature extraction has no per-subject timeout in the v0.1 YAML;
        # disable so long radiomics subjects are not killed.
        subject_timeout_sec=None,
        subject_spawn_timeout_sec=None,
    )
    return ProcessPoolBackend.from_policy(policy)


def _build_domain_extractors(
    feature_types: Sequence[str],
    *,
    params_file_of_non_habitat: Optional[str],
    params_file_of_habitat: Optional[str],
    use_torch_radiomics: Any = "auto",
    torch_device: str = "auto",
    torch_dtype: str = "float32",
    plugin_configs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Instantiate domain extractors for the requested built-in feature types.

    Args:
        feature_types: Names from the YAML ``feature_types`` list.
        params_file_of_non_habitat: PyRadiomics params for raw-image families.
        params_file_of_habitat: PyRadiomics params for whole-habitat radiomics.
        use_torch_radiomics: TorchRadiomics backend switch (scientific
            defaults unchanged; only the compute path may switch).
        torch_device: Torch device string or ``"auto"``.
        torch_dtype: Torch dtype name for the torch path.
        plugin_configs: Optional plugin settings; the ``graph`` entry carries
            the validated graph-extraction parameters from the YAML
            ``graph:`` block.

    Returns:
        Mapping of feature type name to extractor instance.
    """
    # Ensure built-in extractors are registered, then read the routable set
    # from the registry so third-party entry-point plugins dispatch like
    # built-ins.
    import habit.domain.habitat_features  # noqa: F401
    from habit.domain.habitat_features import HabitatFeatureExtractorRegistry
    from habit.utils.radiomics_preset_utils import resolve_params_file

    HabitatFeatureExtractorRegistry.load_entry_points()
    available = set(HabitatFeatureExtractorRegistry.available())

    roi_params = resolve_params_file(params_file_of_non_habitat, preset="roi")
    habitat_params = resolve_params_file(params_file_of_habitat, preset="habitat")

    extractors: Dict[str, Any] = {}
    for name in feature_types:
        if name not in available:
            continue
        if name == "traditional":
            extractors[name] = HabitatFeatureExtractorRegistry.create(
                name,
                params_file=roi_params,
                use_torch_radiomics=use_torch_radiomics,
                torch_device=torch_device,
                torch_dtype=torch_dtype,
            )
        elif name == "each_habitat":
            extractors[name] = HabitatFeatureExtractorRegistry.create(
                name,
                params_file=roi_params,
                use_torch_radiomics=use_torch_radiomics,
                torch_device=torch_device,
                torch_dtype=torch_dtype,
            )
        elif name == "whole_habitat":
            extractors[name] = HabitatFeatureExtractorRegistry.create(
                name,
                params_file=habitat_params,
                use_torch_radiomics=use_torch_radiomics,
                torch_device=torch_device,
                torch_dtype=torch_dtype,
            )
        elif name == "graph":
            extractors[name] = HabitatFeatureExtractorRegistry.create(
                name,
                **_graph_params_from_plugin_configs(plugin_configs),
            )
        else:
            extractors[name] = HabitatFeatureExtractorRegistry.create(name)
    return extractors


def _graph_params_from_plugin_configs(
    plugin_configs: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Extract graph-extraction parameters from the plugin config mapping.

    The ``graph`` entry may be a validated
    :class:`~habit.schemas.workflows.habitat.GraphFeatureBlock` (YAML path), a
    plain mapping (direct API callers), or the deprecated extraction-only
    params model (compat shim). Visualization and legacy block keys are
    steering metadata for the recipe's figure hook, not extractor constructor
    parameters, so they are filtered out here; every other key reaches the
    registry, which rejects unknown names with a precise error.

    Args:
        plugin_configs: Plugin settings keyed by plugin name.

    Returns:
        Keyword arguments for the ``graph`` domain extractor; empty when no
        graph settings were provided (extractor defaults apply).
    """
    if not plugin_configs or "graph" not in plugin_configs:
        return {}
    graph_cfg = plugin_configs["graph"]
    if hasattr(graph_cfg, "model_dump"):
        data = dict(graph_cfg.model_dump())
    elif isinstance(graph_cfg, Mapping):
        data = dict(graph_cfg)
    else:
        return {}

    from habit.domain.habitat_features.graph import GraphHabitatFeaturesParams
    from habit.schemas.workflows.habitat import GraphFeatureBlock

    non_extraction = set(GraphFeatureBlock.model_fields) - set(
        GraphHabitatFeaturesParams.model_fields
    )
    return {key: value for key, value in data.items() if key not in non_extraction}


def _graph_block_from_plugin_configs(
    plugin_configs: Optional[Mapping[str, Any]],
) -> Optional[Any]:
    """
    Coerce the graph plugin config into the validated block schema.

    Accepts the block model itself (YAML path), a plain mapping (direct API
    callers, coerced so the same defaults apply), or the deprecated
    extraction-only params model from the compat shim -- the last carries no
    visualization fields, so the figure hook stays off on that path.

    Args:
        plugin_configs: Plugin settings keyed by plugin name.

    Returns:
        The validated ``GraphFeatureBlock``, or ``None`` when no graph
        settings exist.
    """
    if not plugin_configs or "graph" not in plugin_configs:
        return None
    from habit.schemas.workflows.habitat import GraphFeatureBlock

    graph_cfg = plugin_configs["graph"]
    if isinstance(graph_cfg, GraphFeatureBlock):
        return graph_cfg
    if isinstance(graph_cfg, Mapping):
        return GraphFeatureBlock.model_validate(dict(graph_cfg))
    if hasattr(graph_cfg, "model_dump"):
        # The deprecated compat shim passes the extraction-only params model:
        # keep its extraction values (so the drawn graph still matches the
        # measured features) and let the visualization fields default.
        return GraphFeatureBlock.model_validate(graph_cfg.model_dump())
    return GraphFeatureBlock()


def _map_extract_items(
    items: Sequence[_ExtractItem],
    op: _SubjectHabitatExtractOp,
    *,
    backend: ExecutionBackend,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, FeatureTable]], Dict[str, str]]:
    """
    Apply the extract operator with soft failure (continue on subject error).

    Args:
        items: Per-subject payloads.
        op: Extract operator.
        backend: Execution backend.
        logger: Run logger.

    Returns:
        ``(successful_tables_in_item_order, failures)``.

    Raises:
        ProcessingError: If every subject failed.
    """
    total = len(items)
    bar = CustomTqdm(total=total, desc="Extracting Features")

    def _progress(completed: int, expected: int) -> None:
        bar.total = expected
        bar.n = completed
        bar.refresh()

    try:
        slots: Sequence[SubjectResult[Dict[str, FeatureTable]]] = list(
            backend.map(op, items, progress=_progress)
        )
    finally:
        bar.close()

    failures: Dict[str, str] = {}
    values_by_id: Dict[str, Dict[str, FeatureTable]] = {}
    for slot in slots:
        if slot.error is not None:
            summary = _error_summary(slot.error)
            failures[slot.subject_id] = summary
            logger.warning(
                "[extract] subject %s failed: %s", slot.subject_id, summary
            )
        elif slot.value is not None:
            values_by_id[slot.subject_id] = slot.value

    if not values_by_id:
        detail = "; ".join(
            f"{sid}: {msg}" for sid, msg in sorted(failures.items())
        )
        raise ProcessingError(
            f"All {len(items)} subject(s) failed in feature extraction: {detail}"
        )

    # Preserve discovery order for CSV row stability.
    ordered = [
        values_by_id[item.subject.subject_id]
        for item in items
        if item.subject.subject_id in values_by_id
    ]
    return ordered, failures


def _write_graph_visualizations(
    items: Sequence[_ExtractItem],
    *,
    block: Any,
    out_dir: Path,
    logger: logging.Logger,
) -> List[Path]:
    """
    Render per-subject habitat graph topology figures (``graph.visualize``).

    Figures are rendered serially in the main process after the CSV export:
    rendering is cheap relative to feature extraction, and serial execution
    sidesteps Windows process-pool pickling of matplotlib figures. A missing
    optional rendering backend (matplotlib for 2D, pyvista for 3D) skips the
    affected figures with a warning instead of failing the completed
    extraction. All figure text is English-only.

    Args:
        items: Per-subject payloads (habitat maps are re-read from disk).
        block: Validated ``GraphFeatureBlock`` carrying the visualization
            settings.
        out_dir: Extraction output directory; figures land in
            ``visualizations/graph/`` under it (v0.1 layout convention).
        logger: Run logger.

    Returns:
        Paths of the figure files written.
    """
    from habit.exceptions import OptionalDependencyError
    from habit.utils.optional_deps import require

    purpose = "habitat graph topology figures"
    try:
        # Force the headless Agg canvas before pyplot loads, mirroring
        # habit.viz.habitat_graph._plt: figure export must work on machines
        # without a display.
        matplotlib = require("matplotlib", extra="viz", purpose=purpose)
        if matplotlib.get_backend().lower() not in (
            "agg",
            "module://matplotlib_inline.backend_inline",
        ):
            matplotlib.use("Agg")
        plt = require("matplotlib.pyplot", extra="viz", purpose=purpose)
    except OptionalDependencyError as exc:
        logger.warning("Graph visualization skipped: %s", exc)
        return []

    from habit.domain.habitat_features.graph import GraphHabitatFeaturesParams
    from habit.kernels.habitat_graph import HabitatGraphFeatureOptions

    # The drawn graph must match the measured features, so the renderers get
    # the same construction options as the extractor.
    options = HabitatGraphFeatureOptions(
        **{
            field: getattr(block, field)
            for field in GraphHabitatFeaturesParams.model_fields
        }
    )

    formats = (
        ("png", "pdf")
        if block.visualization_format == "both"
        else (block.visualization_format,)
    )
    figure_dir = out_dir / "visualizations" / "graph"
    figure_dir.mkdir(parents=True, exist_ok=True)

    render_3d = bool(block.visualization_save_3d)
    if render_3d:
        try:
            require("pyvista", extra="view", purpose="3D habitat graph rendering")
            require("skimage", extra="slic", purpose="3D habitat graph rendering")
        except OptionalDependencyError as exc:
            logger.warning("3D graph rendering skipped: %s", exc)
            render_3d = False

    written: List[Path] = []
    bar = CustomTqdm(total=len(items), desc="Rendering Graph Figures")
    try:
        for item in items:
            bar.update(1)
            try:
                written.extend(
                    _render_one_subject_graph_figures(
                        item,
                        options=options,
                        block=block,
                        formats=formats,
                        render_3d=render_3d,
                        figure_dir=figure_dir,
                        plt=plt,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                # A degenerate map must not void the cohort's figures.
                logger.warning(
                    "Graph visualization failed for subject %s: %s",
                    item.subject_id,
                    exc,
                )
    finally:
        bar.close()
    if written:
        logger.info(
            "Wrote %d graph figure file(s) under %s", len(written), figure_dir
        )
    return written


def _render_one_subject_graph_figures(
    item: _ExtractItem,
    *,
    options: Any,
    block: Any,
    formats: Sequence[str],
    render_3d: bool,
    figure_dir: Path,
    plt: Any,
) -> List[Path]:
    """
    Render and save every graph figure for one subject.

    Args:
        item: Subject payload with the habitat map path.
        options: ``HabitatGraphFeatureOptions`` shared with the extractor.
        block: Validated ``GraphFeatureBlock`` (visualization settings).
        formats: 2D file formats to write (``png`` / ``pdf``).
        render_3d: Whether 3D renders are requested AND their optional
            dependencies are present.
        figure_dir: Destination directory (``visualizations/graph/``).
        plt: The matplotlib pyplot module (figure cleanup and imsave).

    Returns:
        Paths of the figure files written for this subject.
    """
    import numpy as np

    from habit.adapters.extract_io import read_habitat_map
    from habit.viz.habitat_graph import (
        plot_habitat_graph_network_2d,
        plot_habitat_graph_slice,
    )

    habitat_map = read_habitat_map(
        item.habitat_path,
        subject_id=item.subject.subject_id,
        habitat_ids=item.habitat_ids,
    )
    labels = np.asarray(habitat_map.label_array)

    written: List[Path] = []
    figures = (
        (
            "graph_slice",
            plot_habitat_graph_slice(
                labels,
                options=options,
                show_grid=block.visualization_show_grid,
                block_size=block.visualization_block_size,
                grid_linestyle=block.visualization_grid_linestyle,
            ),
        ),
        (
            "graph_network_2d",
            plot_habitat_graph_network_2d(
                labels,
                options=options,
                show_background=block.visualization_show_background,
                show_grid=block.visualization_show_grid,
                block_size=block.visualization_block_size,
                grid_linestyle=block.visualization_grid_linestyle,
            ),
        ),
    )
    for stem, figure in figures:
        if figure is None:
            continue
        for fmt in formats:
            destination = figure_dir / f"{item.subject_id}_{stem}.{fmt}"
            figure.savefig(
                destination, dpi=block.visualization_dpi, bbox_inches="tight"
            )
            written.append(destination)
        plt.close(figure)

    if render_3d and labels.ndim == 3:
        from habit.viz.habitat_graph import (
            render_habitat_graph_network_3d,
            render_habitat_graph_surface_3d,
        )

        # Geometry spacing is SimpleITK-ordered (x, y, z); the renderers
        # expect array-axis order (z, y, x).
        spacing = tuple(
            float(v) for v in reversed(habitat_map.geometry.spacing)
        )
        # 3D renders are raster RGB arrays; they are always written as PNG.
        renders = (
            (
                "graph_surface_3d",
                render_habitat_graph_surface_3d(labels, spacing=spacing),
            ),
            (
                "graph_network_3d",
                render_habitat_graph_network_3d(
                    labels, options=options, spacing=spacing
                ),
            ),
        )
        for stem, rgb in renders:
            if rgb is None:
                continue
            destination = figure_dir / f"{item.subject_id}_{stem}.png"
            plt.imsave(destination, rgb, dpi=block.visualization_dpi)
            written.append(destination)
    return written


def _run_domain_extract(
    config: Any,
    *,
    logger: logging.Logger,
    backend: Optional[ExecutionBackend],
    plugin_configs: Optional[Mapping[str, Any]] = None,
) -> WorkflowResult[None]:
    """
    Domain-native extract path for built-in feature families.

    Args:
        config: Validated :class:`FeatureExtractionConfig`.
        logger: Run logger.
        backend: Optional caller-supplied backend; when omitted, derived from
            ``config.n_processes``.

    Returns:
        Workflow result with output directory metadata.
    """
    from habit.adapters.extract_io import (
        discover_habitat_map_paths,
        load_extract_cohort,
        resolve_n_habitats,
        write_extract_feature_csvs,
    )

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_habitats = resolve_n_habitats(config.habitats_map_folder, config.n_habitats)
    logger.info("Using habitat count: %s", n_habitats)
    habitat_ids = tuple(range(1, n_habitats + 1))

    cohort = load_extract_cohort(config.raw_img_folder)
    habitat_paths = discover_habitat_map_paths(
        config.habitats_map_folder, config.habitat_pattern
    )
    matched_ids = [sid for sid in cohort.subject_ids if sid in habitat_paths]
    if not matched_ids:
        raise HABITAPIError(
            "No matching subjects found between raw images and habitat maps "
            f"(images={list(cohort.subject_ids)}, "
            f"habitats={sorted(habitat_paths)})."
        )
    if len(matched_ids) < len(cohort):
        missing = sorted(set(cohort.subject_ids) - set(matched_ids))
        logger.warning(
            "Skipping %d subject(s) without habitat maps: %s",
            len(missing),
            ", ".join(missing),
        )

    subjects_by_id = {subject.subject_id: subject for subject in cohort}
    items = [
        _ExtractItem(
            subject=subjects_by_id[sid],
            habitat_path=str(habitat_paths[sid]),
            habitat_ids=habitat_ids,
        )
        for sid in matched_ids
    ]

    use_torch = getattr(config, "use_torch_radiomics", False)
    torch_device = getattr(config, "torch_device", "auto")
    torch_dtype = getattr(config, "torch_dtype", "float32")
    extractors = _build_domain_extractors(
        config.feature_types,
        params_file_of_non_habitat=config.params_file_of_non_habitat,
        params_file_of_habitat=config.params_file_of_habitat,
        use_torch_radiomics=use_torch,
        torch_device=str(torch_device),
        torch_dtype=str(torch_dtype),
        plugin_configs=plugin_configs,
    )
    if not extractors:
        raise HABITAPIError(
            "No domain extractors constructed for feature_types="
            f"{list(config.feature_types)}."
        )

    runner = backend if backend is not None else _backend_for_processes(
        config.n_processes
    )
    logger.info(
        "Starting domain feature extraction for %s subjects (%s)",
        len(items),
        type(runner).__name__,
    )
    per_subject, failures = _map_extract_items(
        items, _SubjectHabitatExtractOp(extractors), backend=runner, logger=logger
    )
    if failures:
        logger.warning(
            "Feature extraction finished with %d subject failure(s).",
            len(failures),
        )

    family_tables: Dict[str, List[FeatureTable]] = {
        name: [] for name in extractors
    }
    for subject_tables in per_subject:
        for name, table in subject_tables.items():
            family_tables[name].append(table)

    write_extract_feature_csvs(
        out_dir,
        family_tables,
        n_habitats=n_habitats,
        logger=logger,
    )

    # Figure hook: ``graph: {visualize: true}`` renders per-subject topology
    # figures after the CSV export, keeping the v0.1 plugin's output layout.
    graph_block = _graph_block_from_plugin_configs(plugin_configs)
    if "graph" in extractors and graph_block is not None and graph_block.visualize:
        _write_graph_visualizations(
            items, block=graph_block, out_dir=out_dir, logger=logger
        )

    manifest = create_run_manifest(
        "feature_extraction",
        config,
        metadata={
            "engine": "domain",
            "feature_types": list(config.feature_types),
            "n_subjects": len(per_subject),
            "n_failures": len(failures),
        },
    )
    manifest_path = write_run_manifest(manifest, out_dir)
    return WorkflowResult(
        output_dir=out_dir,
        metadata={
            "config_hash": manifest.config_hash,
            "habit_version": manifest.habit_version,
            "engine": "domain",
        },
        run_id=manifest.run_id,
        manifest_path=manifest_path,
    )


def _run_compat_extract(
    config: Any,
    *,
    plugin_configs: Optional[Mapping[str, Any]],
    logger: logging.Logger,
) -> WorkflowResult[None]:
    """
    Compat fallback for legacy-only feature families not in the domain registry.

    Args:
        config: Validated feature-extraction config.
        plugin_configs: Optional plugin settings.
        logger: Run logger.

    Returns:
        Workflow result with output directory metadata.
    """
    from habit.compat.legacy_core import run_feature_extraction_from_config

    logger.info(
        "Using compat HabitatMapAnalyzer for feature_types requiring plugins: %s",
        list(config.feature_types),
    )
    run_feature_extraction_from_config(
        config,
        logger=logger,
        plugin_configs=dict(plugin_configs) if plugin_configs is not None else None,
    )
    manifest = create_run_manifest(
        "feature_extraction",
        config,
        metadata={
            "engine": "compat",
            "plugins": sorted((plugin_configs or {}).keys()),
        },
    )
    manifest_path = write_run_manifest(manifest, config.out_dir)
    return WorkflowResult(
        output_dir=config.out_dir,
        metadata={
            "config_hash": manifest.config_hash,
            "habit_version": manifest.habit_version,
            "engine": "compat",
        },
        run_id=manifest.run_id,
        manifest_path=manifest_path,
    )


def extract_habitat_features(
    config: Any,
    *,
    plugin_configs: Optional[Mapping[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    backend: Optional[ExecutionBackend] = None,
) -> WorkflowResult[None]:
    """
    Extract features from pre-computed habitat maps (``habit extract`` recipe).

    Families registered in the domain
    :class:`~habit.domain.habitat_features.HabitatFeatureExtractorRegistry`
    (built-ins plus ``habit.habitat_feature_extractor`` entry points) run
    through domain
    :class:`~habit.domain.protocols.HabitatFeatureExtractor` instances and an
    optional :class:`~habit.contracts.ops.ExecutionBackend`. A name found only
    in the v0.1 ``HabitatFeatureFactory`` routes the whole request to the
    compat ``HabitatMapAnalyzer`` so legacy plugin YAML keeps working; a name
    known to neither registry raises :class:`~habit.exceptions.HABITAPIError`.

    When the ``graph`` family runs and its settings block has
    ``visualize: true``, per-subject topology figures are rendered under
    ``<out_dir>/visualizations/graph/`` after the CSV export.

    Args:
        config: Validated feature-extraction configuration (schema object or
            mapping accepted by
            :func:`habit.api.habitat.build_feature_extraction_config`).
        plugin_configs: Optional plugin settings (e.g. ``graph``).
        logger: Optional run logger.
        backend: Optional execution backend. When omitted, ``n_processes`` from
            the config selects serial vs process-pool.

    Returns:
        :class:`~habit.api.contracts.WorkflowResult` with output directory
        metadata and a run manifest path.

    Raises:
        HABITAPIError: If a requested feature type is registered nowhere.
    """
    from habit.api.habitat import build_feature_extraction_config
    from habit.schemas.workflows.habitat import FeatureExtractionConfig

    log = logger or _LOG
    resolved_plugins: Optional[Dict[str, Any]]
    if isinstance(config, Mapping):
        validated_config, inferred = build_feature_extraction_config(config)
        resolved_plugins = (
            dict(plugin_configs) if plugin_configs is not None else inferred
        )
    else:
        validated_config = coerce_config(config, FeatureExtractionConfig)
        resolved_plugins = (
            dict(plugin_configs) if plugin_configs is not None else None
        )

    feature_types = list(validated_config.feature_types)
    domain_names = _domain_feature_type_names()
    unknown = [name for name in feature_types if name not in domain_names]
    if unknown:
        legacy_names = _legacy_feature_type_names()
        if all(name in legacy_names for name in unknown):
            return _run_compat_extract(
                validated_config,
                plugin_configs=resolved_plugins,
                logger=log,
            )
        raise HABITAPIError(
            f"Unknown feature_types: {unknown}. Available domain families: "
            f"{sorted(domain_names)}; legacy compat families: "
            f"{sorted(legacy_names)}."
        )
    return _run_domain_extract(
        validated_config,
        logger=log,
        backend=backend,
        plugin_configs=resolved_plugins,
    )


def traditional_radiomics(
    config: Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """
    Run standalone traditional radiomics extraction (``habit radiomics`` recipe).

    Args:
        config: Validated radiomics configuration (v0.1 schema object or
            mapping accepted by :class:`~habit.api.habitat.RadiomicsConfig`).
        logger: Optional run logger forwarded to the workflow helper.

    Returns:
        :class:`~habit.api.contracts.WorkflowResult` with output directory
        metadata and a run manifest path.
    """
    from habit.api.habitat import run_radiomics

    return run_radiomics(config, logger=logger)
