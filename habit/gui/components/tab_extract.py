# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Habitat feature extraction tab for Gradio GUI."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from pydantic import ValidationError

from habit.cli_commands.commands.cmd_extract_features import run_extract_features
from habit.core.habitat_analysis.feature_extraction_loader import (
    parse_feature_extraction_config,
)
from habit.core.habitat_analysis.feature_registry import (
    bootstrap_optional_plugins,
    get_all_feature_type_names,
)
from habit.gui.job_controls import (
    job_end_button_updates,
    job_start_button_updates,
    on_stop_job_click,
)
from habit.gui.pipeline_runner import run_background_job
from habit.gui.utils import (
    abs_path,
    default_radiomics_param,
    extract_validation_msgs,
    load_config_yaml,
    load_gui_draft,
    open_directory,
    render_console_log,
    save_config_yaml,
    save_gui_draft,
    select_local_path,
    translate_pydantic_error,
)

DEFAULT_NON_HABITAT_PARAMS_FILE: str = default_radiomics_param("parameter.yaml")
DEFAULT_HABITAT_PARAMS_FILE: str = default_radiomics_param("parameter_habitat.yaml")

bootstrap_optional_plugins()
FEATURE_TYPE_CHOICES: List[str] = get_all_feature_type_names()
GRAPH_PLUGIN_AVAILABLE: bool = "graph" in FEATURE_TYPE_CHOICES
DEFAULT_FEATURE_TYPES: List[str] = [
    name
    for name in [
        "traditional",
        "non_radiomics",
        "whole_habitat",
        "msi",
        "ith_score",
        "graph",
    ]
    if name in FEATURE_TYPE_CHOICES
]

# Defaults aligned with GraphFeatureConfig in graph_features/config.py (private plugin)
GRAPH_EDGE_METHOD_CHOICES: List[str] = [
    "centroid_distance",
    "adjacency",
]
GRAPH_EDGE_WEIGHT_CHOICES: List[str] = [
    "none",
    "distance",
    "inverse_distance",
    "contact_voxels",
]
GRAPH_CONNECTIVITY_CHOICES: List[str] = ["face", "full"]
GRAPH_ADJACENCY_CONNECTIVITY_CHOICES: List[str] = ["face", "edge", "corner"]

# Number of scalar/widget values returned by load_yaml before graph_block_state.
# Includes row-visibility gr.update objects at the tail of the list.
_LOAD_WIDGET_COUNT: int = 33


def _graph_cfg_value(graph_cfg: Dict[str, Any], key: str, default: Any) -> Any:
    """Read one graph config key with schema default fallback."""
    value = graph_cfg.get(key, default)
    return default if value is None else value


def _build_graph_config_from_gui(
    graph_state_val: Optional[Dict[str, Any]],
    include_single: bool,
    include_pairwise: bool,
    edge_method: str,
    distance_threshold: float,
    adjacency_connectivity: str,
    adjacency_min_voxels: float,
    edge_weight: str,
    min_region_voxels: float,
    connectivity: str,
    erosion_radius: float,
    subdivide_region_voxels: float,
    block_size: float,
    block_min_coverage: float,
    pairwise_intra: bool,
    visualize: bool,
    visualization_format: str,
    visualization_dpi: float,
    visualization_show_background: bool,
    visualization_save_3d: bool,
    include_extended_metrics: bool,
    extended_min_nodes: float,
) -> Dict[str, Any]:
    """
    Merge GUI graph controls into the YAML ``graph:`` block.

    Preserves keys loaded from an existing YAML (via ``graph_state_val``) that are
    not exposed as dedicated widgets, then overwrites known keys from the form.

    Args:
        graph_state_val: Optional graph dict loaded from YAML (unknown-key carrier).
        include_single: Whether to compute single-habitat graphs.
        include_pairwise: Whether to compute pairwise habitat graphs.
        edge_method: Edge identification rule.
        distance_threshold: Centroid distance threshold (voxels).
        adjacency_connectivity: Neighbor rule for adjacency edges (face/edge/corner).
        adjacency_min_voxels: Minimum adjacent voxel pairs for adjacency edges.
        edge_weight: Optional edge weight source.
        min_region_voxels: Minimum region size kept as a node.
        connectivity: Connected-component neighborhood rule.
        erosion_radius: Binary erosion iterations before labeling.
        subdivide_region_voxels: Split large components above this size; 0 disables.
        block_size: Grid block edge length when subdividing.
        block_min_coverage: Minimum block occupancy to keep a node.
        pairwise_intra: Add intra-habitat edges to pairwise graphs.
        visualize: Render graph topology figures.
        visualization_format: Figure format (pdf/png/both).
        visualization_dpi: Raster DPI.
        visualization_show_background: Draw semi-transparent habitat partitions behind graph.
        visualization_save_3d: Save PyVista 3D surface and network figures.
        include_extended_metrics: Compute efficiency, small-world, rich-club, etc.
        extended_min_nodes: Minimum nodes for small-world sigma.

    Returns:
        Dict[str, Any]: Graph configuration mapping for FeatureExtractionConfig.
    """
    graph_data: Dict[str, Any] = (
        dict(graph_state_val) if isinstance(graph_state_val, dict) else {}
    )
    graph_data.update(
        {
            "include_single_habitat_graph": bool(include_single),
            "include_pairwise_habitat_graph": bool(include_pairwise),
            "edge_method": str(edge_method),
            "distance_threshold": float(distance_threshold),
            "adjacency_connectivity": str(adjacency_connectivity),
            "adjacency_min_voxels": int(adjacency_min_voxels),
            "edge_weight": str(edge_weight),
            "min_region_voxels": int(min_region_voxels),
            "connectivity": str(connectivity),
            "erosion_radius": int(erosion_radius),
            "subdivide_region_voxels": int(subdivide_region_voxels),
            "block_size": int(block_size),
            "block_min_coverage": float(block_min_coverage),
            "pairwise_include_intra_edges": bool(pairwise_intra),
            "visualize": bool(visualize),
            "visualization_format": str(visualization_format),
            "visualization_dpi": int(visualization_dpi),
            "visualization_show_background": bool(visualization_show_background),
            "visualization_save_3d": bool(visualization_save_3d),
            "include_extended_metrics": bool(include_extended_metrics),
            "extended_min_nodes": int(extended_min_nodes),
        }
    )
    return graph_data


def _toggle_graph_section(selected_features: List[str]) -> Dict[str, Any]:
    """
    Show or hide the entire graph parameter panel based on feature_types selection.

    Args:
        selected_features: Current feature_types checkbox values.

    Returns:
        Dict[str, Any]: gr.update for the graph parameter group container.
    """
    enabled = "graph" in set(selected_features or [])
    return gr.update(visible=enabled)


def _toggle_edge_method_fields(edge_method: str) -> Tuple[Any, Any]:
    """
    Show the relevant parameter row depending on the selected edge_method.

    Args:
        edge_method: Selected graph edge identification rule.

    Returns:
        Tuple[Any, Any]: Updates for distance_threshold row and adjacency row
        respectively.
    """
    use_distance = edge_method == "centroid_distance"
    use_adjacency = edge_method == "adjacency"
    return (
        gr.update(visible=use_distance),
        gr.update(visible=use_adjacency),
    )


def render_extract_tab(demo=None) -> None:
    """Render habitat feature extraction tab."""
    gr.Markdown(
        "Extract radiomics and habitat metrics from habitat maps (`*_habitats.nrrd`) "
        "and original images; outputs CSV tables for machine learning."
    )

    with gr.Row():
        existing_yaml = gr.Textbox(label="Load existing extraction YAML (optional)", scale=4)
        browse_cfg_btn = gr.Button("Browse config", scale=1)

    with gr.Group():
        gr.Markdown("### 1. Paths")
        with gr.Row():
            raw_img_folder = gr.Textbox(label="raw_img_folder *", scale=4)
            raw_btn = gr.Button("Browse", scale=1)
        with gr.Row():
            habitats_map_folder = gr.Textbox(label="habitats_map_folder *", scale=4)
            hab_btn = gr.Button("Browse", scale=1)
        with gr.Row():
            out_dir = gr.Textbox(label="out_dir *", scale=4)
            out_btn = gr.Button("Browse", scale=1)

    with gr.Group():
        gr.Markdown("### 2. Radiomics parameter files")
        with gr.Row():
            params_non_hab = gr.Textbox(
                label="params_file_of_non_habitat *",
                value=DEFAULT_NON_HABITAT_PARAMS_FILE,
            )
            params_hab = gr.Textbox(
                label="params_file_of_habitat *",
                value=DEFAULT_HABITAT_PARAMS_FILE,
            )

    with gr.Group():
        gr.Markdown("### 3. Extraction controls")
        with gr.Row():
            habitat_pattern = gr.Textbox(label="habitat_pattern *", value="*_habitats.nrrd")
            n_processes = gr.Number(label="n_processes", value=4, minimum=1, maximum=32, step=1)
        feature_types = gr.CheckboxGroup(
            label="feature_types *",
            choices=FEATURE_TYPE_CHOICES,
            value=DEFAULT_FEATURE_TYPES,
        )
        n_habitats = gr.Number(
            label="n_habitats (0 = auto-detect)",
            value=0,
            minimum=0,
            maximum=20,
            step=1,
        )
        debug = gr.Checkbox(label="debug", value=False)

    with gr.Group(visible=GRAPH_PLUGIN_AVAILABLE) as graph_params_box:
        gr.Markdown("### 4. Graph features (when `graph` is selected)")
        gr.Markdown(
            "Graph topology and edge rules for habitat_graph_features.csv. "
            "Disable this section by unchecking **graph** under feature_types."
        )

        gr.Markdown("**Graph scope**")
        with gr.Row():
            graph_include_single = gr.Checkbox(
                label="include_single_habitat_graph",
                value=True,
                info="Within-habitat region graph for each habitat label.",
            )
            graph_include_pairwise = gr.Checkbox(
                label="include_pairwise_habitat_graph",
                value=True,
                info="Pairwise inter-habitat region graphs.",
            )
            graph_pairwise_intra = gr.Checkbox(
                label="pairwise_include_intra_edges",
                value=True,
                info="Same-habitat proximity edges in pairwise graphs (modularity, etc.).",
            )

        gr.Markdown("**Edge construction**")
        with gr.Row():
            graph_edge_method = gr.Dropdown(
                label="edge_method *",
                choices=GRAPH_EDGE_METHOD_CHOICES,
                value="centroid_distance",
                info="How adjacent regions are linked: centroid distance or face contact.",
            )
            graph_edge_weight = gr.Dropdown(
                label="edge_weight",
                choices=GRAPH_EDGE_WEIGHT_CHOICES,
                value="none",
                info="Optional numeric weight stored on each edge.",
            )
        with gr.Row(visible=True) as graph_distance_row:
            graph_distance_threshold = gr.Number(
                label="distance_threshold (voxels)",
                value=5.0,
                minimum=0.0,
                info="Max centroid distance for an edge when edge_method=centroid_distance.",
            )
        with gr.Row(visible=False) as graph_adjacency_row:
            graph_adjacency_connectivity = gr.Dropdown(
                label="adjacency_connectivity",
                choices=GRAPH_ADJACENCY_CONNECTIVITY_CHOICES,
                value="face",
                info=(
                    "Neighbor rule for spatial adjacency edges: "
                    "face = 6-conn (3D), edge = 18-conn, corner = 26-conn."
                ),
            )
            graph_adjacency_min_voxels = gr.Number(
                label="adjacency_min_voxels",
                value=1,
                minimum=1,
                precision=0,
                info="Minimum adjacent voxel pairs required to create an edge.",
            )

        gr.Markdown("**Region nodes and subdivision**")
        with gr.Row():
            graph_min_region_voxels = gr.Number(
                label="min_region_voxels",
                value=1,
                minimum=1,
                precision=0,
                info="Drop connected regions smaller than this (voxels).",
            )
            graph_connectivity = gr.Dropdown(
                label="connectivity",
                choices=GRAPH_CONNECTIVITY_CHOICES,
                value="face",
                info="Neighborhood for connected-component labeling.",
            )
            graph_erosion_radius = gr.Number(
                label="erosion_radius",
                value=1,
                minimum=0,
                precision=0,
                info="Erode boundary before labeling; 0 disables.",
            )
        with gr.Row():
            graph_subdivide_region_voxels = gr.Number(
                label="subdivide_region_voxels",
                value=1000,
                minimum=0,
                precision=0,
                info="Split large components into grid blocks; 0 disables.",
            )
            graph_block_size = gr.Number(
                label="block_size (voxels)",
                value=5,
                minimum=1,
                precision=0,
                info="Grid block edge length; keep near distance_threshold.",
            )
            graph_block_min_coverage = gr.Number(
                label="block_min_coverage",
                value=0.5,
                minimum=0.0,
                maximum=1.0,
                info="Minimum occupied fraction of a block to keep as a node.",
            )

        gr.Markdown("**Graph visualization (optional figures)**")
        graph_visualize = gr.Checkbox(
            label="visualize",
            value=False,
            info="Save topology figures under out_dir/visualizations/graph.",
        )
        with gr.Row():
            graph_visualization_format = gr.Dropdown(
                label="visualization_format",
                choices=["pdf", "png", "both"],
                value="both",
            )
            graph_visualization_dpi = gr.Number(
                label="visualization_dpi",
                value=600,
                minimum=72,
                maximum=2400,
                step=1,
            )
        graph_visualization_show_background = gr.Checkbox(
            label="visualization_show_background",
            value=True,
            info=(
                "Overlay semi-transparent habitat partitions behind 2D network graphs "
                "(same colors as habitat_slice)."
            ),
        )
        graph_visualization_save_3d = gr.Checkbox(
            label="visualization_save_3d",
            value=True,
            info=(
                "Render PyVista 3D surface and network figures. "
                "Requires pyvista + scikit-image; uncheck to skip on headless servers."
            ),
        )

        gr.Markdown("**Extended graph metrics (efficiency, small-world, rich-club)**")
        graph_include_extended_metrics = gr.Checkbox(
            label="include_extended_metrics",
            value=True,
            info=(
                "Add global/local efficiency, small-world sigma, rich-club coefficient, "
                "and node-distribution summaries to habitat_graph_features.csv."
            ),
        )
        graph_extended_min_nodes = gr.Number(
            label="extended_min_nodes",
            value=10,
            minimum=3,
            precision=0,
            info="Minimum nodes required to compute small-world sigma.",
        )

    # Preserve unknown graph keys when loading an existing YAML.
    graph_block_state = gr.State({})
    with gr.Row():
        submit_btn = gr.Button("Validate and run feature extraction", variant="primary")
        stop_btn = gr.Button("Stop", interactive=False)
    log_output = render_console_log(lines=15, elem_id="habit-log-extract")
    open_dir_btn = gr.Button("Open output folder", visible=False)

    def browse_file() -> Any:
        p = select_local_path("file", "Select YAML")
        return p if p else gr.update()

    def browse_folder() -> Any:
        p = select_local_path("folder", "Select folder")
        return p if p else gr.update()

    browse_cfg_btn.click(browse_file, outputs=existing_yaml)
    raw_btn.click(browse_folder, outputs=raw_img_folder)
    hab_btn.click(browse_folder, outputs=habitats_map_folder)
    out_btn.click(browse_folder, outputs=out_dir)

    def load_yaml(path: str) -> List[Any]:
        noop = gr.update()
        if not path or not os.path.exists(path):
            return [noop] * _LOAD_WIDGET_COUNT + [{}]
        loaded = load_config_yaml(path)
        if not loaded:
            return [noop] * _LOAD_WIDGET_COUNT + [{}]
        graph_cfg: Dict[str, Any] = loaded.get("graph") or {}
        if not isinstance(graph_cfg, dict):
            graph_cfg = {}
        nh = loaded.get("n_habitats")
        edge_method = str(_graph_cfg_value(graph_cfg, "edge_method", "centroid_distance"))
        return [
            loaded.get("raw_img_folder", ""),
            loaded.get("habitats_map_folder", ""),
            loaded.get("out_dir", ""),
            loaded.get("params_file_of_non_habitat", DEFAULT_NON_HABITAT_PARAMS_FILE),
            loaded.get("params_file_of_habitat", DEFAULT_HABITAT_PARAMS_FILE),
            loaded.get("habitat_pattern", "*_habitats.nrrd"),
            int(loaded.get("n_processes", 4)),
            loaded.get("feature_types", FEATURE_TYPE_CHOICES),
            int(nh) if nh else 0,
            loaded.get("debug", False),
            bool(_graph_cfg_value(graph_cfg, "include_single_habitat_graph", True)),
            bool(_graph_cfg_value(graph_cfg, "include_pairwise_habitat_graph", True)),
            edge_method,
            float(_graph_cfg_value(graph_cfg, "distance_threshold", 5.0)),
            str(_graph_cfg_value(graph_cfg, "adjacency_connectivity", "face")),
            int(_graph_cfg_value(graph_cfg, "adjacency_min_voxels", 1)),
            str(_graph_cfg_value(graph_cfg, "edge_weight", "none")),
            int(_graph_cfg_value(graph_cfg, "min_region_voxels", 1)),
            str(_graph_cfg_value(graph_cfg, "connectivity", "face")),
            int(_graph_cfg_value(graph_cfg, "erosion_radius", 1)),
            int(_graph_cfg_value(graph_cfg, "subdivide_region_voxels", 1000)),
            int(_graph_cfg_value(graph_cfg, "block_size", 5)),
            float(_graph_cfg_value(graph_cfg, "block_min_coverage", 0.5)),
            bool(_graph_cfg_value(graph_cfg, "pairwise_include_intra_edges", True)),
            bool(_graph_cfg_value(graph_cfg, "visualize", False)),
            str(_graph_cfg_value(graph_cfg, "visualization_format", "both")),
            int(_graph_cfg_value(graph_cfg, "visualization_dpi", 600)),
            bool(_graph_cfg_value(graph_cfg, "visualization_show_background", True)),
            bool(_graph_cfg_value(graph_cfg, "visualization_save_3d", True)),
            bool(_graph_cfg_value(graph_cfg, "include_extended_metrics", True)),
            int(_graph_cfg_value(graph_cfg, "extended_min_nodes", 10)),
            gr.update(visible=edge_method == "centroid_distance"),
            gr.update(visible=edge_method == "adjacency"),
            graph_cfg,
        ]

    load_yaml_outputs: List[Any] = [
        raw_img_folder, habitats_map_folder, out_dir,
        params_non_hab, params_hab, habitat_pattern, n_processes,
        feature_types, n_habitats, debug,
        graph_include_single, graph_include_pairwise,
        graph_edge_method, graph_distance_threshold,
        graph_adjacency_connectivity, graph_adjacency_min_voxels,
        graph_edge_weight, graph_min_region_voxels, graph_connectivity,
        graph_erosion_radius, graph_subdivide_region_voxels, graph_block_size,
        graph_block_min_coverage, graph_pairwise_intra,
        graph_visualize, graph_visualization_format, graph_visualization_dpi,
        graph_visualization_show_background, graph_visualization_save_3d,
        graph_include_extended_metrics, graph_extended_min_nodes,
        graph_distance_row, graph_adjacency_row,
        graph_block_state,
    ]
    existing_yaml.change(load_yaml, inputs=existing_yaml, outputs=load_yaml_outputs)

    feature_types.change(
        _toggle_graph_section,
        inputs=feature_types,
        outputs=[graph_params_box],
    )
    graph_edge_method.change(
        _toggle_edge_method_fields,
        inputs=graph_edge_method,
        outputs=[graph_distance_row, graph_adjacency_row],
    )

    graph_run_inputs: List[Any] = [
        graph_include_single,
        graph_include_pairwise,
        graph_edge_method,
        graph_distance_threshold,
        graph_adjacency_connectivity,
        graph_adjacency_min_voxels,
        graph_edge_weight,
        graph_min_region_voxels,
        graph_connectivity,
        graph_erosion_radius,
        graph_subdivide_region_voxels,
        graph_block_size,
        graph_block_min_coverage,
        graph_pairwise_intra,
        graph_visualize,
        graph_visualization_format,
        graph_visualization_dpi,
        graph_visualization_show_background,
        graph_visualization_save_3d,
        graph_include_extended_metrics,
        graph_extended_min_nodes,
        graph_block_state,
    ]

    def run_extract(
        raw_val: str,
        hab_val: str,
        out_val: str,
        p_non: str,
        p_hab: str,
        pattern: str,
        n_proc: float,
        feats: List[str],
        n_hab: float,
        debug_val: bool,
        graph_include_single_val: bool,
        graph_include_pairwise_val: bool,
        graph_edge_method_val: str,
        graph_distance_threshold_val: float,
        graph_adjacency_connectivity_val: str,
        graph_adjacency_min_voxels_val: float,
        graph_edge_weight_val: str,
        graph_min_region_voxels_val: float,
        graph_connectivity_val: str,
        graph_erosion_radius_val: float,
        graph_subdivide_region_voxels_val: float,
        graph_block_size_val: float,
        graph_block_min_coverage_val: float,
        graph_pairwise_intra_val: bool,
        graph_visualize_val: bool,
        graph_visualization_format_val: str,
        graph_visualization_dpi_val: float,
        graph_visualization_show_background_val: bool,
        graph_visualization_save_3d_val: bool,
        graph_include_extended_metrics_val: bool,
        graph_extended_min_nodes_val: float,
        graph_state_val: Optional[Dict[str, Any]],
    ):
        if not raw_val or not hab_val or not out_val:
            yield "❌ raw_img_folder, habitats_map_folder, and out_dir are required.", gr.update(visible=False)
            return
        if not feats:
            yield "❌ Select at least one feature_types entry.", gr.update(visible=False)
            return

        # Resolve relative paths to absolute before saving to YAML.
        out_abs = abs_path(out_val)

        config_data: Dict[str, Any] = {
            "raw_img_folder": abs_path(raw_val),
            "habitats_map_folder": abs_path(hab_val),
            "out_dir": out_abs,
            "params_file_of_non_habitat": abs_path(p_non) if p_non and p_non.strip() else p_non,
            "params_file_of_habitat": abs_path(p_hab) if p_hab and p_hab.strip() else p_hab,
            "habitat_pattern": pattern,
            "n_processes": int(n_proc),
            "feature_types": list(feats),
            "n_habitats": int(n_hab) if int(n_hab) > 0 else None,
            "debug": debug_val,
        }
        if "graph" in feats:
            config_data["graph"] = _build_graph_config_from_gui(
                graph_state_val,
                graph_include_single_val,
                graph_include_pairwise_val,
                graph_edge_method_val,
                graph_distance_threshold_val,
                graph_adjacency_connectivity_val,
                graph_adjacency_min_voxels_val,
                graph_edge_weight_val,
                graph_min_region_voxels_val,
                graph_connectivity_val,
                graph_erosion_radius_val,
                graph_subdivide_region_voxels_val,
                graph_block_size_val,
                graph_block_min_coverage_val,
                graph_pairwise_intra_val,
                graph_visualize_val,
                graph_visualization_format_val,
                graph_visualization_dpi_val,
                graph_visualization_show_background_val,
                graph_visualization_save_3d_val,
                graph_include_extended_metrics_val,
                int(graph_extended_min_nodes_val),
            )
        try:
            parse_feature_extraction_config(config_data)
            os.makedirs(out_abs, exist_ok=True)
            gui_path = str(Path(out_abs) / "config_extract_gui.yaml")
            save_config_yaml(config_data, gui_path)
            save_gui_draft("extract", gui_path)
            yield f"💾 Config saved to {gui_path}\n🚀 Running extraction...", gr.update(visible=False)
            for log_text in run_background_job(
                run_extract_features,
                kwargs={"config_file": gui_path},
                log_file=Path(out_abs) / "processing.log",
            ):
                yield log_text, gr.update(visible=True)
        except ValidationError as err:
            msgs = translate_pydantic_error(err)
            yield "⚠️ Validation errors:\n" + "\n".join(f"- {m}" for m in msgs), gr.update(visible=False)
        except Exception as exc:  # noqa: BLE001
            val_msgs = extract_validation_msgs(exc)
            if val_msgs:
                yield "⚠️ Validation errors:\n" + "\n".join(f"- {m}" for m in val_msgs), gr.update(visible=False)
            else:
                yield f"❌ Failed: {exc}", gr.update(visible=False)

    submit_btn.click(
        job_start_button_updates,
        outputs=[submit_btn, stop_btn],
    ).then(
        run_extract,
        inputs=[
            raw_img_folder, habitats_map_folder, out_dir,
            params_non_hab, params_hab, habitat_pattern, n_processes,
            feature_types, n_habitats, debug,
            *graph_run_inputs,
        ],
        outputs=[log_output, open_dir_btn],
    ).then(
        job_end_button_updates,
        outputs=[submit_btn, stop_btn],
    )
    stop_btn.click(on_stop_job_click, inputs=[], outputs=[])
    open_dir_btn.click(lambda p: open_directory(p), inputs=out_dir)

    if demo is not None:
        # Restore only the YAML path; existing_yaml.change() then reloads all fields.
        def _restore_extract_path() -> str:
            return load_gui_draft("extract") or ""
        demo.load(_restore_extract_path, inputs=[], outputs=[existing_yaml])
