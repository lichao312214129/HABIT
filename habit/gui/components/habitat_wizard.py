# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Radiologist-oriented habitat analysis wizard (Gradio).

Four-step flow: template → data → parameters → review/run, plus a results panel.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import yaml
from pydantic import ValidationError

from habit.cli_commands.commands.cmd_habitat import run_habitat
from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
from habit.gui.components.feature_construction_utils import (
    VOXEL_FEATURE_TYPE_LABELS,
    VOXEL_FEATURE_TYPES,
    voxel_radiomics_panel_visible,
)
from habit.gui.components.prep_methods_editor import (
    ALL_PREP_METHODS,
    DROPPING_PREP_METHODS,
    PREP_PRESET_FULL_FILTERING_GROUP,
    PREP_PRESET_FULL_FILTERING_SUBJECT,
    PREP_PRESET_MINIMAL,
    PREP_PRESET_STANDARD_GROUP,
    PREP_PRESET_STANDARD_SUBJECT,
    render_prep_methods_editor,
)
from habit.gui.data_inspector import (
    default_modality_selection,
    format_inspection_table,
    inspect_habitat_dataset,
)
from habit.gui.habitat_config_builder import build_habitat_config_data
from habit.gui.habitat_summary import (
    PREP_PRESET_LABELS,
    SELECTION_METHOD_LABELS,
    build_habitat_review_summary,
    wizard_step_title,
)
from habit.gui.job_controls import (
    job_end_button_updates,
    job_start_button_updates,
    on_stop_job_click,
)
from habit.gui.path_picker import PathPickerRegistry
from habit.gui.pipeline_runner import run_background_job
from habit.gui.template_loader import load_habitat_template, template_choices, template_description
from habit.gui.project.context import ProjectContext
from habit.gui.project.step_hooks import finalize_step_from_log, mark_step_running
from habit.gui.step_integration import register_project_path_fill
from habit.gui.step_registry import register_step_paths
from habit.gui.components.template_cards import (
    HABIT_TEMPLATE_CARD_JS,
    render_template_cards_html,
)
from habit.gui.utils import (
    load_gui_draft,
    open_directory,
    read_pipeline_log,
    render_console_log,
    save_config_yaml,
    save_gui_draft,
    translate_pydantic_error,
    extract_validation_msgs,
)
from habit.utils.parallel_utils import default_cluster_search_workers

from habit.gui.utils import default_radiomics_param

DEFAULT_VOXEL_PARAMS_FILE: str = default_radiomics_param("params_voxel_radiomics.yaml")
DEFAULT_SUPERVOXEL_PARAMS_FILE: str = default_radiomics_param("params_supervoxel_radiomics.yaml")

DEFAULT_SUBJECT_METHODS: List[Dict[str, Any]] = list(PREP_PRESET_STANDARD_SUBJECT)
DEFAULT_GROUP_METHODS: List[Dict[str, Any]] = list(PREP_PRESET_STANDARD_GROUP)

SUBJECT_PREP_ALLOWED: List[str] = [
    m for m in ALL_PREP_METHODS if m not in DROPPING_PREP_METHODS
]
SUBJECT_PREP_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "minimal": list(PREP_PRESET_MINIMAL),
    "standard": list(PREP_PRESET_STANDARD_SUBJECT),
    "full": list(PREP_PRESET_FULL_FILTERING_SUBJECT),
}
GROUP_PREP_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "minimal": list(PREP_PRESET_MINIMAL),
    "standard": list(PREP_PRESET_STANDARD_GROUP),
    "full": list(PREP_PRESET_FULL_FILTERING_GROUP),
}

SELECTION_METHODS: List[str] = [
    "elbow", "silhouette", "bic", "aic", "davies_bouldin",
    "calinski_harabasz", "inertia", "kneedle", "gap",
]
CLUSTERING_MODE_CHOICES: List[tuple[str, str]] = [
    ("Two-step (supervoxel → habitat, recommended)", "two_step"),
    ("One-step (per-subject, exploratory)", "one_step"),
    ("Direct pooling (all voxels pooled)", "direct_pooling"),
]
PREP_PRESET_CHOICES: List[tuple[str, str]] = [
    (PREP_PRESET_LABELS["minimal"], "minimal"),
    (PREP_PRESET_LABELS["standard"], "standard"),
    (PREP_PRESET_LABELS["full"], "full"),
]
CC_CONNECTIVITY_CHOICES: List[str] = ["1 (6-neighbor)", "2 (18-neighbor)", "3 (26-neighbor)"]


def _step_visibility(active: int) -> Tuple[Any, Any, Any, Any, Any]:
    """Return gr.update visibility for wizard steps 1–5."""
    return tuple(gr.update(visible=(active == step)) for step in range(1, 6))


def _apply_template(template_id: str) -> Tuple[Any, ...]:
    """Load template defaults into wizard controls."""
    doc = load_habitat_template(template_id) or {}
    desc = template_description(doc)
    clustering = str(doc.get("clustering_mode", "two_step"))
    feat = str(doc.get("voxel_feature_type", "raw"))
    prep = str(doc.get("prep_preset", "standard"))
    expected = list(doc.get("expected_modalities", []))
    return (
        desc,
        clustering,
        feat,
        prep,
        gr.update(choices=expected, value=expected),
    )


def _scan_data(data_dir: str, template_id: str, selected: List[str]) -> Tuple[Any, ...]:
    """Scan dataset and refresh modality checkbox group."""
    report = inspect_habitat_dataset(data_dir)
    doc = load_habitat_template(template_id) or {}
    choices = list(report.modalities) or list(doc.get("expected_modalities", []))
    if not choices:
        choices = ["T2"]
    default_sel = selected or default_modality_selection(
        report,
        list(doc.get("expected_modalities", [])),
    )
    default_sel = [m for m in default_sel if m in choices]
    return (
        format_inspection_table(report),
        report.status_message,
        gr.update(choices=choices, value=default_sel),
        report.n_subjects,
        report.n_ok,
    )


def _clustering_panels(mode: str, algo: str) -> Tuple[Any, Any, Any, Any]:
    """Toggle panels for clustering mode and supervoxel algorithm."""
    is_two = mode == "two_step"
    is_one = mode == "one_step"
    is_slic = is_two and algo == "slic"
    return (
        gr.update(visible=is_two),
        gr.update(visible=is_two),
        gr.update(visible=is_one),
        gr.update(visible=is_slic),
    )


def _build_review(
    template_id: str,
    data_dir: str,
    out_dir: str,
    n_subjects: int,
    n_ok: int,
    modalities: List[str],
    clustering_mode: str,
    sv_n: float,
    habitat_auto: bool,
    habitat_n: float,
    habitat_sel: str,
    feature_type: str,
    prep_preset: str,
    run_mode: str,
) -> str:
    """Build Chinese review summary text."""
    doc = load_habitat_template(template_id) or {}
    name = str(doc.get("display_name", template_id))
    return build_habitat_review_summary(
        template_name=name,
        data_dir=data_dir,
        out_dir=out_dir,
        n_subjects=int(n_subjects or 0),
        n_ok=int(n_ok or 0),
        modalities=list(modalities or []),
        clustering_mode=clustering_mode,
        sv_n_clusters=int(sv_n),
        habitat_auto=habitat_auto,
        habitat_n=int(habitat_n),
        habitat_selection=habitat_sel,
        feature_type=feature_type,
        prep_preset=prep_preset,
        run_mode=run_mode,
    )


def _collect_config_kwargs(args: Tuple[Any, ...]) -> Dict[str, Any]:
    """Map flat form_inputs tuple to build_habitat_config_data keyword args."""
    (
        run_mode, processes, data_dir, out_dir, pipeline_path, clustering_mode,
        sv_algo, sv_n, sv_max_iter, sv_n_init, sv_compact, sv_sigma, sv_enforce, sv_rs,
        os_min, os_max, os_fixed_n, os_sel, os_plot,
        hb_algo, hb_fixed_en, hb_fixed_n, hb_min, hb_max, hb_sel,
        hb_max_iter, hb_n_init, hb_par, hb_workers, hb_rs,
        pp_sv_en, pp_sv_min, pp_sv_maxiter, pp_sv_conn,
        pp_hab_en, pp_hab_min, pp_hab_maxiter, pp_hab_conn,
        voxel_feature_type, selected_modalities, voxel_use_custom_dsl, voxel_method,
        voxel_pf, voxel_kr, voxel_batch, use_torch, torch_gpus, torch_device,
        sv_agg, sv_rad_mod, sv_kw, sv_lm, sv_lpf, sv_crop, use_sv_cext, sv_batch, sv_pad,
        prep_preset, simple_mode, hb_auto, hb_simple_n,
        subj_methods, group_methods,
        resume, plot_curves, save_images, save_results, hab_fmt, rs, debug,
        cap_gpu, subj_to, spawn_to, graceful_sd, on_fail,
        oom_back, oom_red, strict_ck, ck_dir, force_rerun,
        retry_fail, retry_rounds, par_mode, pers_max_fail, pers_recycle, clear_ck, verbose,
    ) = args

    fixed_en = bool(hb_fixed_en)
    fixed_n = int(hb_fixed_n)
    if simple_mode:
        fixed_en = not bool(hb_auto)
        if fixed_en:
            fixed_n = int(hb_simple_n)

    return dict(
        run_mode=run_mode,
        processes=int(processes),
        data_dir=data_dir,
        out_dir=out_dir,
        pipeline_path=pipeline_path or "",
        clustering_mode=clustering_mode,
        sv_algo=sv_algo,
        sv_n_clusters=int(sv_n),
        sv_max_iter=int(sv_max_iter),
        sv_n_init=int(sv_n_init),
        sv_compactness=float(sv_compact),
        sv_sigma=float(sv_sigma),
        sv_enforce_conn=bool(sv_enforce),
        sv_rs=int(sv_rs),
        os_min=int(os_min),
        os_max=int(os_max),
        os_fixed_n=int(os_fixed_n),
        os_sel=os_sel,
        os_plot=bool(os_plot),
        hb_algo=hb_algo,
        hb_fixed_en=fixed_en,
        hb_fixed_n=fixed_n,
        hb_min=int(hb_min),
        hb_max=int(hb_max),
        hb_sel=hb_sel,
        hb_max_iter=int(hb_max_iter),
        hb_n_init=int(hb_n_init),
        hb_par=bool(hb_par),
        hb_workers=int(hb_workers),
        hb_rs=int(hb_rs),
        pp_sv_en=bool(pp_sv_en),
        pp_sv_min=int(pp_sv_min),
        pp_sv_maxiter=int(pp_sv_maxiter),
        pp_sv_conn=pp_sv_conn,
        pp_hab_en=bool(pp_hab_en),
        pp_hab_min=int(pp_hab_min),
        pp_hab_maxiter=int(pp_hab_maxiter),
        pp_hab_conn=pp_hab_conn,
        voxel_feature_type=voxel_feature_type,
        selected_modalities=list(selected_modalities or []),
        voxel_use_custom_dsl=bool(voxel_use_custom_dsl),
        voxel_method=voxel_method,
        voxel_pf=voxel_pf,
        voxel_kr=int(voxel_kr),
        voxel_batch=int(voxel_batch),
        use_torch=use_torch,
        torch_gpus=torch_gpus,
        torch_device=torch_device,
        sv_agg=sv_agg,
        sv_rad_mod=sv_rad_mod,
        sv_kw=sv_kw,
        sv_lm=sv_lm,
        sv_lpf=sv_lpf,
        sv_crop=bool(sv_crop),
        use_sv_cext=use_sv_cext,
        sv_batch=int(sv_batch),
        sv_pad=int(sv_pad),
        prep_preset=prep_preset,
        subj_methods=subj_methods,
        group_methods=group_methods,
        simple_mode=bool(simple_mode),
        resume=bool(resume),
        plot_curves=bool(plot_curves),
        save_images=bool(save_images),
        save_results=bool(save_results),
        hab_fmt=hab_fmt,
        rs=int(rs),
        debug=bool(debug),
        cap_gpu=bool(cap_gpu),
        subj_to=float(subj_to),
        spawn_to=float(spawn_to),
        graceful_sd=float(graceful_sd),
        on_fail=on_fail,
        oom_back=bool(oom_back),
        oom_red=int(oom_red),
        strict_ck=bool(strict_ck),
        ck_dir=ck_dir or "",
        force_rerun=force_rerun or "",
        retry_fail=bool(retry_fail),
        retry_rounds=int(retry_rounds),
        par_mode=par_mode,
        pers_max_fail=int(pers_max_fail),
        pers_recycle=int(pers_recycle),
        clear_ck=bool(clear_ck),
        verbose=bool(verbose),
    )


def render_habitat_tab(
    demo: Optional[Any] = None,
    path_picker: Optional[PathPickerRegistry] = None,
    project_root_state: Optional[Any] = None,
) -> None:
    """
    Render the habitat analysis wizard tab.

    Args:
        demo: Parent Gradio Blocks for session restore hooks.
        path_picker: Shared path picker registry.
        project_root_state: Optional gr.State holding current project root path.
    """
    picker = path_picker if path_picker is not None else PathPickerRegistry()

    gr.Markdown(
        "### Habitat segmentation\n"
        "Step-by-step wizard for clinical users. Fields marked **\\*** are required."
        " Follow **Template → Data → Parameters → Review**."
    )

    wizard_step = gr.State(value=1)
    n_subjects_state = gr.State(value=0)
    n_ok_state = gr.State(value=0)

    step_title = gr.Markdown(wizard_step_title(1))

    with gr.Column(visible=True) as step1_box:
        gr.Markdown("#### Step 1 — Choose template")
        gr.Markdown(
            "Pick the analysis template that best matches your study. "
            "Each template pre-fills the parameters in later steps; you can "
            "still tweak them in Step 3."
        )
        # Card grid showing all templates visually
        template_cards = gr.HTML(value=render_template_cards_html("liver_dce_two_step"))
        # Radio for selection (cleaner than Dropdown for 5 items)
        _tpl_choices = template_choices()
        template_id = gr.Radio(
            label="Select template *",
            choices=_tpl_choices,
            value="liver_dce_two_step",
            visible=False,  # hidden; cards are the primary UI
        )
        # Visible selection indicator
        template_sel = gr.Radio(
            label="Select template *",
            choices=_tpl_choices,
            value="liver_dce_two_step",
        )
        template_desc = gr.Markdown("")
        with gr.Accordion("Load existing config (optional)", open=False):
            existing_yaml = gr.Textbox(label="Config file path", scale=4)
            browse_cfg_btn = gr.Button("Browse", scale=1)
        picker.add(browse_cfg_btn, existing_yaml, pick="file")

    with gr.Column(visible=False) as step2_box:
        gr.Markdown("#### Step 2 — Data and sequences")
        with gr.Row():
            run_mode = gr.Dropdown(
                label="Run mode *",
                choices=[("Train (build habitat model)", "train"), ("Predict (apply saved model)", "predict")],
                value="train",
            )
            processes = gr.Number(label="Parallel processes", value=2, minimum=1, maximum=32, step=1)
        with gr.Row():
            data_dir = gr.Textbox(
                label="Data directory *",
                scale=4,
                info="Preprocessed images/masks root",
            )
            data_btn = gr.Button("Browse", scale=1)
        picker.add(data_btn, data_dir, pick="folder")
        with gr.Row():
            out_dir = gr.Textbox(
                label="Output directory *",
                scale=4,
                info="Usually the project 03_habitat folder",
            )
            out_btn = gr.Button("Browse", scale=1)
        picker.add(out_btn, out_dir, pick="folder")
        with gr.Row(visible=False) as pipe_row:
            pipeline_path = gr.Textbox(label="Pipeline file * (predict mode)", scale=4)
            pipe_btn = gr.Button("Browse", scale=1)
        picker.add(pipe_btn, pipeline_path, pick="file")
        scan_btn = gr.Button("Scan data directory", variant="secondary")
        modality_status = gr.Textbox(label="Scan status", interactive=False)
        modality_checks = gr.CheckboxGroup(
            label="Modalities for analysis *",
            choices=[],
            value=[],
            info="Select image sequences used for habitat analysis (e.g. delay2, T2).",
        )
        inspection_table = gr.Textbox(
            label="Data quality check",
            lines=8,
            interactive=False,
            info="Verify ROI masks and modality folders for each subject.",
        )

    with gr.Column(visible=False) as step3_box:
        gr.Markdown("#### Step 3 — Parameters")
        simple_mode = gr.Checkbox(label="Simple mode (recommended)", value=True)
        clustering_mode = gr.Dropdown(
            label="Clustering mode *",
            choices=CLUSTERING_MODE_CHOICES,
            value="two_step",
        )
        with gr.Row():
            sv_n_clusters = gr.Slider(
                label="Intra-tumor partitions (two-step)",
                minimum=10,
                maximum=200,
                step=5,
                value=50,
                info="Higher values yield finer partitions and longer runtime.",
            )
        with gr.Row():
            hb_auto = gr.Checkbox(label="Auto-select habitat count (recommended)", value=True)
            hb_simple_n = gr.Number(
                label="Fixed habitat count",
                value=3,
                minimum=2,
                maximum=20,
                precision=0,
                visible=False,
            )
            hb_selection = gr.Dropdown(
                label="Selection method",
                choices=[(SELECTION_METHOD_LABELS[m], m) for m in SELECTION_METHODS],
                value="elbow",
            )
        voxel_feature_type = gr.Dropdown(
            label="Voxel feature type",
            choices=[(VOXEL_FEATURE_TYPE_LABELS[t], t) for t in VOXEL_FEATURE_TYPES],
            value="raw",
        )
        prep_preset = gr.Dropdown(label="Feature preprocessing", choices=PREP_PRESET_CHOICES, value="standard")

        with gr.Accordion("Expert settings", open=False, visible=False) as expert_box:
            with gr.Group(visible=True) as box_supervoxel:
                gr.Markdown("**Supervoxel clustering (two-step)**")
                sv_algo = gr.Dropdown(
                    label="Supervoxel algorithm",
                    choices=[("K-Means", "kmeans"), ("GMM", "gmm"), ("SLIC", "slic")],
                    value="kmeans",
                )
                with gr.Row():
                    sv_max_iter = gr.Number(label="Max iterations", value=300, precision=0)
                    sv_n_init = gr.Number(label="N init", value=10, precision=0)
                with gr.Row(visible=False) as slic_row:
                    sv_compactness = gr.Number(label="SLIC compactness", value=0.1)
                    sv_sigma = gr.Number(label="SLIC sigma", value=0.5)
                sv_enforce_conn = gr.Checkbox(label="SLIC enforce connectivity", value=True)
                sv_random_state = gr.Number(label="Random seed (-1 = inherit global)", value=-1, precision=0)
            with gr.Group(visible=False) as box_one_step:
                gr.Markdown("**One-step settings**")
                os_min = gr.Number(label="Min habitats", value=2, precision=0)
                os_max = gr.Number(label="Max habitats", value=10, precision=0)
                os_fixed_n = gr.Number(label="Fixed habitats (0 = auto)", value=0, precision=0)
                os_selection = gr.Dropdown(
                    label="Selection method",
                    choices=[(SELECTION_METHOD_LABELS[m], m) for m in SELECTION_METHODS],
                    value="elbow",
                )
                os_plot_curves = gr.Checkbox(label="Plot validation curves", value=True)
            gr.Markdown("**Habitat clustering (expert)**")
            hb_algo = gr.Dropdown(
                label="Habitat algorithm",
                choices=[("K-Means", "kmeans"), ("GMM", "gmm")],
                value="kmeans",
            )
            hb_fixed_n_enabled = gr.Checkbox(label="Use fixed habitat count", value=False)
            with gr.Row():
                hb_min_clusters = gr.Number(label="Min habitats", value=2, precision=0)
                hb_max_clusters = gr.Number(label="Max habitats", value=10, precision=0)
            hb_fixed_n = gr.Number(label="Fixed habitat count", value=3, precision=0)
            with gr.Row():
                hb_max_iter = gr.Number(label="Max iterations", value=300, precision=0)
                hb_n_init = gr.Number(label="N init", value=10, precision=0)
            hb_parallel_search = gr.Checkbox(label="Parallel cluster search", value=True)
            hb_search_workers = gr.Number(
                label="Search workers (0 = auto)",
                value=default_cluster_search_workers(),
                precision=0,
            )
            hb_random_state = gr.Number(label="Random seed (-1 = inherit)", value=-1, precision=0)
            with gr.Row(visible=False) as voxel_radiomics_row:
                voxel_params_file = gr.Textbox(
                    label="Voxel radiomics params file",
                    value=DEFAULT_VOXEL_PARAMS_FILE,
                )
                voxel_kernel = gr.Number(label="kernelRadius", value=3, precision=0)
            voxel_batch = gr.Number(label="voxelBatch", value=1000, precision=0)
            use_torch = gr.Dropdown(
                label="useTorchRadiomics",
                choices=["auto", "true", "false"],
                value="auto",
            )
            torch_gpus = gr.Textbox(label="GPU IDs (comma-separated, empty = all)", value="")
            torch_device = gr.Textbox(label="torchDevice (optional)", value="")
            voxel_use_custom_dsl = gr.Checkbox(label="Edit voxel DSL manually", value=False)
            voxel_method = gr.Textbox(label="voxel_level.method", value="concat(raw(T2))")
            sv_agg = gr.Dropdown(
                label="Supervoxel aggregation",
                choices=[
                    ("Mean of voxel features", "mean_voxel_features"),
                    ("Supervoxel radiomics", "supervoxel_radiomics"),
                ],
                value="mean_voxel_features",
            )
            sv_rad_mod = gr.Textbox(label="Supervoxel radiomics modality", value="")
            sv_file_kw = gr.Textbox(label="Supervoxel file pattern", value="*_supervoxel.nrrd")
            sv_level_method = gr.Textbox(
                label="supervoxel_level.method",
                value="mean_voxel_features()",
            )
            sv_level_params_file = gr.Textbox(
                label="Supervoxel radiomics params file",
                value=DEFAULT_SUPERVOXEL_PARAMS_FILE,
            )
            sv_union_crop = gr.Checkbox(label="supervoxelUnionBboxCrop", value=True)
            use_sv_cext = gr.Dropdown(
                label="useSupervoxelCext",
                choices=["auto", "true", "false"],
                value="auto",
            )
            sv_batch = gr.Number(label="supervoxelBatch", value=64, precision=0)
            sv_pad_distance = gr.Number(label="supervoxelPadDistance", value=0, precision=0)
            subject_prep_editor = render_prep_methods_editor(
                "**Subject-level preprocessing**",
                default_methods=DEFAULT_SUBJECT_METHODS,
                allowed_methods=SUBJECT_PREP_ALLOWED,
                preset_methods=SUBJECT_PREP_PRESETS,
            )
            group_prep_editor = render_prep_methods_editor(
                "**Group-level preprocessing**",
                default_methods=DEFAULT_GROUP_METHODS,
                allowed_methods=ALL_PREP_METHODS,
                preset_methods=GROUP_PREP_PRESETS,
            )

    pp_sv_enabled = gr.Checkbox(value=False, visible=False)
    pp_sv_min_size = gr.Number(value=30, visible=False)
    pp_sv_max_iter = gr.Number(value=3, visible=False)
    pp_sv_connectivity = gr.Dropdown(choices=CC_CONNECTIVITY_CHOICES, value="1 (6-neighbor)", visible=False)
    pp_hab_enabled = gr.Checkbox(value=False, visible=False)
    pp_hab_min_size = gr.Number(value=30, visible=False)
    pp_hab_max_iter = gr.Number(value=3, visible=False)
    pp_hab_connectivity = gr.Dropdown(choices=CC_CONNECTIVITY_CHOICES, value="1 (6-neighbor)", visible=False)

    resume = gr.Checkbox(value=True, visible=False)
    plot_curves = gr.Checkbox(value=True, visible=False)
    save_images = gr.Checkbox(value=True, visible=False)
    save_results_csv = gr.Checkbox(value=True, visible=False)
    habitats_format = gr.Dropdown(choices=["parquet", "csv"], value="parquet", visible=False)
    random_state = gr.Number(value=42, visible=False)
    debug = gr.Checkbox(value=False, visible=False)
    cap_gpu_pool = gr.Checkbox(value=False, visible=False)
    subject_timeout = gr.Number(value=900, visible=False)
    subject_spawn_timeout = gr.Number(value=120, visible=False)
    graceful_shutdown = gr.Number(value=15, visible=False)
    on_subject_failure = gr.Dropdown(choices=["continue", "fail_fast"], value="continue", visible=False)
    oom_backoff = gr.Checkbox(value=True, visible=False)
    oom_reduce = gr.Number(value=1, visible=False)
    strict_checkpoint = gr.Checkbox(value=False, visible=False)
    checkpoint_dir = gr.Textbox(value="", visible=False)
    force_rerun = gr.Textbox(value="", visible=False)
    retry_failed = gr.Checkbox(value=False, visible=False)
    auto_retry_rounds = gr.Number(value=2, visible=False)
    parallel_mode = gr.Dropdown(choices=["persistent", "isolated"], value="persistent", visible=False)
    persistent_max_fails = gr.Number(value=1, visible=False)
    persistent_recycle = gr.Number(value=0, visible=False)
    clear_checkpoint = gr.Checkbox(value=False, visible=False)
    verbose = gr.Checkbox(value=True, visible=False)

    with gr.Column(visible=False) as step4_box:
        gr.Markdown("#### Step 4 — Review and run")
        review_text = gr.Textbox(label="Analysis summary", lines=12, interactive=False)
        with gr.Accordion("YAML preview (technical)", open=False):
            yaml_preview = gr.Textbox(label="YAML", lines=16, interactive=False)
        with gr.Row():
            validate_btn = gr.Button("Validate config", variant="secondary")
            submit_btn = gr.Button("Run habitat analysis", variant="primary")
            stop_btn = gr.Button("Stop", interactive=False)
        validation_msg = gr.Textbox(label="Validation result", interactive=False)
        log_output = render_console_log(lines=16, elem_id="habit-log-habitat")

    with gr.Column(visible=False) as step5_box:
        gr.Markdown("#### Step 5 — Results")
        result_summary = gr.Textbox(label="Run result", lines=6, interactive=False)
        open_dir_btn = gr.Button("Open output folder", visible=False)
        open_viz_btn = gr.Button("Open 3D visualization (if available)", visible=False)

    with gr.Row():
        prev_btn = gr.Button("Previous")
        next_btn = gr.Button("Next", variant="primary")

    def _sync_habitat_simple(auto: bool) -> Any:
        return gr.update(visible=not auto)

    hb_auto.change(_sync_habitat_simple, hb_auto, hb_simple_n)
    simple_mode.change(lambda s: gr.update(visible=not s), simple_mode, expert_box)
    run_mode.change(lambda m: gr.update(visible=m == "predict"), run_mode, pipe_row)

    template_id.change(
        _apply_template,
        template_id,
        [template_desc, clustering_mode, voxel_feature_type, prep_preset, modality_checks],
    )

    scan_outputs = [
        inspection_table,
        modality_status,
        modality_checks,
        n_subjects_state,
        n_ok_state,
    ]
    scan_btn.click(_scan_data, [data_dir, template_id, modality_checks], scan_outputs)
    data_dir.change(_scan_data, [data_dir, template_id, modality_checks], scan_outputs)

    clustering_mode.change(
        _clustering_panels,
        [clustering_mode, sv_algo],
        [box_supervoxel, box_supervoxel, box_one_step, slic_row],
    )
    sv_algo.change(
        _clustering_panels,
        [clustering_mode, sv_algo],
        [box_supervoxel, box_supervoxel, box_one_step, slic_row],
    )
    voxel_feature_type.change(
        lambda ft: gr.update(visible=voxel_radiomics_panel_visible(ft)),
        voxel_feature_type,
        voxel_radiomics_row,
    )

    if project_root_state is not None:
        register_step_paths(
            "habitat",
            ["data_dir", "out_dir"],
            [data_dir, out_dir],
        )

        def _extra_habitat_paths(ctx: ProjectContext) -> Dict[str, str]:
            """Fill pipeline path in predict mode when discoverable."""
            bundle = ctx.paths_for_step("ml")
            pipe = bundle.inputs.get("pipeline_path", "")
            if not pipe:
                return {}
            try:
                from habit.utils.docker_path_utils import to_user_visible_path
                return {"pipeline_path": to_user_visible_path(pipe)}
            except Exception:
                return {"pipeline_path": pipe}

        register_project_path_fill(
            project_root_state,
            "habitat",
            ["data_dir", "out_dir", "pipeline_path"],
            [data_dir, out_dir, pipeline_path],
            extra_fill_fn=_extra_habitat_paths,
        )

    form_inputs: List[Any] = [
        run_mode, processes, data_dir, out_dir, pipeline_path, clustering_mode,
        sv_algo, sv_n_clusters, sv_max_iter, sv_n_init, sv_compactness, sv_sigma,
        sv_enforce_conn, sv_random_state,
        os_min, os_max, os_fixed_n, os_selection, os_plot_curves,
        hb_algo, hb_fixed_n_enabled, hb_fixed_n, hb_min_clusters, hb_max_clusters, hb_selection,
        hb_max_iter, hb_n_init, hb_parallel_search, hb_search_workers, hb_random_state,
        pp_sv_enabled, pp_sv_min_size, pp_sv_max_iter, pp_sv_connectivity,
        pp_hab_enabled, pp_hab_min_size, pp_hab_max_iter, pp_hab_connectivity,
        voxel_feature_type, modality_checks, voxel_use_custom_dsl, voxel_method,
        voxel_params_file, voxel_kernel, voxel_batch, use_torch, torch_gpus, torch_device,
        sv_agg, sv_rad_mod, sv_file_kw, sv_level_method, sv_level_params_file,
        sv_union_crop, use_sv_cext, sv_batch, sv_pad_distance,
        prep_preset, simple_mode, hb_auto, hb_simple_n,
        subject_prep_editor.state, group_prep_editor.state,
        resume, plot_curves, save_images, save_results_csv, habitats_format,
        random_state, debug,
        cap_gpu_pool, subject_timeout, subject_spawn_timeout, graceful_shutdown,
        on_subject_failure, oom_backoff, oom_reduce, strict_checkpoint, checkpoint_dir,
        force_rerun, retry_failed, auto_retry_rounds, parallel_mode,
        persistent_max_fails, persistent_recycle, clear_checkpoint, verbose,
    ]

    review_args_inputs = [
        template_id, data_dir, out_dir, n_subjects_state, n_ok_state, modality_checks,
        clustering_mode, sv_n_clusters, hb_auto, hb_simple_n, hb_selection,
        voxel_feature_type, prep_preset, run_mode,
    ]

    def _navigate(step: int, direction: int, *review_args: Any) -> Tuple[Any, ...]:
        new_step = max(1, min(5, step + direction))
        review = _build_review(*review_args) if new_step == 4 else gr.update()
        if new_step != 4:
            review = ""
        return (
            new_step,
            gr.update(value=wizard_step_title(new_step)),
            *_step_visibility(new_step),
            review,
        )

    nav_outputs = [
        wizard_step, step_title,
        step1_box, step2_box, step3_box, step4_box, step5_box,
        review_text,
    ]
    next_btn.click(
        lambda step, *review_args: _navigate(step, 1, *review_args),
        [wizard_step, *review_args_inputs],
        nav_outputs,
    )
    prev_btn.click(
        lambda step, *review_args: _navigate(step, -1, *review_args),
        [wizard_step, *review_args_inputs],
        nav_outputs,
    )

    def _validate_only(*args: Any) -> Tuple[str, str]:
        try:
            config_data = build_habitat_config_data(**_collect_config_kwargs(args))
            yaml_text = yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False)
            HabitatAnalysisConfig(**config_data)
            return "Configuration valid. Ready to run.", yaml_text
        except ValidationError as err:
            msgs = translate_pydantic_error(err)
            return "Validation errors:\n" + "\n".join(f"- {m}" for m in msgs), ""
        except Exception as exc:  # noqa: BLE001
            return f"Validation failed: {exc}", ""

    def _run_pipeline(*args: Any):
        project_root = args[0] if args else ""
        kwargs = _collect_config_kwargs(args[1:])
        if not kwargs["data_dir"] or not kwargs["out_dir"]:
            yield "data_dir and out_dir are required.", gr.update(visible=False), gr.update(visible=False), ""
            return
        if kwargs["run_mode"] == "predict" and not kwargs["pipeline_path"]:
            yield "pipeline_path is required in predict mode.", gr.update(visible=False), gr.update(visible=False), ""
            return
        if project_root:
            mark_step_running(str(project_root), "habitat")
        try:
            config_data = build_habitat_config_data(**kwargs)
            config = HabitatAnalysisConfig(**config_data)
            out_abs = config.out_dir
            os.makedirs(out_abs, exist_ok=True)
            gui_path = str(Path(out_abs) / "config_habitat_gui.yaml")
            save_config_yaml(config_data, gui_path)
            save_gui_draft("habitat", gui_path)
            yield "Config saved. Running habitat analysis...", gr.update(visible=False), gr.update(visible=False), ""

            def job() -> None:
                run_habitat(
                    config_file=gui_path,
                    debug_mode=config.debug,
                    mode=kwargs["run_mode"],
                    pipeline_path=kwargs["pipeline_path"] if kwargs["run_mode"] == "predict" else None,
                    resume=kwargs["resume"],
                    exit_on_error=False,
                )

            log_file = Path(out_abs) / "habitat_analysis.log"
            last_log = ""
            try:
                for log_text in run_background_job(job, log_file=log_file):
                    last_log = log_text
                    yield log_text, gr.update(visible=True), gr.update(visible=False), ""
            except Exception as run_exc:  # noqa: BLE001
                partial = read_pipeline_log(log_file)
                if partial.strip():
                    last_log = f"{partial}\n\nLog stream error: {run_exc}"
                    yield last_log, gr.update(visible=True), gr.update(visible=False), ""
                else:
                    raise

            viz = Path(out_abs) / "visualizations" / "habitat_clustering" / "habitat_clustering_3D_interactive.html"
            summary = f"Analysis complete.\nOutput: {out_abs}"
            if viz.is_file():
                summary += f"\n3D visualization: {viz}"
            finalize_step_from_log(
                str(project_root) if project_root else "",
                "habitat",
                last_log or summary,
                config_path=gui_path,
                output_dir=str(out_abs),
            )
            yield summary, gr.update(visible=True), gr.update(visible=viz.is_file()), summary
        except ValidationError as err:
            msgs = translate_pydantic_error(err)
            yield "Validation errors:\n" + "\n".join(f"- {m}" for m in msgs), gr.update(visible=False), gr.update(visible=False), ""
        except Exception as exc:  # noqa: BLE001
            val_msgs = extract_validation_msgs(exc)
            if val_msgs:
                yield "Validation errors:\n" + "\n".join(f"- {m}" for m in val_msgs), gr.update(visible=False), gr.update(visible=False), ""
            else:
                partial = read_pipeline_log(Path(kwargs.get("out_dir", "")) / "habitat_analysis.log")
                if partial.strip():
                    yield f"{partial}\n\nFailed: {exc}", gr.update(visible=True), gr.update(visible=False), ""
                else:
                    yield f"Run failed: {exc}", gr.update(visible=False), gr.update(visible=False), ""

    validate_btn.click(_validate_only, form_inputs, [validation_msg, yaml_preview])
    submit_btn.click(job_start_button_updates, outputs=[submit_btn, stop_btn]).then(
        _run_pipeline,
        [project_root_state, *form_inputs] if project_root_state is not None else form_inputs,
        [log_output, open_dir_btn, open_viz_btn, result_summary],
    ).then(job_end_button_updates, outputs=[submit_btn, stop_btn]).then(
        lambda: 5, outputs=[wizard_step],
    ).then(
        lambda: gr.update(value=wizard_step_title(5)), outputs=[step_title],
    ).then(
        lambda: _step_visibility(5),
        outputs=[step1_box, step2_box, step3_box, step4_box, step5_box],
    )
    stop_btn.click(on_stop_job_click, inputs=[], outputs=[])
    open_dir_btn.click(lambda p: open_directory(p), inputs=out_dir)

    def _open_viz(out_path: str) -> str:
        viz_dir = Path(out_path) / "visualizations" / "habitat_clustering"
        if viz_dir.is_dir():
            open_directory(str(viz_dir))
            return f"Opened: {viz_dir}"
        return "Visualization directory not found."

    open_viz_btn.click(_open_viz, out_dir, result_summary)

    # Keep hidden template_id in sync with visible template_sel
    def _sync_template(sel: str) -> tuple:
        """Update cards, description, and hidden state when user picks a template."""
        cards_html = render_template_cards_html(sel)
        desc = template_description(load_habitat_template(sel) or {})
        return sel, cards_html, desc

    template_sel.change(
        _sync_template, [template_sel],
        [template_id, template_cards, template_desc],
    )

    if demo is not None:
        demo.load(lambda: load_gui_draft("habitat") or "", inputs=[], outputs=[existing_yaml])
        demo.load(
            lambda tid: template_description(load_habitat_template(tid) or {}),
            inputs=[template_id],
            outputs=[template_desc],
        )
        demo.load(None, None, None, js=HABIT_TEMPLATE_CARD_JS)

    if path_picker is None:
        picker.finalize()
