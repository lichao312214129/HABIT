# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Habitat clustering analysis tab for Gradio GUI."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from pydantic import ValidationError

from habit.cli_commands.commands.cmd_habitat import run_habitat
from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
from habit.gui.pipeline_runner import run_background_job
from habit.gui.utils import (
    dict_to_yaml_block,
    load_config_yaml,
    open_directory,
    save_config_yaml,
    select_local_path,
    translate_pydantic_error,
    yaml_block_to_dict,
)

DEFAULT_SUBJECT_PREP: str = dict_to_yaml_block({
    "methods": [
        {"method": "winsorize", "winsor_limits": [0.05, 0.05], "global_normalize": False},
        {"method": "minmax", "global_normalize": False},
    ]
})
DEFAULT_GROUP_PREP: str = dict_to_yaml_block({
    "methods": [
        {"method": "winsorize", "winsor_limits": [0.05, 0.05], "global_normalize": False},
        {"method": "variance_filter", "variance_threshold": 0.01, "global_normalize": False},
        {"method": "correlation_filter", "corr_threshold": 0.9, "corr_method": "spearman", "global_normalize": False},
        {"method": "minmax", "global_normalize": False},
    ]
})

SELECTION_METHODS: List[str] = [
    "elbow", "silhouette", "bic", "aic", "davies_bouldin",
    "calinski_harabasz", "inertia", "kneedle", "gap",
]


def _clustering_mode_panels(mode: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Toggle supervoxel / one-step panels based on clustering_mode."""
    return (
        gr.update(visible=mode == "two_step"),
        gr.update(visible=mode == "two_step"),
        gr.update(visible=mode == "one_step"),
    )


def render_habitat_tab() -> None:
    """Render habitat segmentation and analysis tab."""
    gr.Markdown("Configure habitat clustering, feature construction, and run train/predict pipelines.")

    with gr.Row():
        existing_yaml = gr.Textbox(label="Load existing habitat YAML (optional)", scale=4)
        browse_cfg_btn = gr.Button("Browse config", scale=1)

    with gr.Group():
        gr.Markdown("### 1. Mode and paths")
        with gr.Row():
            run_mode = gr.Dropdown(label="run_mode *", choices=["train", "predict"], value="train")
            processes = gr.Number(label="processes", value=2, minimum=1, maximum=32, step=1)
        with gr.Row():
            data_dir = gr.Textbox(label="data_dir *", scale=4)
            data_btn = gr.Button("Browse", scale=1)
        with gr.Row():
            out_dir = gr.Textbox(label="out_dir *", scale=4)
            out_btn = gr.Button("Browse", scale=1)
        with gr.Row(visible=False) as pipe_row:
            pipeline_path = gr.Textbox(label="pipeline_path * (predict mode)", scale=4)
            pipe_btn = gr.Button("Browse", scale=1)

    with gr.Group():
        gr.Markdown("### 2. HabitatSegmentation")
        clustering_mode = gr.Dropdown(
            label="clustering_mode *",
            choices=["two_step", "one_step", "direct_pooling"],
            value="two_step",
        )
        with gr.Column(visible=True) as box_supervoxel:
            gr.Markdown("**Supervoxel (two_step)**")
            with gr.Row():
                sv_algo = gr.Dropdown(label="supervoxel.algorithm", choices=["kmeans", "gmm", "slic"], value="kmeans")
                sv_n_clusters = gr.Number(label="supervoxel.n_clusters", value=50, minimum=10, maximum=500, step=1)
            with gr.Row():
                sv_max_iter = gr.Number(label="supervoxel.max_iter", value=300, precision=0)
                sv_n_init = gr.Number(label="supervoxel.n_init", value=10, precision=0)
            with gr.Row():
                sv_compactness = gr.Number(label="supervoxel.compactness (SLIC)", value=0.1)
                sv_sigma = gr.Number(label="supervoxel.sigma (SLIC)", value=0.0)
            sv_enforce_conn = gr.Checkbox(label="supervoxel.enforce_connectivity (SLIC)", value=True)

        with gr.Column(visible=False) as box_one_step:
            gr.Markdown("**one_step_settings (one_step mode)**")
            with gr.Row():
                os_min = gr.Number(label="one_step_settings.min_clusters", value=2, precision=0)
                os_max = gr.Number(label="one_step_settings.max_clusters", value=10, precision=0)
            os_fixed_n = gr.Number(label="one_step_settings.fixed_n_clusters (0=auto)", value=0, precision=0)
            os_selection = gr.Dropdown(label="one_step_settings.selection_method", choices=SELECTION_METHODS, value="elbow")
            os_plot_curves = gr.Checkbox(label="one_step_settings.plot_validation_curves", value=True)

        gr.Markdown("**Habitat clustering**")
        with gr.Row():
            hb_algo = gr.Dropdown(label="habitat.algorithm", choices=["kmeans", "gmm"], value="kmeans")
            hb_fixed_n_enabled = gr.Checkbox(label="Use fixed_n_clusters", value=False)
        with gr.Row():
            hb_fixed_n = gr.Number(label="habitat.fixed_n_clusters", value=3, minimum=2, maximum=20, step=1)
            hb_min_clusters = gr.Number(label="habitat.min_clusters", value=2, precision=0)
            hb_max_clusters = gr.Number(label="habitat.max_clusters", value=10, precision=0)
        hb_selection = gr.Dropdown(
            label="habitat_cluster_selection_method",
            choices=SELECTION_METHODS,
            value="elbow",
        )
        with gr.Row():
            hb_max_iter = gr.Number(label="habitat.max_iter", value=300, precision=0)
            hb_n_init = gr.Number(label="habitat.n_init", value=10, precision=0)
        with gr.Row():
            hb_parallel_search = gr.Checkbox(label="habitat.parallel_cluster_search", value=True)
            hb_search_workers = gr.Number(label="habitat.cluster_search_workers (0=null)", value=0, precision=0)

        with gr.Accordion("Connected-component postprocess", open=False):
            pp_sv_enabled = gr.Checkbox(label="postprocess_supervoxel.enabled", value=False)
            pp_sv_min_size = gr.Number(label="postprocess_supervoxel.min_component_size", value=30, precision=0)
            pp_hab_enabled = gr.Checkbox(label="postprocess_habitat.enabled", value=False)
            pp_hab_min_size = gr.Number(label="postprocess_habitat.min_component_size", value=30, precision=0)

    with gr.Group():
        gr.Markdown("### 3. FeatureConstruction")
        voxel_method = gr.Textbox(
            label="voxel_level.method *",
            value="concat(voxel_radiomics(T2, params_file, kernelRadius))",
        )
        with gr.Row():
            voxel_params_file = gr.Textbox(
                label="voxel_level.params.params_file",
                value="../radiomics/params_voxel_radiomics.yaml",
            )
            voxel_kernel = gr.Number(label="voxel_level.params.kernelRadius", value=3, precision=0)
        with gr.Row():
            voxel_batch = gr.Number(label="voxel_level.params.voxelBatch", value=1000, precision=0)
            use_torch = gr.Dropdown(
                label="voxel_level.params.useTorchRadiomics",
                choices=["auto", "true", "false"],
                value="auto",
            )

        with gr.Column(visible=True) as box_sv_level:
            gr.Markdown("**supervoxel_level (two_step)**")
            sv_file_kw = gr.Textbox(label="supervoxel_file_keyword", value="*_supervoxel.nrrd")
            sv_level_method = gr.Textbox(label="supervoxel_level.method", value="mean_voxel_features()")
            sv_level_params_file = gr.Textbox(
                label="supervoxel_level.params.params_file",
                value="../radiomics/params_supervoxel_radiomics.yaml",
            )
            sv_union_crop = gr.Checkbox(label="supervoxelUnionBboxCrop", value=True)
            use_sv_cext = gr.Dropdown(
                label="useSupervoxelCext",
                choices=["auto", "true", "false"],
                value="auto",
            )

        gr.Markdown("**preprocessing_for_subject_level** (YAML)")
        subject_prep_yaml = gr.Textbox(label="Subject-level feature preprocessing", lines=6, value=DEFAULT_SUBJECT_PREP)
        gr.Markdown("**preprocessing_for_group_level** (YAML)")
        group_prep_yaml = gr.Textbox(label="Group-level feature preprocessing", lines=8, value=DEFAULT_GROUP_PREP)

    with gr.Group():
        gr.Markdown("### 4. Run controls")
        with gr.Row():
            resume = gr.Checkbox(label="resume", value=True)
            plot_curves = gr.Checkbox(label="plot_curves", value=True)
            save_images = gr.Checkbox(label="save_images", value=True)
        with gr.Row():
            save_results_csv = gr.Checkbox(label="save_results_csv", value=True)
            habitats_format = gr.Dropdown(label="habitats_results_format", choices=["parquet", "csv"], value="parquet")
            random_state = gr.Number(label="random_state", value=42, precision=0)
        debug = gr.Checkbox(label="debug", value=False)

    with gr.Accordion("Advanced parallel / checkpoint / OOM controls", open=False):
        cap_gpu_pool = gr.Checkbox(label="cap_processes_to_gpu_pool", value=False)
        subject_timeout = gr.Number(label="individual_subject_timeout_sec (0=disable)", value=900, precision=0)
        subject_spawn_timeout = gr.Number(label="individual_subject_spawn_timeout_sec (0=disable)", value=120, precision=0)
        on_subject_failure = gr.Dropdown(label="on_subject_failure", choices=["continue", "fail_fast"], value="continue")
        oom_backoff = gr.Checkbox(label="oom_backoff", value=True)
        oom_reduce = gr.Number(label="oom_reduce_workers_by", value=1, minimum=1, precision=0)
        strict_checkpoint = gr.Checkbox(label="strict_checkpoint_hash", value=False)
        checkpoint_dir = gr.Textbox(label="checkpoint_dir (empty=null)", value="")
        force_rerun = gr.Textbox(label="force_rerun_subjects (comma-separated)", value="")
        retry_failed = gr.Checkbox(label="retry_failed_subjects", value=False)
        auto_retry_rounds = gr.Number(label="individual_subject_auto_retry_rounds", value=2, precision=0)
        parallel_mode = gr.Dropdown(
            label="individual_subject_parallel_mode",
            choices=["persistent", "isolated"],
            value="persistent",
        )
        clear_checkpoint = gr.Checkbox(label="clear_checkpoint_on_success", value=False)
        verbose = gr.Checkbox(label="verbose", value=True)

    submit_btn = gr.Button("Validate and run habitat analysis", variant="primary")
    log_output = gr.Textbox(label="Console log", lines=18, interactive=False)
    open_dir_btn = gr.Button("Open output folder", visible=False)

    def on_mode_change(mode: str) -> Dict[str, Any]:
        return gr.update(visible=mode == "predict")

    run_mode.change(on_mode_change, inputs=run_mode, outputs=pipe_row)
    clustering_mode.change(
        _clustering_mode_panels,
        inputs=clustering_mode,
        outputs=[box_supervoxel, box_sv_level, box_one_step],
    )

    def browse_file() -> Any:
        p = select_local_path("file", "Select file")
        return p if p else gr.update()

    def browse_folder() -> Any:
        p = select_local_path("folder", "Select folder")
        return p if p else gr.update()

    browse_cfg_btn.click(browse_file, outputs=existing_yaml)
    data_btn.click(browse_folder, outputs=data_dir)
    out_btn.click(browse_folder, outputs=out_dir)
    pipe_btn.click(browse_file, outputs=pipeline_path)

    # Load YAML — returns updates for major widgets (abbreviated mapping from loaded dict)
    def load_yaml(path: str) -> List[Any]:
        noop = gr.update()
        if not path or not os.path.exists(path):
            return [noop] * 65
        loaded = load_config_yaml(path)
        if not loaded:
            return [noop] * 65

        seg = loaded.get("HabitatSegmentation", {}) or {}
        sv = seg.get("supervoxel", {}) or {}
        hb = seg.get("habitat", {}) or {}
        os_settings = sv.get("one_step_settings", {}) or {}
        fc = loaded.get("FeatureConstruction", {}) or {}
        vl = fc.get("voxel_level", {}) or {}
        vp = vl.get("params", {}) or {}
        sl = fc.get("supervoxel_level", {}) or {}
        sp = sl.get("params", {}) or {}
        ppsv = seg.get("postprocess_supervoxel", {}) or {}
        pphb = seg.get("postprocess_habitat", {}) or {}

        raw_methods = hb.get("habitat_cluster_selection_method", ["elbow"])
        sel_method = raw_methods[0] if isinstance(raw_methods, list) else raw_methods

        subj_prep = dict_to_yaml_block(fc.get("preprocessing_for_subject_level"))
        group_prep = dict_to_yaml_block(fc.get("preprocessing_for_group_level"))

        force_list = loaded.get("force_rerun_subjects", []) or []
        timeout = loaded.get("individual_subject_timeout_sec", 900)
        spawn_to = loaded.get("individual_subject_spawn_timeout_sec", 120)

        return [
            loaded.get("run_mode", "train"),
            int(loaded.get("processes", 2)),
            loaded.get("data_dir", ""),
            loaded.get("out_dir", ""),
            loaded.get("pipeline_path", "") or "",
            seg.get("clustering_mode", "two_step"),
            sv.get("algorithm", "kmeans"),
            int(sv.get("n_clusters", 50)),
            int(sv.get("max_iter", 300)),
            int(sv.get("n_init", 10)),
            float(sv.get("compactness", 0.1)),
            float(sv.get("sigma", 0.0)),
            sv.get("enforce_connectivity", True),
            int(os_settings.get("min_clusters", 2)),
            int(os_settings.get("max_clusters", 10)),
            int(os_settings.get("fixed_n_clusters") or 0),
            os_settings.get("selection_method", "elbow"),
            os_settings.get("plot_validation_curves", True),
            hb.get("algorithm", "kmeans"),
            hb.get("fixed_n_clusters") is not None,
            int(hb.get("fixed_n_clusters") or 3),
            int(hb.get("min_clusters", 2) or 2),
            int(hb.get("max_clusters", 10) or 10),
            sel_method,
            int(hb.get("max_iter", 300)),
            int(hb.get("n_init", 10)),
            hb.get("parallel_cluster_search", True),
            int(hb.get("cluster_search_workers") or 0),
            ppsv.get("enabled", False),
            int(ppsv.get("min_component_size", 30)),
            pphb.get("enabled", False),
            int(pphb.get("min_component_size", 30)),
            vl.get("method", "concat(voxel_radiomics(T2, params_file, kernelRadius))"),
            vp.get("params_file", "../radiomics/params_voxel_radiomics.yaml"),
            int(vp.get("kernelRadius", 3)),
            int(vp.get("voxelBatch", 1000)),
            str(vp.get("useTorchRadiomics", "auto")),
            sl.get("supervoxel_file_keyword", "*_supervoxel.nrrd"),
            sl.get("method", "mean_voxel_features()"),
            sp.get("params_file", "../radiomics/params_supervoxel_radiomics.yaml"),
            sp.get("supervoxelUnionBboxCrop", True),
            str(sp.get("useSupervoxelCext", "auto")),
            subj_prep or DEFAULT_SUBJECT_PREP,
            group_prep or DEFAULT_GROUP_PREP,
            loaded.get("resume", True),
            loaded.get("plot_curves", True),
            loaded.get("save_images", True),
            loaded.get("save_results_csv", True),
            loaded.get("habitats_results_format", "parquet"),
            int(loaded.get("random_state", 42)),
            loaded.get("debug", False),
            loaded.get("cap_processes_to_gpu_pool", False),
            int(timeout) if timeout else 0,
            int(spawn_to) if spawn_to else 0,
            loaded.get("on_subject_failure", "continue"),
            loaded.get("oom_backoff", True),
            int(loaded.get("oom_reduce_workers_by", 1)),
            loaded.get("strict_checkpoint_hash", False),
            loaded.get("checkpoint_dir", "") or "",
            ", ".join(force_list),
            loaded.get("retry_failed_subjects", False),
            int(loaded.get("individual_subject_auto_retry_rounds", 2)),
            loaded.get("individual_subject_parallel_mode", "persistent"),
            loaded.get("clear_checkpoint_on_success", False),
            loaded.get("verbose", True),
        ]

    load_outputs = [
        run_mode, processes, data_dir, out_dir, pipeline_path, clustering_mode,
        sv_algo, sv_n_clusters, sv_max_iter, sv_n_init, sv_compactness, sv_sigma, sv_enforce_conn,
        os_min, os_max, os_fixed_n, os_selection, os_plot_curves,
        hb_algo, hb_fixed_n_enabled, hb_fixed_n, hb_min_clusters, hb_max_clusters, hb_selection,
        hb_max_iter, hb_n_init, hb_parallel_search, hb_search_workers,
        pp_sv_enabled, pp_sv_min_size, pp_hab_enabled, pp_hab_min_size,
        voxel_method, voxel_params_file, voxel_kernel, voxel_batch, use_torch,
        sv_file_kw, sv_level_method, sv_level_params_file, sv_union_crop, use_sv_cext,
        subject_prep_yaml, group_prep_yaml,
        resume, plot_curves, save_images, save_results_csv, habitats_format, random_state, debug,
        cap_gpu_pool, subject_timeout, subject_spawn_timeout, on_subject_failure,
        oom_backoff, oom_reduce, strict_checkpoint, checkpoint_dir, force_rerun,
        retry_failed, auto_retry_rounds, parallel_mode, clear_checkpoint, verbose,
    ]
    existing_yaml.change(load_yaml, inputs=existing_yaml, outputs=load_outputs)

    def run_habitat_pipeline(*args: Any):
        """Build HabitatAnalysisConfig, save YAML, and execute pipeline."""
        (
            run_mode_val, processes_val, data_dir_val, out_dir_val, pipeline_path_val,
            clustering_mode_val,
            sv_algo_val, sv_n_val, sv_max_iter_val, sv_n_init_val, sv_compact_val, sv_sigma_val, sv_enforce_val,
            os_min_val, os_max_val, os_fixed_n_val, os_sel_val, os_plot_val,
            hb_algo_val, hb_fixed_en_val, hb_fixed_n_val, hb_min_val, hb_max_val, hb_sel_val,
            hb_max_iter_val, hb_n_init_val, hb_par_val, hb_workers_val,
            pp_sv_en_val, pp_sv_min_val, pp_hab_en_val, pp_hab_min_val,
            voxel_method_val, voxel_pf_val, voxel_kr_val, voxel_batch_val, use_torch_val,
            sv_kw_val, sv_lm_val, sv_lpf_val, sv_crop_val, use_sv_cext_val,
            subj_yaml_val, group_yaml_val,
            resume_val, plot_curves_val, save_images_val, save_results_val, hab_fmt_val, rs_val, debug_val,
            cap_gpu_val, subj_to_val, spawn_to_val, on_fail_val,
            oom_back_val, oom_red_val, strict_ck_val, ck_dir_val, force_rerun_val,
            retry_fail_val, retry_rounds_val, par_mode_val, clear_ck_val, verbose_val,
        ) = args

        if not data_dir_val or not out_dir_val:
            yield "❌ data_dir and out_dir are required.", gr.update(visible=False)
            return
        if run_mode_val == "predict" and not pipeline_path_val:
            yield "❌ pipeline_path is required in predict mode.", gr.update(visible=False)
            return

        try:
            subj_prep = yaml_block_to_dict(subj_yaml_val) if subj_yaml_val.strip() else None
            group_prep = yaml_block_to_dict(group_yaml_val) if group_yaml_val.strip() else None
        except ValueError as exc:
            yield f"❌ {exc}", gr.update(visible=False)
            return

        seg_data: Dict[str, Any] = {
            "clustering_mode": clustering_mode_val,
            "supervoxel": {
                "algorithm": sv_algo_val,
                "n_clusters": int(sv_n_val),
                "max_iter": int(sv_max_iter_val),
                "n_init": int(sv_n_init_val),
                "compactness": float(sv_compact_val),
                "sigma": float(sv_sigma_val),
                "enforce_connectivity": sv_enforce_val,
                "one_step_settings": {
                    "min_clusters": int(os_min_val),
                    "max_clusters": int(os_max_val),
                    "fixed_n_clusters": int(os_fixed_n_val) if int(os_fixed_n_val) > 0 else None,
                    "selection_method": os_sel_val,
                    "plot_validation_curves": os_plot_val,
                },
            },
            "habitat": {
                "algorithm": hb_algo_val,
                "max_clusters": int(hb_max_val),
                "min_clusters": int(hb_min_val),
                "habitat_cluster_selection_method": [hb_sel_val],
                "fixed_n_clusters": int(hb_fixed_n_val) if hb_fixed_en_val else None,
                "max_iter": int(hb_max_iter_val),
                "n_init": int(hb_n_init_val),
                "parallel_cluster_search": hb_par_val,
                "cluster_search_workers": int(hb_workers_val) if int(hb_workers_val) > 0 else None,
            },
            "postprocess_supervoxel": {
                "enabled": pp_sv_en_val,
                "min_component_size": int(pp_sv_min_val),
            },
            "postprocess_habitat": {
                "enabled": pp_hab_en_val,
                "min_component_size": int(pp_hab_min_val),
            },
        }

        fc_data: Dict[str, Any] = {
            "voxel_level": {
                "method": voxel_method_val,
                "params": {
                    "params_file": voxel_pf_val,
                    "kernelRadius": int(voxel_kr_val),
                    "voxelBatch": int(voxel_batch_val),
                    "useTorchRadiomics": use_torch_val,
                },
            },
            "preprocessing_for_subject_level": subj_prep,
            "preprocessing_for_group_level": group_prep,
        }
        if clustering_mode_val == "two_step":
            fc_data["supervoxel_level"] = {
                "supervoxel_file_keyword": sv_kw_val,
                "method": sv_lm_val,
                "params": {
                    "params_file": sv_lpf_val,
                    "supervoxelUnionBboxCrop": sv_crop_val,
                    "useSupervoxelCext": use_sv_cext_val,
                },
            }

        config_data: Dict[str, Any] = {
            "data_dir": data_dir_val,
            "out_dir": out_dir_val,
            "run_mode": run_mode_val,
            "pipeline_path": pipeline_path_val if run_mode_val == "predict" else None,
            "processes": int(processes_val),
            "FeatureConstruction": fc_data if run_mode_val == "train" else fc_data,
            "HabitatSegmentation": seg_data,
            "resume": resume_val,
            "plot_curves": plot_curves_val,
            "save_images": save_images_val,
            "save_results_csv": save_results_val,
            "habitats_results_format": hab_fmt_val,
            "random_state": int(rs_val),
            "debug": debug_val,
            "cap_processes_to_gpu_pool": cap_gpu_val,
            "individual_subject_timeout_sec": float(subj_to_val) if float(subj_to_val) > 0 else None,
            "individual_subject_spawn_timeout_sec": float(spawn_to_val) if float(spawn_to_val) > 0 else None,
            "on_subject_failure": on_fail_val,
            "oom_backoff": oom_back_val,
            "oom_reduce_workers_by": int(oom_red_val),
            "strict_checkpoint_hash": strict_ck_val,
            "checkpoint_dir": ck_dir_val.strip() or None,
            "force_rerun_subjects": [s.strip() for s in force_rerun_val.split(",") if s.strip()],
            "retry_failed_subjects": retry_fail_val,
            "individual_subject_auto_retry_rounds": int(retry_rounds_val),
            "individual_subject_parallel_mode": par_mode_val,
            "clear_checkpoint_on_success": clear_ck_val,
            "verbose": verbose_val,
        }

        try:
            config = HabitatAnalysisConfig(**config_data)
            os.makedirs(out_dir_val, exist_ok=True)
            gui_path = str(Path(out_dir_val) / "config_habitat_gui.yaml")
            save_config_yaml(config_data, gui_path)
            yield f"💾 Config saved to {gui_path}\n🚀 Running habitat analysis...", gr.update(visible=False)

            def job() -> None:
                run_habitat(
                    config_file=gui_path,
                    debug_mode=config.debug,
                    mode=run_mode_val,
                    pipeline_path=pipeline_path_val if run_mode_val == "predict" else None,
                    resume=resume_val,
                )

            for log_text in run_background_job(job):
                yield log_text, gr.update(visible=True)
        except ValidationError as err:
            msgs = translate_pydantic_error(err)
            yield "⚠️ Validation errors:\n" + "\n".join(f"- {m}" for m in msgs), gr.update(visible=False)
        except Exception as exc:  # noqa: BLE001
            yield f"❌ Failed: {exc}", gr.update(visible=False)

    submit_btn.click(run_habitat_pipeline, inputs=load_outputs, outputs=[log_output, open_dir_btn])
    open_dir_btn.click(lambda p: open_directory(p), inputs=out_dir)
