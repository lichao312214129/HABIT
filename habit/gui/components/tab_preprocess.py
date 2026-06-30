# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Preprocessing tab component for Gradio GUI (schema-driven top-level + pipeline editor).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from pydantic import ValidationError

from habit.cli_commands.commands.cmd_preprocess import run_preprocess
from habit.core.preprocessing.config_schemas import PreprocessingConfig, SaveOptionsConfig
from habit.gui.components.preprocess_steps import (
    BUILTIN_FORM_STEPS,
    KNOWN_STEPS,
    PREPROCESS_QUICK_PRESETS,
    DEFAULT_HIST_PERCENTILES,
    STEP_LABELS,
    StepParamValues,
    apply_order_from_number,
    build_preprocessing_steps,
    dcm2_single_file_from_yaml,
    enabled_from_check_values,
    modalities_display_text,
    move_step_in_order,
    order_number_updates,
    panel_visible,
    parse_step_order,
    resolve_modalities_for_run,
    sync_order_with_enabled,
    toggle_backend_panels,
)
from habit.gui.path_picker import PathPickerRegistry
from habit.gui.pipeline_runner import run_background_job
from habit.gui.schema_form import OverrideSpec, SchemaForm
from habit.gui.tab_job_shell import (
    finalize_step_if_project,
    format_run_error,
    mark_step_if_project,
    wire_standard_job,
    wire_yaml_autoload,
)
from habit.gui.utils import (
    abs_path,
    coerce_str_list,
    dict_to_yaml_block,
    discover_modalities_from_data_dir,
    load_config_yaml,
    load_gui_draft,
    open_directory,
    render_console_log,
    save_config_yaml,
    save_gui_draft,
    user_visible_path,
    yaml_block_to_dict,
)
from habit.gui.step_integration import register_project_path_fill
from habit.gui.step_registry import register_step_paths
from habit.utils.docker_path_utils import display_path_value
def render_preprocess_tab(
    demo=None,
    path_picker: PathPickerRegistry | None = None,
    project_root_state: Any | None = None,
) -> None:
    """Render the image preprocessing tab inside a parent Gradio Blocks context."""
    picker = path_picker if path_picker is not None else PathPickerRegistry()
    gr.Markdown(
        "**Step 2 — Preprocessing:** Register, resample, and normalize images. "
        "Output is written to `02_preprocessed/processed_images/` under your project."
    )

    with gr.Row():
        quick_preset = gr.Dropdown(
            label="Quick preset (optional)",
            choices=list(PREPROCESS_QUICK_PRESETS.keys()),
            value=None,
        )
        apply_quick_btn = gr.Button("Apply quick preset", size="sm")

    with gr.Row():
        existing_yaml = gr.Textbox(label="Load existing preprocessing YAML (optional)", scale=4)
        browse_prep_config_btn = gr.Button("Browse config", scale=1)
    picker.add(browse_prep_config_btn, existing_yaml, pick="file")

    with gr.Group():
        gr.Markdown("### 1. Paths and system settings")
        main_form = SchemaForm.build(
            PreprocessingConfig,
            exclude={"Preprocessing", "save_options"},
            group_order=["Paths", "Advanced"],
            open_groups={"Paths", "Advanced"},
            path_picker=picker,
            overrides={
                "processes": OverrideSpec(label="Parallel workers (CPU)"),
                "auto_select_first_file": OverrideSpec(
                    label="Auto-select first file in modality folder",
                ),
            },
        )
        with gr.Row():
            detect_modalities_btn = gr.Button("Detect modalities", scale=1)
        modality_status = gr.Textbox(
            label="Detected modalities",
            value="Set input data root directory to detect modalities.",
            interactive=False,
        )

    # State holding the current step execution order (enabled steps only).
    step_order_state = gr.State(value=[])

    step_enabled: Dict[str, gr.Checkbox] = {}
    step_order_num: Dict[str, gr.Number] = {}
    step_up_btns: Dict[str, gr.Button] = {}
    step_down_btns: Dict[str, gr.Button] = {}

    with gr.Group():
        gr.Markdown("### 2. Select preprocessing steps and order")
        gr.Markdown(
            "Check steps to include. Edit **#** or click **⬆ / ⬇** on each row to set execution order."
        )
        with gr.Row():
            gr.Markdown("**#**", scale=0)
            gr.Markdown("**Step**", scale=4)
            gr.Markdown("**Move**", scale=0)

        for step_key in KNOWN_STEPS:
            with gr.Row():
                step_order_num[step_key] = gr.Number(
                    label="",
                    value=None,
                    minimum=1,
                    maximum=len(KNOWN_STEPS),
                    precision=0,
                    scale=0,
                )
                step_enabled[step_key] = gr.Checkbox(
                    label=STEP_LABELS[step_key],
                    value=False,
                    scale=4,
                )
                with gr.Row(scale=0):
                    step_up_btns[step_key] = gr.Button("⬆")
                    step_down_btns[step_key] = gr.Button("⬇")

    with gr.Group():
        gr.Markdown("### 3. Step parameters")
        gr.Markdown(
            "Modalities are auto-detected from the input directory "
            "(``images/<subject>/<modality>/``). Override per-step ``images`` in Extra YAML if needed."
        )
        with gr.Column(visible=False) as box_n4:
            gr.Markdown(f"**{STEP_LABELS['n4_correction']}**")
            n4_images = gr.Textbox(
                label="Modalities (auto-detected)",
                value="",
                interactive=False,
            )
            n4_levels = gr.Number(label="num_fitting_levels", value=4, minimum=1, maximum=6, step=1)

        with gr.Column(visible=False) as box_res:
            gr.Markdown(f"**{STEP_LABELS['resample']}**")
            res_images = gr.Textbox(
                label="Modalities (auto-detected)",
                value="",
                interactive=False,
            )
            res_spacing = gr.Textbox(label="target_spacing (x, y, z mm)", value="1.0, 1.0, 1.0")

        with gr.Column(visible=False) as box_zs:
            gr.Markdown(f"**{STEP_LABELS['zscore_normalization']}**")
            zs_images = gr.Textbox(
                label="Modalities (auto-detected)",
                value="",
                interactive=False,
            )
            zs_mask = gr.Checkbox(label="only_inmask", value=False)
            zs_mask_key = gr.Textbox(label="mask_key (optional)", value="")

        with gr.Column(visible=False) as box_reg:
            gr.Markdown(f"**{STEP_LABELS['registration']}**")
            reg_images = gr.Textbox(
                label="images (auto-detected; must include fixed_image)",
                value="",
                interactive=False,
            )
            reg_fixed = gr.Textbox(label="fixed_image *", value="")
            reg_backend = gr.Dropdown(
                label="backend",
                choices=["ants", "simpleitk", "elastix"],
                value="ants",
            )
            with gr.Column(visible=True) as box_ants_sitk:
                with gr.Row():
                    reg_transform = gr.Dropdown(
                        label="type_of_transform",
                        choices=["SyNRA", "SyN", "Rigid", "Affine", "Translation"],
                        value="SyNRA",
                    )
                    reg_metric = gr.Dropdown(
                        label="metric",
                        choices=["MI", "CC", "MeanSquares"],
                        value="MI",
                    )
                reg_optimizer = gr.Textbox(label="optimizer (optional)", value="")
            with gr.Column(visible=False) as box_elastix:
                elastix_parameter_files = gr.Textbox(
                    label="elastix_parameter_files (path or comma-separated list) *",
                    value="",
                )
                with gr.Row():
                    elastix_path = gr.Textbox(label="elastix_path (optional)", value="")
                    transformix_path = gr.Textbox(label="transformix_path (optional)", value="")
                elastix_threads = gr.Number(label="elastix_threads (optional)", value=0, precision=0)
                elastix_parameter_overrides = gr.Textbox(
                    label="elastix_parameter_overrides (YAML dict, optional)",
                    lines=4,
                    placeholder="MaximumNumberOfIterations: 200\nFinalGridSpacingInPhysicalUnits: 8",
                )
            with gr.Accordion("SimpleITK-only tuning (backend=simpleitk)", open=False, visible=False) as box_sitk:
                sitk_bins = gr.Number(label="number_of_histogram_bins", value=50, precision=0)
                sitk_sampling = gr.Number(label="metric_sampling_percentage", value=0.01)
                sitk_shrink = gr.Textbox(label="shrink_factors_per_level", value="4, 2, 1")
                sitk_sigmas = gr.Textbox(label="smoothing_sigmas_per_level", value="2.1, 1.0, 0.0")
                sitk_lr = gr.Number(label="learning_rate", value=1.0)
                sitk_iters = gr.Number(label="number_of_iterations", value=100, precision=0)
                sitk_bspline_mesh = gr.Number(label="bspline_mesh_size", value=8, precision=0)
                sitk_bspline_order = gr.Number(label="bspline_order", value=3, precision=0)
            reg_mask = gr.Checkbox(label="use_mask", value=False)
            reg_replace_mask = gr.Checkbox(label="replace_by_fixed_image_mask", value=True)
            reg_mask_key = gr.Textbox(label="mask_key (optional, when use_mask=true)", value="")

        with gr.Column(visible=False) as box_hist:
            gr.Markdown(f"**{STEP_LABELS['histogram_standardization']}**")
            hist_images = gr.Textbox(
                label="Modalities (auto-detected)",
                value="",
                interactive=False,
            )
            hist_percentiles = gr.Textbox(
                label="percentiles (comma-separated)",
                value=DEFAULT_HIST_PERCENTILES,
            )
            with gr.Row():
                hist_target_min = gr.Number(label="target_min", value=0.0)
                hist_target_max = gr.Number(label="target_max", value=100.0)
            hist_mask_key = gr.Textbox(label="mask_key (optional)", value="")

        with gr.Column(visible=False) as box_ahe:
            gr.Markdown(f"**{STEP_LABELS['adaptive_histogram_equalization']}**")
            ahe_images = gr.Textbox(
                label="Modalities (auto-detected)",
                value="",
                interactive=False,
            )
            with gr.Row():
                ahe_alpha = gr.Number(label="alpha [0,1]", value=0.3)
                ahe_beta = gr.Number(label="beta [0,1]", value=0.3)
            ahe_radius = gr.Number(label="radius (pixels)", value=5, precision=0)

        with gr.Column(visible=False) as box_reorient:
            gr.Markdown(f"**{STEP_LABELS['reorientation']}**")
            reorient_images = gr.Textbox(
                label="Modalities (auto-detected)",
                value="",
                interactive=False,
            )
            with gr.Row():
                reorient_target = gr.Dropdown(
                    label="target_orientation",
                    choices=["LPS", "RAS", "LIA", "RIA", "PIR", "ASL"],
                    value="LPS",
                )
                reorient_mode = gr.Dropdown(
                    label="mode",
                    choices=["closest", "strict"],
                    value="closest",
                )

        with gr.Column(visible=False) as box_dcm2nii:
            gr.Markdown(f"**{STEP_LABELS['dcm2nii']}**")
            dcm2_images = gr.Textbox(
                label="Modalities (auto-detected; flat DICOM layout uses \"dicom\")",
                value="dicom",
                interactive=False,
            )
            dcm2_path = gr.Textbox(label="dcm2niix_path (optional)", value="")
            dcm2_filename_format = gr.Textbox(label="filename_format (optional)", value="")
            with gr.Row():
                dcm2_compress = gr.Checkbox(label="compress", value=True)
                dcm2_anonymize = gr.Checkbox(label="anonymize", value=False)
                dcm2_ignore_derived = gr.Checkbox(label="ignore_derived", value=False)
            with gr.Row():
                dcm2_crop = gr.Checkbox(label="crop_images", value=False)
                dcm2_generate_json = gr.Checkbox(label="generate_json (BIDS sidecar)", value=False)
                dcm2_verbose = gr.Checkbox(label="verbose", value=False)
            with gr.Row():
                dcm2_batch_mode = gr.Checkbox(label="batch_mode", value=True)
                dcm2_adjacent = gr.Checkbox(label="adjacent_dicoms", value=True)
            with gr.Row():
                dcm2_merge_slices = gr.Dropdown(
                    label="merge_slices",
                    choices=["2", "y", "1", "n", "0"],
                    value="2",
                )
                dcm2_single_file = gr.Dropdown(
                    label="single_file_mode",
                    choices=["auto", "true", "false"],
                    value="auto",
                )

    with gr.Accordion("Extra preprocessing YAML (optional overrides)", open=False):
        extra_prep_yaml = gr.Textbox(
            label="Additional Preprocessing blocks (YAML mapping)",
            lines=8,
            placeholder=(
                "histogram_standardization:\n"
                "  images: [T1, T2]\n"
                "  percentiles: [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]\n"
            ),
        )

    with gr.Group():
        gr.Markdown("### 4. Intermediate outputs")
        save_form = SchemaForm.build(
            SaveOptionsConfig,
            group_order=["Save"],
            open_groups={"Save"},
            overrides={
                "intermediate_steps": OverrideSpec(choices=KNOWN_STEPS),
            },
        )

    # Aliases for pipeline wiring (schema-driven top-level fields).
    data_dir = main_form.widget("data_dir")
    out_dir = main_form.widget("out_dir")
    processes = main_form.widget("processes")
    auto_select = main_form.widget("auto_select_first_file")
    random_state = main_form.widget("random_state")
    save_intermediate = save_form.widget("save_intermediate")
    intermediate_steps = save_form.widget("intermediate_steps")

    with gr.Row():
        submit_btn = gr.Button("Validate and run preprocessing", variant="primary")
        stop_btn = gr.Button("Stop", interactive=False)
    log_output = render_console_log(lines=18, elem_id="habit-log-preprocess")
    open_dir_btn = gr.Button("Open output folder", visible=False)

    modality_fields = [
        n4_images,
        res_images,
        zs_images,
        reg_images,
        hist_images,
        ahe_images,
        reorient_images,
        dcm2_images,
    ]
    step_param_panels = [
        box_n4,
        box_res,
        box_zs,
        box_reg,
        box_hist,
        box_ahe,
        box_reorient,
        box_dcm2nii,
    ]
    step_panel_keys = [
        "n4_correction",
        "resample",
        "zscore_normalization",
        "registration",
        "histogram_standardization",
        "adaptive_histogram_equalization",
        "reorientation",
        "dcm2nii",
    ]

    step_check_inputs: List[Any] = [data_dir, auto_select, step_order_state] + [
        step_enabled[step_key] for step_key in KNOWN_STEPS
    ]
    step_order_num_outputs: List[Any] = [step_order_num[step_key] for step_key in KNOWN_STEPS]
    step_check_outputs: List[Any] = (
        [step_order_state, *step_order_num_outputs, *step_param_panels]
    )

    def _apply_detected_modalities(
        data_dir_val: str,
        auto_select_val: bool,
    ) -> Tuple[Any, ...]:
        """
        Scan data_dir and refresh all read-only modality text fields.

        Args:
            data_dir_val: Input data root directory from the GUI.
            auto_select_val: Whether to auto-pick the first file in modality folders.

        Returns:
            Tuple of gr updates: status, modality textboxes, reg_fixed default.
        """
        modalities, status = discover_modalities_from_data_dir(
            data_dir_val,
            bool(auto_select_val),
        )
        if modalities:
            text = modalities_display_text(modalities)
            dcm2_text = text
            fixed_image = modalities[0]
        else:
            text = ""
            dcm2_text = "dicom"
            fixed_image = ""
        return (
            status,
            gr.update(value=text),
            gr.update(value=text),
            gr.update(value=text),
            gr.update(value=text),
            gr.update(value=text),
            gr.update(value=text),
            gr.update(value=text),
            gr.update(value=dcm2_text),
            gr.update(value=fixed_image),
        )

    def _apply_quick_preset(preset_name: Optional[str]) -> Tuple[Any, ...]:
        """Enable steps and order from a named quick preset."""
        steps: List[str] = PREPROCESS_QUICK_PRESETS.get(preset_name or "", [])
        if not steps:
            noop = gr.update()
            n = 1 + len(KNOWN_STEPS) + len(KNOWN_STEPS) + len(step_panel_keys)
            return tuple(noop for _ in range(n))
        enabled_set = set(steps)
        check_updates = tuple(
            gr.update(value=(step_key in enabled_set)) for step_key in KNOWN_STEPS
        )
        panel_updates = tuple(
            panel_visible(key, enabled_set) for key in step_panel_keys
        )
        return (
            steps,
            *order_number_updates(steps),
            *check_updates,
            *panel_updates,
        )

    apply_quick_btn.click(
        _apply_quick_preset,
        inputs=[quick_preset],
        outputs=[
            step_order_state,
            *step_order_num_outputs,
            *[step_enabled[step_key] for step_key in KNOWN_STEPS],
            *step_param_panels,
        ],
    )

    def _on_step_toggle(
        data_dir_val: str,
        auto_select_val: bool,
        current_order: List[str],
        *check_values: bool,
    ) -> Tuple[Any, ...]:
        """Sync order numbers and parameter panels when any step checkbox toggles."""
        enabled = enabled_from_check_values(list(check_values))
        new_order = sync_order_with_enabled(enabled, current_order)
        panel_updates = tuple(panel_visible(key, enabled) for key in step_panel_keys)
        return (
            new_order,
            *order_number_updates(new_order),
            *panel_updates,
        )

    def _on_step_move(
        step_key: str,
        direction: int,
        current_order: List[str],
        *check_values: bool,
    ) -> Tuple[Any, ...]:
        """Move one enabled step up/down and refresh order numbers."""
        enabled = enabled_from_check_values(list(check_values))
        order = [step for step in current_order if step in enabled]
        if step_key not in enabled:
            return (order, *order_number_updates(order))
        new_order = move_step_in_order(order, step_key, direction)
        return (new_order, *order_number_updates(new_order))

    def _on_step_order_edit(
        step_key: str,
        new_num: Optional[float],
        current_order: List[str],
        *check_values: bool,
    ) -> Tuple[Any, ...]:
        """Apply a manually edited sequence number for one step."""
        enabled = enabled_from_check_values(list(check_values))
        new_order = apply_order_from_number(step_key, new_num, current_order, enabled)
        return (new_order, *order_number_updates(new_order))

    step_move_inputs: List[Any] = [step_order_state] + [
        step_enabled[step_key] for step_key in KNOWN_STEPS
    ]
    step_move_outputs: List[Any] = [step_order_state, *step_order_num_outputs]
    detect_modalities_inputs: List[Any] = [data_dir, auto_select]
    detect_modalities_outputs: List[Any] = [modality_status, *modality_fields, reg_fixed]

    for step_key in KNOWN_STEPS:
        def _make_move_handler(step: str, direction: int):
            def _handler(order: List[str], *check_values: bool) -> Tuple[Any, ...]:
                return _on_step_move(step, direction, order, *check_values)

            return _handler

        def _make_order_handler(step: str):
            def _handler(
                new_num: Optional[float],
                order: List[str],
                *check_values: bool,
            ) -> Tuple[Any, ...]:
                return _on_step_order_edit(step, new_num, order, *check_values)

            return _handler

        step_enabled[step_key].change(
            _on_step_toggle,
            inputs=step_check_inputs,
            outputs=step_check_outputs,
        ).then(
            _apply_detected_modalities,
            inputs=detect_modalities_inputs,
            outputs=detect_modalities_outputs,
        )
        step_up_btns[step_key].click(
            _make_move_handler(step_key, -1),
            inputs=step_move_inputs,
            outputs=step_move_outputs,
        )
        step_down_btns[step_key].click(
            _make_move_handler(step_key, 1),
            inputs=step_move_inputs,
            outputs=step_move_outputs,
        )
        step_order_num[step_key].change(
            _make_order_handler(step_key),
            inputs=[step_order_num[step_key], step_order_state] + step_move_inputs[1:],
            outputs=step_move_outputs,
        )

    data_dir.change(
        _apply_detected_modalities,
        inputs=detect_modalities_inputs,
        outputs=detect_modalities_outputs,
    )
    auto_select.change(
        _apply_detected_modalities,
        inputs=detect_modalities_inputs,
        outputs=detect_modalities_outputs,
    )
    detect_modalities_btn.click(
        _apply_detected_modalities,
        inputs=detect_modalities_inputs,
        outputs=detect_modalities_outputs,
    )

    reg_backend.change(
        toggle_backend_panels,
        inputs=reg_backend,
        outputs=[box_ants_sitk, box_elastix, box_sitk],
    )

    load_outputs = [
        *main_form.outputs(),
        modality_status,
        *[step_enabled[step_key] for step_key in KNOWN_STEPS],
        step_order_state,
        *[step_order_num[step_key] for step_key in KNOWN_STEPS],
        box_n4, box_res, box_zs, box_reg, box_hist, box_ahe, box_reorient, box_dcm2nii,
        n4_images, n4_levels,
        res_images, res_spacing,
        zs_images, zs_mask, zs_mask_key,
        reg_images, reg_fixed, reg_backend, reg_transform, reg_metric, reg_optimizer,
        elastix_parameter_files, elastix_path, transformix_path, elastix_threads,
        elastix_parameter_overrides,
        sitk_bins, sitk_sampling, sitk_shrink, sitk_sigmas, sitk_lr, sitk_iters,
        sitk_bspline_mesh, sitk_bspline_order,
        reg_mask, reg_replace_mask, reg_mask_key,
        hist_images, hist_percentiles, hist_target_min, hist_target_max, hist_mask_key,
        ahe_images, ahe_alpha, ahe_beta, ahe_radius,
        reorient_images, reorient_target, reorient_mode,
        dcm2_images, dcm2_path, dcm2_filename_format,
        dcm2_compress, dcm2_anonymize, dcm2_ignore_derived, dcm2_crop,
        dcm2_generate_json, dcm2_verbose, dcm2_batch_mode, dcm2_adjacent,
        dcm2_merge_slices, dcm2_single_file,
        extra_prep_yaml,
        *save_form.outputs(),
    ]

    def load_config(yaml_path: str) -> List[Any]:
        """Hydrate form widgets from an existing preprocessing YAML file."""
        noop = gr.update()
        resolved = abs_path(yaml_path) if yaml_path and str(yaml_path).strip() else ""
        if not resolved or not os.path.exists(resolved):
            return [noop] * len(load_outputs)

        loaded: Optional[Dict[str, Any]] = load_config_yaml(resolved)
        if not loaded:
            return [noop] * len(load_outputs)

        prep: Dict[str, Any] = loaded.get("Preprocessing", {}) or {}
        enabled_keys: List[str] = [k for k in prep.keys() if k in KNOWN_STEPS]
        order_keys: List[str] = list(enabled_keys)

        n4 = prep.get("n4_correction", {})
        res = prep.get("resample", {})
        zs = prep.get("zscore_normalization", {})
        reg = prep.get("registration", {})
        hist = prep.get("histogram_standardization", {})
        ahe = prep.get("adaptive_histogram_equalization", {})
        reorient = prep.get("reorientation", {})
        dcm2 = prep.get("dcm2nii", {})
        save_opt: Dict[str, Any] = loaded.get("save_options", {}) or {}

        extra_blocks: Dict[str, Any] = {
            k: v for k, v in prep.items() if k not in BUILTIN_FORM_STEPS
        }

        sitk_shrink_val = reg.get("shrink_factors_per_level", [4, 2, 1])
        sitk_sigmas_val = reg.get("smoothing_sigmas_per_level", [2.1, 1.0, 0.0])
        elastix_files = reg.get("elastix_parameter_files", "")
        if isinstance(elastix_files, list):
            elastix_files_str = ", ".join(str(x) for x in elastix_files)
        else:
            elastix_files_str = str(elastix_files or "")

        hist_percentiles_val = hist.get(
            "percentiles",
            [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99],
        )

        loaded_data_dir = loaded.get("data_dir", "")
        auto_select_loaded = loaded.get("auto_select_first_file", True)
        detected_modalities, detect_status = discover_modalities_from_data_dir(
            abs_path(loaded_data_dir) if loaded_data_dir else loaded_data_dir,
            bool(auto_select_loaded),
        )
        if detected_modalities:
            modality_text = modalities_display_text(detected_modalities)
            dcm2_modality_text = modality_text
            default_fixed = reg.get("fixed_image") or detected_modalities[0]
        else:
            modality_text = ", ".join(coerce_str_list(n4.get("images", [])))
            dcm2_modality_text = ", ".join(coerce_str_list(dcm2.get("images", ["dicom"]))) or "dicom"
            default_fixed = reg.get("fixed_image", "")

        n4_images_text = ", ".join(coerce_str_list(n4.get("images", []))) or modality_text
        res_images_text = ", ".join(coerce_str_list(res.get("images", []))) or modality_text
        zs_images_text = ", ".join(coerce_str_list(zs.get("images", []))) or modality_text
        reg_images_text = ", ".join(coerce_str_list(reg.get("images", []))) or modality_text
        hist_images_text = ", ".join(coerce_str_list(hist.get("images", []))) or modality_text
        ahe_images_text = ", ".join(coerce_str_list(ahe.get("images", []))) or modality_text
        reorient_images_text = ", ".join(coerce_str_list(reorient.get("images", []))) or modality_text

        return [
            *main_form.populate({
                "data_dir": display_path_value("data_dir", loaded.get("data_dir", "")),
                "out_dir": display_path_value("out_dir", loaded.get("out_dir", "")),
                "processes": int(loaded.get("processes", 1)),
                "random_state": int(loaded.get("random_state", 42)),
                "auto_select_first_file": loaded.get("auto_select_first_file", True),
                "preprocessing_input_layout": loaded.get(
                    "preprocessing_input_layout", "habit_default",
                ),
            }),
            detect_status,
            *[step_key in enabled_keys for step_key in KNOWN_STEPS],
            order_keys,
            *order_number_updates(order_keys),
            panel_visible("n4_correction", enabled_keys),
            panel_visible("resample", enabled_keys),
            panel_visible("zscore_normalization", enabled_keys),
            panel_visible("registration", enabled_keys),
            panel_visible("histogram_standardization", enabled_keys),
            panel_visible("adaptive_histogram_equalization", enabled_keys),
            panel_visible("reorientation", enabled_keys),
            panel_visible("dcm2nii", enabled_keys),
            n4_images_text,
            int(n4.get("num_fitting_levels", 4)),
            res_images_text,
            ", ".join(str(s) for s in res.get("target_spacing", [1.0, 1.0, 1.0])),
            zs_images_text,
            zs.get("only_inmask", False),
            zs.get("mask_key", "") or "",
            reg_images_text,
            default_fixed,
            reg.get("backend", "ants"),
            reg.get("type_of_transform", "SyNRA"),
            reg.get("metric", "MI"),
            reg.get("optimizer", "") or "",
            elastix_files_str,
            display_path_value("elastix_path", reg.get("elastix_path", "") or ""),
            display_path_value("transformix_path", reg.get("transformix_path", "") or ""),
            int(reg.get("elastix_threads", 0) or 0),
            dict_to_yaml_block(reg.get("elastix_parameter_overrides", {}) or {}),
            int(reg.get("number_of_histogram_bins", 50)),
            float(reg.get("metric_sampling_percentage", 0.01)),
            ", ".join(str(x) for x in sitk_shrink_val),
            ", ".join(str(x) for x in sitk_sigmas_val),
            float(reg.get("learning_rate", 1.0)),
            int(reg.get("number_of_iterations", 100)),
            int(reg.get("bspline_mesh_size", 8)),
            int(reg.get("bspline_order", 3)),
            reg.get("use_mask", False),
            reg.get("replace_by_fixed_image_mask", True),
            reg.get("mask_key", "") or "",
            hist_images_text,
            ", ".join(str(x) for x in hist_percentiles_val),
            float(hist.get("target_min", 0.0)),
            float(hist.get("target_max", 100.0)),
            hist.get("mask_key", "") or "",
            ahe_images_text,
            float(ahe.get("alpha", 0.3)),
            float(ahe.get("beta", 0.3)),
            int(ahe.get("radius", 5)),
            reorient_images_text,
            str(reorient.get("target_orientation", "LPS")).upper(),
            reorient.get("mode", "closest"),
            dcm2_modality_text,
            display_path_value("dcm2niix_path", dcm2.get("dcm2niix_path", "") or ""),
            dcm2.get("filename_format", "") or "",
            dcm2.get("compress", True),
            dcm2.get("anonymize", False),
            dcm2.get("ignore_derived", False),
            dcm2.get("crop_images", False),
            dcm2.get("generate_json", False),
            dcm2.get("verbose", False),
            dcm2.get("batch_mode", True),
            dcm2.get("adjacent_dicoms", True),
            str(dcm2.get("merge_slices", "2")),
            dcm2_single_file_from_yaml(dcm2.get("single_file_mode")),
            dict_to_yaml_block(extra_blocks),
            *save_form.populate(save_opt),
        ]

    wire_yaml_autoload(existing_yaml, load_config, load_outputs)

    def run_pipeline(*args: Any):
        """Validate config, save YAML, and run preprocessing with live logs."""
        (
            project_root,
            data_dir_val, out_dir_val, processes_val, auto_select_val, random_state_val,
            step_order_val, *step_check_vals,
            n4_images_val, n4_levels_val,
            res_images_val, res_spacing_val,
            zs_images_val, zs_mask_val, zs_mask_key_val,
            reg_images_val, reg_fixed_val, reg_backend_val,
            reg_transform_val, reg_metric_val, reg_optimizer_val,
            elastix_files_val, elastix_path_val, transformix_path_val, elastix_threads_val,
            elastix_overrides_val,
            sitk_bins_val, sitk_sampling_val, sitk_shrink_val, sitk_sigmas_val,
            sitk_lr_val, sitk_iters_val, sitk_bspline_mesh_val, sitk_bspline_order_val,
            reg_mask_val, reg_replace_mask_val, reg_mask_key_val,
            hist_images_val, hist_percentiles_val, hist_target_min_val, hist_target_max_val,
            hist_mask_key_val,
            ahe_images_val, ahe_alpha_val, ahe_beta_val, ahe_radius_val,
            reorient_images_val, reorient_target_val, reorient_mode_val,
            dcm2_images_val, dcm2_path_val, dcm2_format_val,
            dcm2_compress_val, dcm2_anonymize_val, dcm2_ignore_derived_val, dcm2_crop_val,
            dcm2_json_val, dcm2_verbose_val, dcm2_batch_val, dcm2_adjacent_val,
            dcm2_merge_val, dcm2_single_file_val,
            extra_prep_yaml_val, save_intermediate_val, intermediate_steps_val,
        ) = args

        if not data_dir_val or not out_dir_val:
            yield "❌ data_dir and out_dir are required.", gr.update(visible=False)
            return

        mark_step_if_project(str(project_root) if project_root else "", "preprocess")

        enabled_set = set(enabled_from_check_values(list(step_check_vals)))
        if not enabled_set:
            yield "❌ Select at least one preprocessing step.", gr.update(visible=False)
            return

        # Resolve relative paths to absolute before building the config.
        data_dir_abs = abs_path(data_dir_val)
        out_dir_abs = abs_path(out_dir_val)

        try:
            step_order_all: List[str] = parse_step_order(step_order_val)
        except ValueError as exc:
            yield f"❌ {exc}", gr.update(visible=False)
            return

        step_order = [step for step in step_order_all if step in enabled_set]
        if not step_order:
            yield "❌ Step execution order is empty for the selected steps.", gr.update(visible=False)
            return

        modalities, modality_err = resolve_modalities_for_run(
            data_dir_abs,
            bool(auto_select_val),
            str(n4_images_val),
        )
        dcm2_modalities, dcm2_err = resolve_modalities_for_run(
            data_dir_abs,
            bool(auto_select_val),
            str(dcm2_images_val),
        )
        if not dcm2_modalities:
            dcm2_modalities = ["dicom"]

        step_params = StepParamValues(
            n4_levels=n4_levels_val,
            res_spacing=res_spacing_val,
            zs_mask=zs_mask_val,
            zs_mask_key=zs_mask_key_val,
            reg_fixed=reg_fixed_val,
            reg_backend=reg_backend_val,
            reg_transform=reg_transform_val,
            reg_metric=reg_metric_val,
            reg_optimizer=reg_optimizer_val,
            elastix_files=elastix_files_val,
            elastix_path=elastix_path_val,
            transformix_path=transformix_path_val,
            elastix_threads=elastix_threads_val,
            elastix_overrides=elastix_overrides_val,
            sitk_bins=sitk_bins_val,
            sitk_sampling=sitk_sampling_val,
            sitk_shrink=sitk_shrink_val,
            sitk_sigmas=sitk_sigmas_val,
            sitk_lr=sitk_lr_val,
            sitk_iters=sitk_iters_val,
            sitk_bspline_mesh=sitk_bspline_mesh_val,
            sitk_bspline_order=sitk_bspline_order_val,
            reg_mask=reg_mask_val,
            reg_replace_mask=reg_replace_mask_val,
            reg_mask_key=reg_mask_key_val,
            hist_percentiles=hist_percentiles_val,
            hist_target_min=hist_target_min_val,
            hist_target_max=hist_target_max_val,
            hist_mask_key=hist_mask_key_val,
            ahe_alpha=ahe_alpha_val,
            ahe_beta=ahe_beta_val,
            ahe_radius=ahe_radius_val,
            reorient_target=reorient_target_val,
            reorient_mode=reorient_mode_val,
            dcm2_path=dcm2_path_val,
            dcm2_format=dcm2_format_val,
            dcm2_compress=dcm2_compress_val,
            dcm2_anonymize=dcm2_anonymize_val,
            dcm2_ignore_derived=dcm2_ignore_derived_val,
            dcm2_crop=dcm2_crop_val,
            dcm2_json=dcm2_json_val,
            dcm2_verbose=dcm2_verbose_val,
            dcm2_batch=dcm2_batch_val,
            dcm2_adjacent=dcm2_adjacent_val,
            dcm2_merge=dcm2_merge_val,
            dcm2_single_file=dcm2_single_file_val,
            extra_prep_yaml=extra_prep_yaml_val,
        )
        err_msg, prep_steps_config = build_preprocessing_steps(
            enabled_set,
            step_order,
            modalities,
            modality_err,
            dcm2_modalities,
            dcm2_err,
            step_params,
        )
        if err_msg:
            yield err_msg, gr.update(visible=False)
            return

        config_data: Dict[str, Any] = {
            "data_dir": data_dir_abs,
            "out_dir": out_dir_abs,
            "Preprocessing": prep_steps_config,
            "save_options": {
                "save_intermediate": save_intermediate_val,
                "intermediate_steps": list(intermediate_steps_val or []),
            },
            "processes": int(processes_val),
            "auto_select_first_file": auto_select_val,
            "random_state": int(random_state_val),
            "preprocessing_input_layout": "habit_default",
        }

        try:
            PreprocessingConfig(**config_data)
            os.makedirs(out_dir_abs, exist_ok=True)
            gui_config_path = str(Path(out_dir_abs) / "config_preprocess_gui.yaml")
            save_config_yaml(config_data, gui_config_path)
            save_gui_draft("preprocess", gui_config_path)
            yield (
                f"💾 Config validated and saved to {gui_config_path}\n🚀 Running preprocessing...",
                gr.update(visible=False),
            )
            last_log = ""
            for log_text in run_background_job(
                run_preprocess,
                args=(gui_config_path,),
                log_file=Path(out_dir_abs) / "processing.log",
            ):
                last_log = log_text
                yield log_text, gr.update(visible=True)
            processed_out = str(Path(out_dir_abs) / "processed_images")
            finalize_step_if_project(
                str(project_root) if project_root else "",
                "preprocess",
                last_log,
                config_path=gui_config_path,
                output_dir=processed_out,
            )
        except ValidationError as val_err:
            yield format_run_error(val_err), gr.update(visible=False)
        except Exception as exc:  # noqa: BLE001
            yield format_run_error(exc), gr.update(visible=False)

    wire_standard_job(
        submit_btn,
        stop_btn,
        run_pipeline,
        inputs=[
            project_root_state,
            data_dir, out_dir, processes, auto_select, random_state, step_order_state,
            *[step_enabled[step_key] for step_key in KNOWN_STEPS],
            n4_images, n4_levels,
            res_images, res_spacing,
            zs_images, zs_mask, zs_mask_key,
            reg_images, reg_fixed, reg_backend, reg_transform, reg_metric, reg_optimizer,
            elastix_parameter_files, elastix_path, transformix_path, elastix_threads,
            elastix_parameter_overrides,
            sitk_bins, sitk_sampling, sitk_shrink, sitk_sigmas, sitk_lr, sitk_iters,
            sitk_bspline_mesh, sitk_bspline_order,
            reg_mask, reg_replace_mask, reg_mask_key,
            hist_images, hist_percentiles, hist_target_min, hist_target_max, hist_mask_key,
            ahe_images, ahe_alpha, ahe_beta, ahe_radius,
            reorient_images, reorient_target, reorient_mode,
            dcm2_images, dcm2_path, dcm2_filename_format,
            dcm2_compress, dcm2_anonymize, dcm2_ignore_derived, dcm2_crop,
            dcm2_generate_json, dcm2_verbose, dcm2_batch_mode, dcm2_adjacent,
            dcm2_merge_slices, dcm2_single_file,
            extra_prep_yaml, save_intermediate, intermediate_steps,
        ],
        outputs=[log_output, open_dir_btn],
    )
    open_dir_btn.click(lambda p: open_directory(abs_path(p)), inputs=out_dir)

    if project_root_state is not None:
        register_step_paths(
            "preprocess",
            ["data_dir", "out_dir"],
            [data_dir, out_dir],
        )
        register_project_path_fill(
            project_root_state,
            "preprocess",
            ["data_dir", "out_dir"],
            [data_dir, out_dir],
        )

    if path_picker is None:
        picker.finalize()

    if demo is not None:
        # Restore only the YAML path; existing_yaml.change() then reloads all fields.
        def _restore_preprocess_path() -> str:
            draft = load_gui_draft("preprocess") or ""
            return user_visible_path(draft) if draft else ""
        demo.load(_restore_preprocess_path, inputs=[], outputs=[existing_yaml])
