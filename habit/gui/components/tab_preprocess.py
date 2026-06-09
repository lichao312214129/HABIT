# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Preprocessing tab component for Gradio GUI.
Provides clinician-friendly inputs for image preprocessing pipelines.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
from pydantic import ValidationError

from habit.cli_commands.commands.cmd_preprocess import run_preprocess
from habit.core.preprocessing.config_schemas import PreprocessingConfig
from habit.gui.pipeline_runner import run_background_job
from habit.gui.utils import (
    abs_path,
    coerce_str_list,
    dict_to_yaml_block,
    discover_modalities_from_data_dir,
    extract_validation_msgs,
    load_config_yaml,
    open_directory,
    parse_comma_list,
    save_config_yaml,
    select_local_path,
    translate_pydantic_error,
    yaml_block_to_dict,
)

KNOWN_STEPS: List[str] = [
    "n4_correction",
    "resample",
    "zscore_normalization",
    "registration",
    "histogram_standardization",
    "adaptive_histogram_equalization",
    "reorientation",
    "dcm2nii",
]
STEP_LABELS: Dict[str, str] = {
    "n4_correction": "N4 bias correction",
    "resample": "Resample",
    "zscore_normalization": "Z-score normalization",
    "registration": "Registration",
    "histogram_standardization": "Histogram standardization",
    "adaptive_histogram_equalization": "Adaptive histogram equalization",
    "reorientation": "Reorientation",
    "dcm2nii": "DICOM to NIfTI (dcm2nii)",
}
DEFAULT_HIST_PERCENTILES: str = "1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99"
BUILTIN_FORM_STEPS: frozenset[str] = frozenset({
    "n4_correction",
    "resample",
    "zscore_normalization",
    "registration",
    "histogram_standardization",
    "adaptive_histogram_equalization",
    "reorientation",
    "dcm2nii",
})
LOAD_OUTPUT_COUNT: int = 88


def _dcm2_single_file_from_yaml(value: Any) -> str:
    """Map YAML single_file_mode to GUI dropdown value."""
    if value is None:
        return "auto"
    return "true" if bool(value) else "false"


def _dcm2_single_file_to_yaml(value: str) -> Optional[bool]:
    """Map GUI dropdown value to YAML single_file_mode."""
    if value == "auto":
        return None
    return value == "true"


def _parse_step_order(order: Union[str, List[str]]) -> List[str]:
    """
    Validate the step execution order.

    Accepts either a list of step keys (from the new gr.State widget) or a
    legacy comma-separated string (for backward compatibility with YAML loading).

    Args:
        order: List of step keys, or comma-separated string.

    Returns:
        List[str]: Validated, ordered list of step keys.
    """
    if isinstance(order, list):
        steps: List[str] = [str(s).strip() for s in order if s and str(s).strip()]
    else:
        steps = parse_comma_list(str(order))
    if not steps:
        raise ValueError("Step execution order cannot be empty.")
    unknown: List[str] = [s for s in steps if s not in KNOWN_STEPS]
    if unknown:
        raise ValueError(f"Unknown preprocessing steps: {', '.join(unknown)}")
    return steps


def _enabled_from_check_values(check_values: List[bool]) -> List[str]:
    """
    Map parallel checkbox values to enabled preprocessing step keys.

    Args:
        check_values: Checkbox values in ``KNOWN_STEPS`` order.

    Returns:
        List[str]: Enabled step keys preserving ``KNOWN_STEPS`` order.
    """
    return [KNOWN_STEPS[idx] for idx, checked in enumerate(check_values) if checked]


def _order_number_updates(order: List[str]) -> Tuple[Any, ...]:
    """
    Build gr.update payloads for per-step order number widgets.

    Args:
        order: Current execution order (enabled steps only).

    Returns:
        Tuple[Any, ...]: One gr.update per step in ``KNOWN_STEPS`` order.
    """
    return tuple(
        gr.update(value=(order.index(step_key) + 1) if step_key in order else None)
        for step_key in KNOWN_STEPS
    )


def _apply_order_from_number(
    step_key: str,
    new_num: Optional[Union[int, float]],
    current_order: List[str],
    enabled: List[str],
) -> List[str]:
    """
    Reorder one enabled step by editing its sequence number.

    Args:
        step_key: Step key whose order number was edited.
        new_num: Target 1-based position entered by the user.
        current_order: Current execution order.
        enabled: Enabled step keys.

    Returns:
        List[str]: Updated execution order.
    """
    order = [step for step in (current_order or []) if step in enabled]
    if step_key not in enabled:
        return order
    if new_num is None or (isinstance(new_num, str) and not str(new_num).strip()):
        return _sync_order_with_enabled(enabled, order)
    if step_key in order:
        order.remove(step_key)
    target = max(1, min(int(new_num), len(enabled)))
    order.insert(target - 1, step_key)
    return order


def _move_step_in_order(order: List[str], step_key: str, direction: int) -> List[str]:
    """
    Move a step up (-1) or down (+1) within the current order list.

    Args:
        order: Current execution order.
        step_key: Step key to move.
        direction: -1 for up, +1 for down.

    Returns:
        List[str]: Updated order; unchanged when move is not possible.
    """
    if step_key not in order:
        return list(order)
    order = list(order)
    idx = order.index(step_key)
    new_idx = idx + direction
    if 0 <= new_idx < len(order):
        order[idx], order[new_idx] = order[new_idx], order[idx]
    return order


def _sync_order_with_enabled(enabled: List[str], current_order: List[str]) -> List[str]:
    """
    Reconcile execution order when the user toggles enabled preprocessing steps.

    Keeps relative order for steps that remain enabled and appends newly enabled steps
    following the default ``KNOWN_STEPS`` sequence.

    Args:
        enabled: Step keys currently selected in the CheckboxGroup.
        current_order: Previous execution order (may contain disabled steps).

    Returns:
        List[str]: Ordered list containing only enabled step keys.
    """
    enabled_set = set(enabled or [])
    preserved = [step for step in (current_order or []) if step in enabled_set]
    for step in KNOWN_STEPS:
        if step in enabled_set and step not in preserved:
            preserved.append(step)
    return preserved


def _panel_visible(step_key: str, enabled: List[str]) -> Dict[str, Any]:
    """Return a gr.update dict toggling a parameter panel based on enabled steps."""
    return gr.update(visible=step_key in set(enabled or []))


def _modalities_display_text(modalities: List[str]) -> str:
    """Format modality keys for read-only GUI text fields."""
    return ", ".join(modalities)


def _resolve_modalities_for_run(
    data_dir: str,
    auto_select: bool,
    modalities_text: str,
) -> Tuple[List[str], Optional[str]]:
    """
    Resolve modality keys at run time from the auto-detected display text or rescan data_dir.

    Args:
        data_dir: Absolute input data root directory.
        auto_select: Whether to auto-pick the first file inside modality folders.
        modalities_text: Comma-separated modality string shown in the GUI.

    Returns:
        Tuple[List[str], Optional[str]]: Modality keys and an error message when empty.
    """
    parsed = parse_comma_list(modalities_text)
    if parsed:
        return parsed, None
    modalities, status = discover_modalities_from_data_dir(data_dir, auto_select)
    if modalities:
        return modalities, None
    return [], status or "No modalities detected. Check data_dir layout."


def _toggle_backend_panels(backend: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Show ANTs/SimpleITK vs elastix-specific registration fields."""
    is_elastix: bool = backend == "elastix"
    return (
        gr.update(visible=not is_elastix),
        gr.update(visible=is_elastix),
        gr.update(visible=backend == "simpleitk"),
    )


def render_preprocess_tab() -> None:
    """Render the image preprocessing tab inside a parent Gradio Blocks context."""
    gr.Markdown(
        "Configure a custom preprocessing pipeline (registration, resampling, intensity "
        "normalization, etc.) and batch-process all subjects."
    )

    with gr.Row():
        existing_yaml = gr.Textbox(label="Load existing preprocessing YAML (optional)", scale=4)
        browse_prep_config_btn = gr.Button("Browse config", scale=1)

    with gr.Group():
        gr.Markdown("### 1. Paths and system settings")
        with gr.Row():
            data_dir = gr.Textbox(label="Input data root directory *", scale=4)
            prep_data_btn = gr.Button("Browse", scale=1)
            detect_modalities_btn = gr.Button("Detect modalities", scale=1)
        with gr.Row():
            out_dir = gr.Textbox(label="Output directory *", scale=4)
            prep_out_btn = gr.Button("Browse", scale=1)
        with gr.Row():
            processes = gr.Number(label="Parallel workers (CPU)", value=1, minimum=1, maximum=32, step=1)
            auto_select = gr.Checkbox(label="Auto-select first file in modality folder", value=True)
            random_state = gr.Number(label="Random state", value=42, precision=0)
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
        save_intermediate = gr.Checkbox(label="save_intermediate", value=False)
        intermediate_steps = gr.CheckboxGroup(
            label="intermediate_steps (empty = save all enabled steps)",
            choices=KNOWN_STEPS,
            value=[],
        )

    submit_btn = gr.Button("Validate and run preprocessing", variant="primary")
    log_output = gr.Textbox(label="Console log", lines=18, interactive=False)
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
            text = _modalities_display_text(modalities)
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

    def _on_step_toggle(
        data_dir_val: str,
        auto_select_val: bool,
        current_order: List[str],
        *check_values: bool,
    ) -> Tuple[Any, ...]:
        """Sync order numbers and parameter panels when any step checkbox toggles."""
        enabled = _enabled_from_check_values(list(check_values))
        new_order = _sync_order_with_enabled(enabled, current_order)
        panel_updates = tuple(_panel_visible(key, enabled) for key in step_panel_keys)
        return (
            new_order,
            *_order_number_updates(new_order),
            *panel_updates,
        )

    def _on_step_move(
        step_key: str,
        direction: int,
        current_order: List[str],
        *check_values: bool,
    ) -> Tuple[Any, ...]:
        """Move one enabled step up/down and refresh order numbers."""
        enabled = _enabled_from_check_values(list(check_values))
        order = [step for step in current_order if step in enabled]
        if step_key not in enabled:
            return (order, *_order_number_updates(order))
        new_order = _move_step_in_order(order, step_key, direction)
        return (new_order, *_order_number_updates(new_order))

    def _on_step_order_edit(
        step_key: str,
        new_num: Optional[float],
        current_order: List[str],
        *check_values: bool,
    ) -> Tuple[Any, ...]:
        """Apply a manually edited sequence number for one step."""
        enabled = _enabled_from_check_values(list(check_values))
        new_order = _apply_order_from_number(step_key, new_num, current_order, enabled)
        return (new_order, *_order_number_updates(new_order))

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
        _toggle_backend_panels,
        inputs=reg_backend,
        outputs=[box_ants_sitk, box_elastix, box_sitk],
    )

    def browse_file() -> Any:
        path: Optional[str] = select_local_path("file", "Select preprocessing YAML")
        return path if path else gr.update()

    def browse_folder() -> Any:
        path: Optional[str] = select_local_path("folder", "Select directory")
        return path if path else gr.update()

    browse_prep_config_btn.click(browse_file, outputs=existing_yaml)
    prep_data_btn.click(browse_folder, outputs=data_dir).then(
        _apply_detected_modalities,
        inputs=detect_modalities_inputs,
        outputs=detect_modalities_outputs,
    )
    prep_out_btn.click(browse_folder, outputs=out_dir)

    def load_config(yaml_path: str) -> List[Any]:
        """Hydrate form widgets from an existing preprocessing YAML file."""
        noop = gr.update()
        if not yaml_path or not os.path.exists(yaml_path):
            return [noop] * LOAD_OUTPUT_COUNT

        loaded: Optional[Dict[str, Any]] = load_config_yaml(yaml_path)
        if not loaded:
            return [noop] * LOAD_OUTPUT_COUNT

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
            loaded_data_dir,
            bool(auto_select_loaded),
        )
        if detected_modalities:
            modality_text = _modalities_display_text(detected_modalities)
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
            loaded_data_dir,
            loaded.get("out_dir", ""),
            int(loaded.get("processes", 1)),
            auto_select_loaded,
            int(loaded.get("random_state", 42)),
            detect_status,
            *[step_key in enabled_keys for step_key in KNOWN_STEPS],
            order_keys,
            *_order_number_updates(order_keys),
            _panel_visible("n4_correction", enabled_keys),
            _panel_visible("resample", enabled_keys),
            _panel_visible("zscore_normalization", enabled_keys),
            _panel_visible("registration", enabled_keys),
            _panel_visible("histogram_standardization", enabled_keys),
            _panel_visible("adaptive_histogram_equalization", enabled_keys),
            _panel_visible("reorientation", enabled_keys),
            _panel_visible("dcm2nii", enabled_keys),
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
            reg.get("elastix_path", "") or "",
            reg.get("transformix_path", "") or "",
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
            dcm2.get("dcm2niix_path", "") or "",
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
            _dcm2_single_file_from_yaml(dcm2.get("single_file_mode")),
            dict_to_yaml_block(extra_blocks),
            save_opt.get("save_intermediate", False),
            save_opt.get("intermediate_steps", []) or [],
        ]

    load_outputs = [
        data_dir, out_dir, processes, auto_select, random_state,
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
        extra_prep_yaml, save_intermediate, intermediate_steps,
    ]
    existing_yaml.change(load_config, inputs=existing_yaml, outputs=load_outputs)

    def run_pipeline(*args: Any):
        """Validate config, save YAML, and run preprocessing with live logs."""
        (
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

        enabled_set = set(_enabled_from_check_values(list(step_check_vals)))
        if not enabled_set:
            yield "❌ Select at least one preprocessing step.", gr.update(visible=False)
            return

        # Resolve relative paths to absolute before building the config.
        data_dir_abs = abs_path(data_dir_val)
        out_dir_abs = abs_path(out_dir_val)

        try:
            step_order_all: List[str] = _parse_step_order(step_order_val)
        except ValueError as exc:
            yield f"❌ {exc}", gr.update(visible=False)
            return

        step_order = [step for step in step_order_all if step in enabled_set]
        if not step_order:
            yield "❌ Step execution order is empty for the selected steps.", gr.update(visible=False)
            return

        modalities, modality_err = _resolve_modalities_for_run(
            data_dir_abs,
            bool(auto_select_val),
            str(n4_images_val),
        )
        dcm2_modalities, dcm2_err = _resolve_modalities_for_run(
            data_dir_abs,
            bool(auto_select_val),
            str(dcm2_images_val),
        )
        if not dcm2_modalities:
            dcm2_modalities = ["dicom"]

        built_steps: Dict[str, Dict[str, Any]] = {}

        if "n4_correction" in enabled_set:
            if not modalities:
                yield f"❌ n4_correction: {modality_err}", gr.update(visible=False)
                return
            built_steps["n4_correction"] = {
                "images": modalities,
                "num_fitting_levels": int(n4_levels_val),
            }
        if "resample" in enabled_set:
            if not modalities:
                yield f"❌ resample: {modality_err}", gr.update(visible=False)
                return
            try:
                spacing = [float(s) for s in parse_comma_list(res_spacing_val)]
                if len(spacing) != 3:
                    raise ValueError
            except ValueError:
                yield "❌ target_spacing must be three comma-separated numbers.", gr.update(visible=False)
                return
            built_steps["resample"] = {
                "images": modalities,
                "target_spacing": spacing,
            }
        if "zscore_normalization" in enabled_set:
            if not modalities:
                yield f"❌ zscore_normalization: {modality_err}", gr.update(visible=False)
                return
            zs_cfg: Dict[str, Any] = {
                "images": modalities,
                "only_inmask": zs_mask_val,
            }
            if zs_mask_key_val and str(zs_mask_key_val).strip():
                zs_cfg["mask_key"] = str(zs_mask_key_val).strip()
            built_steps["zscore_normalization"] = zs_cfg

        if "registration" in enabled_set:
            if not modalities:
                yield f"❌ registration: {modality_err}", gr.update(visible=False)
                return
            fixed_image = str(reg_fixed_val).strip() or modalities[0]
            if fixed_image not in modalities:
                yield (
                    f"❌ fixed_image \"{fixed_image}\" must be one of the detected modalities: "
                    f"{', '.join(modalities)}.",
                    gr.update(visible=False),
                )
                return
            reg_cfg: Dict[str, Any] = {
                "images": modalities,
                "fixed_image": fixed_image,
                "backend": reg_backend_val,
                "use_mask": reg_mask_val,
                "replace_by_fixed_image_mask": reg_replace_mask_val,
            }
            if reg_mask_key_val and str(reg_mask_key_val).strip():
                reg_cfg["mask_key"] = str(reg_mask_key_val).strip()
            if reg_backend_val == "elastix":
                files_raw = parse_comma_list(elastix_files_val)
                if not files_raw:
                    yield "❌ elastix_parameter_files is required when backend=elastix.", gr.update(visible=False)
                    return
                # Resolve elastix paths to absolute
                abs_files = [abs_path(f) for f in files_raw]
                reg_cfg["elastix_parameter_files"] = abs_files if len(abs_files) > 1 else abs_files[0]
                if elastix_path_val.strip():
                    reg_cfg["elastix_path"] = abs_path(elastix_path_val.strip())
                if transformix_path_val.strip():
                    reg_cfg["transformix_path"] = abs_path(transformix_path_val.strip())
                if int(elastix_threads_val or 0) > 0:
                    reg_cfg["elastix_threads"] = int(elastix_threads_val)
                if elastix_overrides_val and str(elastix_overrides_val).strip():
                    try:
                        overrides = yaml_block_to_dict(str(elastix_overrides_val))
                        if overrides:
                            reg_cfg["elastix_parameter_overrides"] = overrides
                    except ValueError as exc:
                        yield f"❌ elastix_parameter_overrides: {exc}", gr.update(visible=False)
                        return
            else:
                reg_cfg["type_of_transform"] = reg_transform_val
                reg_cfg["metric"] = reg_metric_val
                if reg_optimizer_val.strip():
                    reg_cfg["optimizer"] = reg_optimizer_val.strip()
                if reg_backend_val == "simpleitk":
                    reg_cfg["number_of_histogram_bins"] = int(sitk_bins_val)
                    reg_cfg["metric_sampling_percentage"] = float(sitk_sampling_val)
                    reg_cfg["shrink_factors_per_level"] = [int(x) for x in parse_comma_list(sitk_shrink_val)]
                    reg_cfg["smoothing_sigmas_per_level"] = [
                        float(x) for x in parse_comma_list(sitk_sigmas_val)
                    ]
                    reg_cfg["learning_rate"] = float(sitk_lr_val)
                    reg_cfg["number_of_iterations"] = int(sitk_iters_val)
                    reg_cfg["bspline_mesh_size"] = int(sitk_bspline_mesh_val)
                    reg_cfg["bspline_order"] = int(sitk_bspline_order_val)
            built_steps["registration"] = reg_cfg

        if "histogram_standardization" in enabled_set:
            if not modalities:
                yield f"❌ histogram_standardization: {modality_err}", gr.update(visible=False)
                return
            try:
                percentiles = [float(x) for x in parse_comma_list(hist_percentiles_val)]
            except ValueError:
                yield "❌ histogram percentiles must be comma-separated numbers.", gr.update(visible=False)
                return
            hist_cfg: Dict[str, Any] = {
                "images": modalities,
                "percentiles": percentiles,
                "target_min": float(hist_target_min_val),
                "target_max": float(hist_target_max_val),
            }
            if hist_mask_key_val and str(hist_mask_key_val).strip():
                hist_cfg["mask_key"] = str(hist_mask_key_val).strip()
            built_steps["histogram_standardization"] = hist_cfg

        if "adaptive_histogram_equalization" in enabled_set:
            if not modalities:
                yield f"❌ adaptive_histogram_equalization: {modality_err}", gr.update(visible=False)
                return
            built_steps["adaptive_histogram_equalization"] = {
                "images": modalities,
                "alpha": float(ahe_alpha_val),
                "beta": float(ahe_beta_val),
                "radius": int(ahe_radius_val),
            }

        if "reorientation" in enabled_set:
            if not modalities:
                yield f"❌ reorientation: {modality_err}", gr.update(visible=False)
                return
            built_steps["reorientation"] = {
                "images": modalities,
                "target_orientation": str(reorient_target_val).strip(),
                "mode": str(reorient_mode_val).strip(),
            }

        if "dcm2nii" in enabled_set:
            if not dcm2_modalities:
                yield f"❌ dcm2nii: {dcm2_err or 'No modality key available.'}", gr.update(visible=False)
                return
            dcm2_cfg: Dict[str, Any] = {
                "images": dcm2_modalities,
                "compress": dcm2_compress_val,
                "anonymize": dcm2_anonymize_val,
                "ignore_derived": dcm2_ignore_derived_val,
                "crop_images": dcm2_crop_val,
                "generate_json": dcm2_json_val,
                "verbose": dcm2_verbose_val,
                "batch_mode": dcm2_batch_val,
                "adjacent_dicoms": dcm2_adjacent_val,
                "merge_slices": str(dcm2_merge_val),
            }
            if dcm2_path_val.strip():
                dcm2_cfg["dcm2niix_path"] = dcm2_path_val.strip()
            if dcm2_format_val.strip():
                dcm2_cfg["filename_format"] = dcm2_format_val.strip()
            single_file_mode = _dcm2_single_file_to_yaml(str(dcm2_single_file_val))
            if single_file_mode is not None:
                dcm2_cfg["single_file_mode"] = single_file_mode
            built_steps["dcm2nii"] = dcm2_cfg

        if extra_prep_yaml_val and str(extra_prep_yaml_val).strip():
            try:
                extra_block = yaml_block_to_dict(str(extra_prep_yaml_val))
                if extra_block:
                    built_steps.update(extra_block)
            except ValueError as exc:
                yield f"❌ {exc}", gr.update(visible=False)
                return

        prep_steps_config: Dict[str, Dict[str, Any]] = {}
        for step_key in step_order:
            if step_key in built_steps:
                prep_steps_config[step_key] = built_steps[step_key]

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
            yield (
                f"💾 Config validated and saved to {gui_config_path}\n🚀 Running preprocessing...",
                gr.update(visible=False),
            )
            for log_text in run_background_job(
                run_preprocess,
                args=(gui_config_path,),
                log_file=Path(out_dir_abs) / "processing.log",
            ):
                yield log_text, gr.update(visible=True)
        except ValidationError as val_err:
            friendly = translate_pydantic_error(val_err)
            yield "⚠️ Validation errors:\n" + "\n".join(f"- {e}" for e in friendly), gr.update(visible=False)
        except Exception as exc:  # noqa: BLE001
            val_msgs = extract_validation_msgs(exc)
            if val_msgs:
                yield "⚠️ Validation errors:\n" + "\n".join(f"- {e}" for e in val_msgs), gr.update(visible=False)
            else:
                yield f"❌ Run failed: {exc}", gr.update(visible=False)

    submit_btn.click(
        run_pipeline,
        inputs=[
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
    open_dir_btn.click(lambda p: open_directory(p), inputs=out_dir)
