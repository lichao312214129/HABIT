# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Machine learning modeling tab — schema-driven rewrite.

Replaces 600+ lines of hardcoded widgets with SchemaForm auto-generation.
Dynamic model/selector lists come from the registry, not hardcoded constants.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
from pydantic import ValidationError

from habit.cli_commands.commands.cmd_ml import run_kfold, run_ml
from habit.core.machine_learning.config_schemas import (
    InputFileConfig,
    MLConfig,
    ModelConfig,
    NormalizationConfig,
    ResamplingConfig,
    VisualizationConfig,
)
from habit.gui.job_controls import (
    job_end_button_updates,
    job_start_button_updates,
    on_stop_job_click,
)
from habit.gui.path_picker import PathPickerRegistry
from habit.gui.pipeline_runner import run_background_job
from habit.gui.project.step_hooks import finalize_step_from_log, mark_step_running
from habit.gui.schema_form import OverrideSpec, SchemaForm
from habit.gui.schema_form.registry import get_model_choices, get_selector_choices
from habit.gui.step_integration import register_project_path_fill
from habit.gui.step_registry import register_step_paths
from habit.gui.utils import (
    abs_path,
    extract_validation_msgs,
    load_config_yaml,
    load_gui_draft,
    open_directory,
    render_console_log,
    save_config_yaml,
    save_gui_draft,
    translate_pydantic_error,
    user_visible_path,
)
from habit.utils.docker_path_utils import display_path_value

# Plot type choices for visualization.plot_types override.
_PLOT_TYPES: List[str] = ["roc", "dca", "calibration", "pr", "confusion", "shap"]


def _build_selectors(
    enabled: List[str],
    corr_th: float,
    var_th: float,
    stat_p_th: float,
    vif_max: float,
    rs: int,
) -> List[Dict[str, Any]]:
    """Build feature_selection_methods list from GUI checkbox + params."""
    methods: List[Dict[str, Any]] = []
    if "variance" in enabled:
        methods.append({
            "method": "variance",
            "params": {"threshold": var_th, "plot_variances": True, "before_z_score": True},
        })
    if "correlation" in enabled:
        methods.append({
            "method": "correlation",
            "params": {"threshold": corr_th, "method": "spearman", "visualize": False, "before_z_score": False},
        })
    if "statistical_test" in enabled:
        methods.append({
            "method": "statistical_test",
            "params": {
                "p_threshold": stat_p_th,
                "normality_test_threshold": 0.05,
                "plot_importance": True,
                "before_z_score": False,
            },
        })
        methods.append({
            "method": "vif",
            "params": {"max_vif": int(vif_max), "visualize": False, "before_z_score": False},
        })
    for name in enabled:
        if name in {"lasso", "rfecv", "mrmr", "stepwise", "anova", "chi2", "univariate_logistic", "icc"}:
            methods.append({"method": name, "params": {"random_state": rs}})
    return methods


def _build_models(selected: List[str], rs: int) -> Dict[str, Any]:
    """Build models dict from GUI checkbox selection."""
    defaults: Dict[str, Dict[str, Any]] = {
        "LogisticRegression": {"random_state": rs, "max_iter": 1000, "C": 1.0, "penalty": "l2"},
        "RandomForest": {"random_state": rs, "n_estimators": 100, "max_features": "sqrt", "class_weight": "balanced"},
        "SVM": {"random_state": rs, "C": 1.0, "kernel": "rbf", "probability": True},
        "XGBoost": {"random_state": rs, "n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "objective": "binary:logistic"},
        "GaussianNB": {"var_smoothing": 1e-9},
        "KNN": {"n_neighbors": 5},
        "DecisionTree": {"random_state": rs, "max_depth": 5},
        "AdaBoost": {"random_state": rs, "n_estimators": 50},
        "MLP": {"random_state": rs, "max_iter": 500},
        "GradientBoosting": {"random_state": rs, "n_estimators": 100},
        "MultinomialNB": {},
        "BernoulliNB": {},
        "AutoGluonTabular": {"random_state": rs, "time_limit": 120},
    }
    return {m: {"params": defaults.get(m, {"random_state": rs})} for m in selected}


def render_ml_tab(
    demo=None,
    path_picker: PathPickerRegistry | None = None,
    project_root_state: Any | None = None,
) -> None:
    """Render ML train/predict and K-fold tab — fully schema-driven."""
    picker = path_picker if path_picker is not None else PathPickerRegistry()
    gr.Markdown(
        "**Step 5 — Machine Learning:** Train or apply models on feature tables "
        "with optional feature selection and evaluation plots."
    )

    # --- Quick presets (convenience layer, not schema-driven) ---
    with gr.Accordion("Quick start (clinical defaults)", open=True):
        gr.Markdown(
            "Use **Fill paths from project** then select **LogisticRegression** with "
            "**correlation** feature selection for a standard binary classification workflow."
        )
        quick_models = gr.CheckboxGroup(
            label="Quick model selection",
            choices=["LogisticRegression", "RandomForest", "SVM"],
            value=["LogisticRegression"],
        )
        apply_ml_quick_btn = gr.Button("Apply quick ML preset", size="sm")

    # --- YAML load ---
    with gr.Row():
        existing_yaml = gr.Textbox(label="Load existing ML YAML (optional)", scale=4)
        browse_cfg_btn = gr.Button("Browse config", scale=1)
    picker.add(browse_cfg_btn, existing_yaml, pick="file")

    # Workflow type is a GUI convenience (not in MLConfig schema).
    workflow_type = gr.Radio(
        label="Workflow",
        choices=["Holdout train/predict", "K-Fold cross-validation"],
        value="Holdout train/predict",
    )

    # === Main form: simple MLConfig fields ===
    main_form = SchemaForm.build(
        MLConfig,
        exclude={"input", "normalization", "resampling",
                 "feature_selection_methods", "models", "visualization"},
        group_order=["Mode", "Data", "Split", "Output", "Predict", "Advanced"],
        open_groups={"Mode", "Data", "Split"},
        path_picker=picker,
    )

    # === Data input sub-form (InputFileConfig) ===
    input_form = SchemaForm.build(
        InputFileConfig,
        exclude={"features_from_log", "pred_col"},
        group_order=["General"],
        open_groups={"General"},
        path_picker=picker,
        overrides={
            "path": OverrideSpec(label="Feature table (CSV/Excel) *"),
            "name": OverrideSpec(label="Feature name prefix", value="gui_"),
            "subject_id_col": OverrideSpec(label="Subject ID column *", value="Subject"),
            "label_col": OverrideSpec(label="Label column *", value="label"),
            "features": OverrideSpec(label="Feature columns (comma-separated; empty = all)"),
            "split_col": OverrideSpec(label="Split column (optional)"),
        },
    )

    # === Normalization sub-form ===
    norm_form = SchemaForm.build(
        NormalizationConfig,
        group_order=["General"],
        open_groups={"General"},
    )

    # === Resampling sub-form ===
    res_form = SchemaForm.build(
        ResamplingConfig,
        group_order=["General"],
        open_groups={"General"},
    )

    # === Feature selection (checkbox + params) ===
    with gr.Accordion("Feature selection (sequential)", open=False):
        enabled_selectors = gr.CheckboxGroup(
            label="feature_selection_methods",
            choices=get_selector_choices(),
            value=[],
        )
        with gr.Row():
            corr_threshold = gr.Number(label="correlation.threshold", value=0.8)
            var_threshold = gr.Number(label="variance.threshold", value=0.01)
            stat_p = gr.Number(label="statistical_test.p_threshold", value=0.05)
            max_vif = gr.Number(label="vif.max_vif", value=10, precision=0)

    # === Models (dynamic from registry) ===
    with gr.Accordion("Models", open=True):
        selected_models = gr.CheckboxGroup(
            label="models *",
            choices=get_model_choices(),
            value=["LogisticRegression"],
        )

    # === Visualization sub-form (plot_types override → CheckboxGroup) ===
    vis_form = SchemaForm.build(
        VisualizationConfig,
        group_order=["General"],
        open_groups={"General"},
        overrides={
            "plot_types": OverrideSpec(choices=_PLOT_TYPES),
        },
    )

    # Wire conditional visibility (run_mode → predict fields).
    main_form.wire_visibility()

    # --- Submit / stop / log ---
    with gr.Row():
        submit_btn = gr.Button("Validate and run ML workflow", variant="primary")
        stop_btn = gr.Button("Stop", interactive=False)
    log_output = render_console_log(lines=18, elem_id="habit-log-ml")
    open_dir_btn = gr.Button("Open output folder", visible=False)

    # --- Helper: build flat inputs list for events ---
    # Order: [project_root, workflow_type, *main, *input, *norm, *res,
    #         selectors+params, models, *vis]
    def _all_inputs() -> list:
        result = (
            [project_root_state, workflow_type]
            + main_form.inputs()
            + input_form.inputs()
            + norm_form.inputs()
            + res_form.inputs()
            + [enabled_selectors, corr_threshold, var_threshold, stat_p, max_vif]
            + [selected_models]
            + vis_form.inputs()
        )
        return [r for r in result if r is not None]

    # --- Workflow type → run_mode联动 ---
    def on_workflow(wf: str) -> list:
        is_kfold = "K-Fold" in wf
        run_mode_w = main_form.widget("run_mode")
        return [
            gr.update(value="train"),
            gr.update(interactive=not is_kfold),
        ]

    workflow_type.change(
        on_workflow,
        inputs=workflow_type,
        outputs=[main_form.widget("run_mode"), main_form.widget("run_mode")],
    )

    # --- Quick preset ---
    def _apply_quick(models: list) -> list:
        return [
            gr.update(value="Holdout train/predict"),
            gr.update(value="train"),
            gr.update(value=["correlation"]),
            gr.update(value=list(models or ["LogisticRegression"])),
            gr.update(value=["roc", "calibration", "dca"]),
        ]

    apply_ml_quick_btn.click(
        _apply_quick,
        inputs=[quick_models],
        outputs=[
            workflow_type,
            main_form.widget("run_mode"),
            enabled_selectors,
            selected_models,
            vis_form.widget("plot_types"),
        ],
    )

    # --- Load YAML ---
    def load_yaml(path: str) -> list:
        noop = gr.update()
        n_total = len(_all_inputs()) - 2  # minus project_root + workflow_type
        resolved = abs_path(path) if path and str(path).strip() else ""
        if not resolved or not os.path.exists(resolved):
            return [noop] * n_total

        loaded = load_config_yaml(resolved)
        if not loaded:
            return [noop] * n_total

        # Workflow type inference
        is_kfold = "n_splits" in loaded and loaded.get("split_method") != "custom"
        wf = "K-Fold cross-validation" if is_kfold else "Holdout train/predict"

        # Main form
        main_vals = main_form.populate(loaded)

        # Input form
        inp = (loaded.get("input") or [{}])[0]
        input_vals = input_form.populate(inp)

        # Normalization
        norm_vals = norm_form.populate(loaded.get("normalization", {}))

        # Resampling
        res_vals = res_form.populate(loaded.get("resampling", {}))

        # Feature selection
        fsm = loaded.get("feature_selection_methods", []) or []
        enabled = [x.get("method") for x in fsm if isinstance(x, dict)]
        corr_p = next((x.get("params", {}) for x in fsm if x.get("method") == "correlation"), {})
        var_p = next((x.get("params", {}) for x in fsm if x.get("method") == "variance"), {})
        stat_p_d = next((x.get("params", {}) for x in fsm if x.get("method") == "statistical_test"), {})
        vif_p = next((x.get("params", {}) for x in fsm if x.get("method") == "vif"), {})

        # Models
        models = list((loaded.get("models") or {}).keys())

        # Visualization
        vis_vals = vis_form.populate(loaded.get("visualization", {}))

        # Return: main + input + norm + res + [sel, corr, var, stat, vif] + models + vis
        return (
            main_vals + input_vals + norm_vals + res_vals
            + [enabled,
               float(corr_p.get("threshold", 0.8)),
               float(var_p.get("threshold", 0.01)),
               float(stat_p_d.get("p_threshold", 0.05)),
               int(vif_p.get("max_vif", 10)),
               models or ["LogisticRegression"]]
            + vis_vals
        )

    _load_outputs = (
        main_form.outputs()
        + input_form.outputs()
        + norm_form.outputs()
        + res_form.outputs()
        + [enabled_selectors, corr_threshold, var_threshold, stat_p, max_vif]
        + [selected_models]
        + vis_form.outputs()
    )
    existing_yaml.change(load_yaml, inputs=existing_yaml, outputs=_load_outputs)

    # --- Run handler ---
    def run_ml_pipeline(*args: Any):
        # Parse positional args
        idx = 0
        project_root = args[idx]; idx += 1
        wf = args[idx]; idx += 1

        main_vals = args[idx:idx + len(main_form.inputs())]; idx += len(main_form.inputs())
        input_vals = args[idx:idx + len(input_form.inputs())]; idx += len(input_form.inputs())
        norm_vals = args[idx:idx + len(norm_form.inputs())]; idx += len(norm_form.inputs())
        res_vals = args[idx:idx + len(res_form.inputs())]; idx += len(res_form.inputs())
        sel_enabled = args[idx]; idx += 1
        corr_th = args[idx]; idx += 1
        var_th = args[idx]; idx += 1
        stat_p_th = args[idx]; idx += 1
        vif_max = args[idx]; idx += 1
        models_sel = args[idx]; idx += 1
        vis_vals = args[idx:idx + len(vis_form.inputs())]; idx += len(vis_form.inputs())

        is_kfold = "K-Fold" in wf

        # Collect main form values
        config_data = main_form.collect(*main_vals)

        # Collect input form → build input list
        input_data = input_form.collect(*input_vals)
        csv_p = input_data.get("path", "")
        if not csv_p:
            yield "✗ input path is required.", gr.update(visible=False)
            return

        # Resolve relative paths
        input_data["path"] = abs_path(csv_p)
        out_p = config_data.get("output", "")
        if not out_p:
            yield "✗ output is required.", gr.update(visible=False)
            return
        out_p_abs = abs_path(out_p)
        config_data["output"] = out_p_abs

        config_data["input"] = [input_data]

        # Pipeline path (predict mode)
        mode = config_data.get("run_mode", "train")
        pipe_p = config_data.get("pipeline_path", "")
        if mode == "predict" and not is_kfold:
            if not pipe_p or not str(pipe_p).strip():
                yield "✗ pipeline_path is required in predict mode.", gr.update(visible=False)
                return
            config_data["pipeline_path"] = abs_path(pipe_p)
        else:
            config_data["pipeline_path"] = None

        # Train-only fields
        if is_kfold or mode == "train":
            rs = int(config_data.get("random_state", 42))
            config_data["normalization"] = norm_form.collect(*norm_vals)
            config_data["resampling"] = res_form.collect(*res_vals)
            config_data["feature_selection_methods"] = _build_selectors(
                list(sel_enabled or []), float(corr_th), float(var_th),
                float(stat_p_th), float(vif_max), rs,
            )
            if not models_sel:
                yield "✗ Select at least one model for training.", gr.update(visible=False)
                return
            config_data["models"] = _build_models(list(models_sel), rs)

            # Custom split paths
            split_m = config_data.get("split_method", "stratified")
            if split_m == "custom":
                train_ids = config_data.get("train_ids_file", "")
                test_ids = config_data.get("test_ids_file", "")
                if not train_ids or not test_ids:
                    yield "✗ train_ids_file and test_ids_file are required for custom split.", gr.update(visible=False)
                    return
                config_data["train_ids_file"] = abs_path(train_ids)
                config_data["test_ids_file"] = abs_path(test_ids)
            else:
                config_data["train_ids_file"] = None
                config_data["test_ids_file"] = None

            # K-Fold forces stratified
            if is_kfold:
                config_data["split_method"] = "stratified"

        # Visualization
        config_data["visualization"] = vis_form.collect(*vis_vals)

        try:
            MLConfig(**config_data)
            os.makedirs(out_p_abs, exist_ok=True)
            gui_path = str(Path(out_p_abs) / "config_ml_gui.yaml")
            save_config_yaml(config_data, gui_path)
            save_gui_draft("ml", gui_path)
            yield f"💾 Config saved to {gui_path}\n🚀 Running ML...", gr.update(visible=False)

            if is_kfold:
                job = lambda: run_kfold(config_file=gui_path)
                ml_log_name = "kfold_cv.log"
            else:
                job = lambda: run_ml(config_path=gui_path, mode=mode)
                ml_log_name = "prediction.log" if mode == "predict" else "processing.log"

            last_log = ""
            for log_text in run_background_job(job, log_file=Path(out_p_abs) / ml_log_name):
                last_log = log_text
                yield log_text, gr.update(visible=True)
            finalize_step_from_log(
                str(project_root) if project_root else "",
                "ml", last_log, config_path=gui_path, output_dir=out_p_abs,
            )
        except ValidationError as err:
            msgs = translate_pydantic_error(err)
            yield "⚠️ Validation errors:\n" + "\n".join(f"- {m}" for m in msgs), gr.update(visible=False)
        except Exception as exc:
            val_msgs = extract_validation_msgs(exc)
            if val_msgs:
                yield "⚠️ Validation errors:\n" + "\n".join(f"- {m}" for m in val_msgs), gr.update(visible=False)
            else:
                yield f"✗ Failed: {exc}", gr.update(visible=False)

    submit_btn.click(
        job_start_button_updates,
        outputs=[submit_btn, stop_btn],
    ).then(
        run_ml_pipeline,
        inputs=_all_inputs(),
        outputs=[log_output, open_dir_btn],
    ).then(
        job_end_button_updates,
        outputs=[submit_btn, stop_btn],
    )
    stop_btn.click(on_stop_job_click, inputs=[], outputs=[])
    open_dir_btn.click(lambda p: open_directory(p), inputs=main_form.widget("output"))

    # --- Step integration ---
    if project_root_state is not None:
        register_step_paths(
            "ml",
            ["csv_path", "output_dir", "pipeline_path"],
            [input_form.widget("path"), main_form.widget("output"), main_form.widget("pipeline_path")],
        )
        register_project_path_fill(
            project_root_state,
            "ml",
            ["csv_path", "output_dir", "pipeline_path"],
            [input_form.widget("path"), main_form.widget("output"), main_form.widget("pipeline_path")],
        )

    picker.finalize()

    # --- Restore draft YAML path on load ---
    if demo is not None:
        def _restore_ml_path() -> str:
            draft = load_gui_draft("ml") or ""
            return user_visible_path(draft) if draft else ""
        demo.load(_restore_ml_path, inputs=[], outputs=[existing_yaml])
