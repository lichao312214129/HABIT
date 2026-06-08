# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Machine learning modeling tab for Gradio GUI."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
from pydantic import ValidationError

from habit.cli_commands.commands.cmd_ml import run_kfold, run_ml
from habit.core.machine_learning.config_schemas import MLConfig
from habit.gui.pipeline_runner import run_background_job
from habit.gui.utils import (
    load_config_yaml,
    open_directory,
    parse_comma_list,
    save_config_yaml,
    select_local_path,
    translate_pydantic_error,
)

MODEL_CHOICES: List[str] = [
    "LogisticRegression",
    "RandomForest",
    "SVM",
    "XGBoost",
    "GaussianNB",
    "MultinomialNB",
    "BernoulliNB",
    "KNN",
    "DecisionTree",
    "AdaBoost",
    "MLP",
    "GradientBoosting",
    "AutoGluonTabular",
]

SELECTOR_CHOICES: List[str] = [
    "variance",
    "correlation",
    "statistical_test",
    "vif",
    "lasso",
    "rfecv",
    "mrmr",
    "stepwise",
    "anova",
    "chi2",
    "univariate_logistic",
    "icc",
]

NORM_CHOICES: List[str] = [
    "z_score",
    "min_max",
    "robust",
    "max_abs",
    "normalizer",
    "quantile",
    "power",
]

PLOT_TYPES: List[str] = ["roc", "dca", "calibration", "pr", "confusion", "shap"]


def render_ml_tab() -> None:
    """Render ML train/predict and K-fold tab."""
    gr.Markdown("Train or predict ML models on feature tables with feature selection and visualization.")

    with gr.Row():
        existing_yaml = gr.Textbox(label="Load existing ML YAML (optional)", scale=4)
        browse_cfg_btn = gr.Button("Browse config", scale=1)

    workflow_type = gr.Radio(
        label="Workflow",
        choices=["Holdout train/predict", "K-Fold cross-validation"],
        value="Holdout train/predict",
    )
    run_mode = gr.Dropdown(label="run_mode", choices=["train", "predict"], value="train")

    with gr.Group():
        gr.Markdown("### 1. Data input")
        with gr.Row():
            csv_path = gr.Textbox(label="input[0].path (CSV/Excel) *", scale=4)
            csv_btn = gr.Button("Browse", scale=1)
        with gr.Row():
            input_name = gr.Textbox(label="input[0].name", value="gui_")
            subject_id_col = gr.Textbox(label="subject_id_col *", value="Subject")
            label_col = gr.Textbox(label="label_col *", value="label")
        features_cols = gr.Textbox(label="input[0].features (comma-separated; empty=all)", value="")
        split_col = gr.Textbox(label="input[0].split_col (optional)", value="")

    with gr.Row():
        output_dir = gr.Textbox(label="output *", scale=4)
        out_btn = gr.Button("Browse", scale=1)

    with gr.Row(visible=False) as pipe_row:
        pipeline_path = gr.Textbox(label="pipeline_path * (predict)", scale=4)
        pipe_btn = gr.Button("Browse", scale=1)

    with gr.Group(visible=True) as train_box:
        gr.Markdown("### 2. Train / CV settings")
        with gr.Row():
            split_method = gr.Dropdown(
                label="split_method",
                choices=["stratified", "random", "custom"],
                value="stratified",
            )
            test_size = gr.Number(label="test_size", value=0.3, minimum=0.05, maximum=0.95)
            n_splits = gr.Number(label="n_splits (K-Fold)", value=5, minimum=2, maximum=20, step=1)
        with gr.Row():
            random_state = gr.Number(label="random_state", value=42, precision=0)
            stratified = gr.Checkbox(label="stratified (K-Fold)", value=True)
        with gr.Row(visible=False) as custom_split_row:
            train_ids_file = gr.Textbox(label="train_ids_file *", scale=4)
            train_ids_btn = gr.Button("Browse", scale=1)
            test_ids_file = gr.Textbox(label="test_ids_file *", scale=4)
            test_ids_btn = gr.Button("Browse", scale=1)

        gr.Markdown("### 3. Normalization and resampling")
        with gr.Row():
            norm_method = gr.Dropdown(label="normalization.method", choices=NORM_CHOICES, value="z_score")
            resampling_enabled = gr.Checkbox(label="resampling.enabled", value=False)
        with gr.Row():
            resampling_method = gr.Dropdown(
                label="resampling.method",
                choices=["random_over", "random_under", "smote"],
                value="random_over",
            )
            resampling_position = gr.Dropdown(
                label="resampling.position",
                choices=[
                    "before_feature_selection",
                    "before_normalization",
                    "after_normalization",
                    "before_model",
                ],
                value="before_model",
            )
            resampling_ratio = gr.Number(label="resampling.ratio", value=1.0, minimum=0.1, maximum=1.0)

        gr.Markdown("### 4. Feature selection (sequential)")
        enabled_selectors = gr.CheckboxGroup(label="feature_selection_methods", choices=SELECTOR_CHOICES, value=[])
        with gr.Row():
            corr_threshold = gr.Number(label="correlation.threshold", value=0.8)
            var_threshold = gr.Number(label="variance.threshold", value=0.01)
            stat_p = gr.Number(label="statistical_test.p_threshold", value=0.05)
            max_vif = gr.Number(label="vif.max_vif", value=10, precision=0)

        gr.Markdown("### 5. Models")
        selected_models = gr.CheckboxGroup(label="models *", choices=MODEL_CHOICES, value=["LogisticRegression"])

        gr.Markdown("### 6. Output flags")
        with gr.Row():
            is_visualize = gr.Checkbox(label="is_visualize", value=True)
            is_save_model = gr.Checkbox(label="is_save_model", value=True)
        plot_types = gr.CheckboxGroup(label="visualization.plot_types", choices=PLOT_TYPES, value=PLOT_TYPES)
        with gr.Row():
            plot_dpi = gr.Number(label="visualization.dpi", value=300, precision=0)
            plot_format = gr.Dropdown(label="visualization.format", choices=["png", "pdf"], value="png")

    with gr.Accordion("Predict-mode options", open=False):
        evaluate = gr.Checkbox(label="evaluate", value=False)
        output_label_col = gr.Textbox(label="output_label_col", value="predicted_label")
        output_prob_col = gr.Textbox(label="output_prob_col", value="predicted_probability")
        binary_pos_idx = gr.Number(label="binary_positive_class_index", value=1, precision=0)

    submit_btn = gr.Button("Validate and run ML workflow", variant="primary")
    log_output = gr.Textbox(label="Console log", lines=18, interactive=False)
    open_dir_btn = gr.Button("Open output folder", visible=False)

    def on_workflow(wf: str) -> tuple:
        is_kfold = "K-Fold" in wf
        return (
            gr.update(value="train" if is_kfold else "train"),
            gr.update(interactive=not is_kfold),
        )

    workflow_type.change(on_workflow, inputs=workflow_type, outputs=[run_mode, run_mode])

    def on_run_mode(mode: str) -> Dict[str, Any]:
        return gr.update(visible=mode == "predict")

    run_mode.change(on_run_mode, inputs=run_mode, outputs=pipe_row)

    def on_split_method(method: str) -> Dict[str, Any]:
        return gr.update(visible=method == "custom")

    split_method.change(on_split_method, inputs=split_method, outputs=custom_split_row)

    def browse_file() -> Any:
        p = select_local_path("file", "Select file")
        return p if p else gr.update()

    def browse_folder() -> Any:
        p = select_local_path("folder", "Select folder")
        return p if p else gr.update()

    browse_cfg_btn.click(browse_file, outputs=existing_yaml)
    csv_btn.click(browse_file, outputs=csv_path)
    out_btn.click(browse_folder, outputs=output_dir)
    pipe_btn.click(browse_file, outputs=pipeline_path)
    train_ids_btn.click(browse_file, outputs=train_ids_file)
    test_ids_btn.click(browse_file, outputs=test_ids_file)

    def load_yaml(path: str) -> List[Any]:
        noop = gr.update()
        if not path or not os.path.exists(path):
            return [noop] * 36
        loaded = load_config_yaml(path)
        if not loaded:
            return [noop] * 36
        inp = (loaded.get("input") or [{}])[0]
        is_kfold = "n_splits" in loaded and loaded.get("split_method") != "custom"
        wf = "K-Fold cross-validation" if is_kfold else "Holdout train/predict"
        norm = loaded.get("normalization", {}) or {}
        res = loaded.get("resampling", {}) or {}
        fsm = loaded.get("feature_selection_methods", []) or []
        enabled = [x.get("method") for x in fsm if isinstance(x, dict)]
        models = list((loaded.get("models") or {}).keys())
        vis = loaded.get("visualization", {}) or {}
        corr_p = next((x.get("params", {}) for x in fsm if x.get("method") == "correlation"), {})
        var_p = next((x.get("params", {}) for x in fsm if x.get("method") == "variance"), {})
        stat_p = next((x.get("params", {}) for x in fsm if x.get("method") == "statistical_test"), {})
        vif_p = next((x.get("params", {}) for x in fsm if x.get("method") == "vif"), {})
        feats = inp.get("features")
        return [
            wf,
            loaded.get("run_mode", "train"),
            inp.get("path", ""),
            inp.get("name", "gui_"),
            inp.get("subject_id_col", "Subject"),
            inp.get("label_col", "label"),
            ", ".join(feats) if feats else "",
            inp.get("split_col", "") or "",
            loaded.get("output", ""),
            loaded.get("pipeline_path", "") or "",
            loaded.get("split_method", "stratified"),
            float(loaded.get("test_size", 0.3)),
            int(loaded.get("n_splits", 5)),
            int(loaded.get("random_state", 42)),
            loaded.get("stratified", True),
            loaded.get("train_ids_file", "") or "",
            loaded.get("test_ids_file", "") or "",
            norm.get("method", "z_score"),
            res.get("enabled", False),
            res.get("method", "random_over"),
            res.get("position", "before_model"),
            float(res.get("ratio", 1.0)),
            enabled,
            float(corr_p.get("threshold", 0.8)),
            float(var_p.get("threshold", 0.01)),
            float(stat_p.get("p_threshold", 0.05)),
            int(vif_p.get("max_vif", 10)),
            models or ["LogisticRegression"],
            loaded.get("is_visualize", True),
            loaded.get("is_save_model", True),
            vis.get("plot_types", PLOT_TYPES),
            int(vis.get("dpi", 300)),
            vis.get("format", "png"),
            loaded.get("evaluate", False),
            loaded.get("output_label_col", "predicted_label"),
            loaded.get("output_prob_col", "predicted_probability"),
            int(loaded.get("binary_positive_class_index", 1)),
        ]

    load_outputs = [
        workflow_type, run_mode, csv_path, input_name, subject_id_col, label_col,
        features_cols, split_col, output_dir, pipeline_path,
        split_method, test_size, n_splits, random_state, stratified,
        train_ids_file, test_ids_file,
        norm_method, resampling_enabled, resampling_method, resampling_position, resampling_ratio,
        enabled_selectors, corr_threshold, var_threshold, stat_p, max_vif,
        selected_models, is_visualize, is_save_model, plot_types, plot_dpi, plot_format,
        evaluate, output_label_col, output_prob_col, binary_pos_idx,
    ]
    existing_yaml.change(load_yaml, inputs=existing_yaml, outputs=load_outputs)

    def _build_selectors(
        enabled: List[str],
        corr_th: float,
        var_th: float,
        stat_p_th: float,
        vif_max: float,
        rs: int,
    ) -> List[Dict[str, Any]]:
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
        if "vif" in enabled:
            methods.append({
                "method": "vif",
                "params": {"max_vif": int(vif_max), "visualize": False, "before_z_score": False},
            })
        for name in enabled:
            if name in {"lasso", "rfecv", "mrmr", "stepwise", "anova", "chi2", "univariate_logistic", "icc"}:
                methods.append({"method": name, "params": {"random_state": rs}})
        return methods

    def _build_models(selected: List[str], rs: int) -> Dict[str, Any]:
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

    def run_ml_pipeline(*args: Any):
        (
            wf, mode, csv_p, in_name, subj_col, lbl_col, feat_cols, split_c, out_p, pipe_p,
            split_m, test_sz, n_spl, rs, strat,
            train_ids, test_ids,
            norm_m, res_en, res_method, res_pos, res_ratio,
            sel_enabled, corr_th, var_th, stat_p_th, vif_max,
            models_sel, vis_en, save_m, plots, dpi, fmt,
            eval_en, out_lbl, out_prob, bin_pos,
        ) = args

        is_kfold = "K-Fold" in wf
        if not csv_p or not out_p:
            yield "❌ input path and output are required.", gr.update(visible=False)
            return
        if mode == "predict" and not pipe_p:
            yield "❌ pipeline_path is required in predict mode.", gr.update(visible=False)
            return
        if split_m == "custom" and (not train_ids or not test_ids):
            yield "❌ train_ids_file and test_ids_file are required for custom split.", gr.update(visible=False)
            return

        input_entry: Dict[str, Any] = {
            "path": csv_p,
            "name": in_name,
            "subject_id_col": subj_col,
            "label_col": lbl_col,
        }
        feats = parse_comma_list(feat_cols)
        if feats:
            input_entry["features"] = feats
        if split_c.strip():
            input_entry["split_col"] = split_c.strip()

        config_data: Dict[str, Any] = {
            "run_mode": "train" if is_kfold else mode,
            "pipeline_path": pipe_p if mode == "predict" and not is_kfold else None,
            "input": [input_entry],
            "output": out_p,
            "random_state": int(rs),
            "is_visualize": vis_en,
            "is_save_model": save_m,
            "visualization": {
                "enabled": vis_en,
                "plot_types": list(plots or PLOT_TYPES),
                "dpi": int(dpi),
                "format": fmt,
            },
            "evaluate": eval_en,
            "output_label_col": out_lbl,
            "output_prob_col": out_prob,
            "binary_positive_class_index": int(bin_pos),
        }

        if is_kfold or mode == "train":
            if not models_sel:
                yield "❌ Select at least one model for training.", gr.update(visible=False)
                return
            config_data.update({
                "split_method": split_m if not is_kfold else "stratified",
                "test_size": float(test_sz),
                "n_splits": int(n_spl),
                "stratified": strat,
                "train_ids_file": train_ids if split_m == "custom" else None,
                "test_ids_file": test_ids if split_m == "custom" else None,
                "normalization": {"method": norm_m, "params": {}},
                "resampling": {
                    "enabled": res_en,
                    "method": res_method,
                    "position": res_pos,
                    "ratio": float(res_ratio),
                    "random_state": int(rs),
                },
                "feature_selection_methods": _build_selectors(
                    list(sel_enabled or []), float(corr_th), float(var_th), float(stat_p_th), float(vif_max), int(rs)
                ),
                "models": _build_models(list(models_sel), int(rs)),
            })

        try:
            MLConfig(**config_data)
            os.makedirs(out_p, exist_ok=True)
            gui_path = str(Path(out_p) / "config_ml_gui.yaml")
            save_config_yaml(config_data, gui_path)
            yield f"💾 Config saved to {gui_path}\n🚀 Running ML...", gr.update(visible=False)

            if is_kfold:
                job = lambda: run_kfold(config_file=gui_path)
            else:
                job = lambda: run_ml(config_path=gui_path, mode=mode)

            for log_text in run_background_job(job):
                yield log_text, gr.update(visible=True)
        except ValidationError as err:
            msgs = translate_pydantic_error(err)
            yield "⚠️ Validation errors:\n" + "\n".join(f"- {m}" for m in msgs), gr.update(visible=False)
        except Exception as exc:  # noqa: BLE001
            yield f"❌ Failed: {exc}", gr.update(visible=False)

    submit_btn.click(run_ml_pipeline, inputs=load_outputs, outputs=[log_output, open_dir_btn])
    open_dir_btn.click(lambda p: open_directory(p), inputs=output_dir)
