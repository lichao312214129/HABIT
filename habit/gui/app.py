# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""HABIT Web GUI main entry point (Gradio).

Project-centric workspace: open a project, auto-fill step paths, track workflow
progress in habit_project.json, and guide users through the six-step pipeline.
"""

import os
from pathlib import Path
from typing import Any, List, Tuple

from habit.utils.project_urls import DOCS_BASE_URL, DOCS_SITE_LABEL

apply_gradio_windows_patches()

import gradio as gr

from habit.gui.components.tab_dicom_sort import render_dicom_sort_tab
from habit.gui.components.tab_preprocess import render_preprocess_tab
from habit.gui.components.tab_habitat import render_habitat_tab
from habit.gui.components.tab_extract import render_extract_tab
from habit.gui.components.tab_ml import render_ml_tab
from habit.gui.components.tab_compare import render_compare_tab
from habit.gui.path_picker import (
    HABIT_PATH_PICKER_PORTAL_JS,
    PathPickerRegistry,
    should_use_web_path_picker,
)
from habit.gui.project.context import ProjectContext
from habit.gui.project.step_hooks import migrate_project_meta
from habit.gui.project_manager import (
    create_project,
    list_recent_projects,
    load_project,
    normalize_project_root,
    register_recent_project,
    save_project,
)
from habit.gui.step_registry import all_path_components, fill_all_registered_paths
from habit.gui.workflow_state import (
    STEP_LABELS,
    WORKFLOW_STEPS,
    get_progress_summary,
    render_stepper_html,
)
from habit.utils.browser_utils import (
    ensure_localhost_no_proxy,
    get_gradio_bind_host,
    get_gradio_browser_url,
    get_wsl_browser_access_hint,
    is_wsl,
    schedule_browser_open,
    should_use_host_browser,
)
from habit.utils.docker_path_utils import is_docker_runtime, list_docker_browse_roots
from habit.utils.gradio_patches import apply_gradio_patches

_NUM_STEPS = len(WORKFLOW_STEPS)

custom_css = """
/* ===== Home page ===== */
.habit-home-title {
    font-size: 2.2rem; font-weight: 800; color: #1E3A8A;
    text-align: center; margin: 1.5rem 0 0.3rem;
}
.habit-home-subtitle {
    font-size: 1.05rem; color: #6B7280; text-align: center; margin-bottom: 2rem;
}
.habit-home-card {
    border: 1px solid var(--border-color-primary, #ddd) !important;
    border-radius: 12px !important; padding: 20px 24px !important;
    background: var(--background-fill-primary, #fff) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}
.habit-home-card h3 { margin-top: 0 !important; color: #1E3A8A !important; }

/* ===== Workspace header ===== */
.habit-workspace-header h2 {
    margin: 0 0 4px !important; font-size: 1.4rem !important; color: #1E3A8A !important;
}
.habit-workspace-header .habit-project-path {
    font-size: 0.8rem !important; color: #6B7280 !important;
    font-family: ui-monospace, monospace !important;
}
.habit-progress-bar {
    margin-top: 6px !important; font-size: 0.85rem !important;
    color: #2563EB !important; font-weight: 600 !important;
}

/* ===== Stepper sidebar ===== */
.habit-stepper { display: flex !important; flex-direction: column !important; gap: 2px !important; margin-bottom: 1rem !important; }
.habit-step-item {
    display: flex !important; align-items: flex-start !important; gap: 8px !important;
    padding: 8px 10px !important; border-radius: 8px !important;
    border-left: 3px solid transparent !important; cursor: default !important;
}
.habit-step-item.active { border-left-color: #2563EB !important; background: rgba(37,99,235,0.08) !important; }
.habit-step-item.completed .habit-step-icon { color: #16A34A !important; }
.habit-step-item.in_progress .habit-step-icon { color: #F59E0B !important; }
.habit-step-item.failed .habit-step-icon { color: #DC2626 !important; }
.habit-step-num { font-weight: 700 !important; color: #6B7280 !important; min-width: 18px !important; font-size: 0.85rem !important; }
.habit-step-icon { font-weight: 700 !important; min-width: 16px !important; font-size: 0.85rem !important; }
.habit-step-text { display: flex !important; flex-direction: column !important; gap: 1px !important; overflow: hidden !important; }
.habit-step-label { font-weight: 600 !important; font-size: 0.88rem !important; }
.habit-step-desc { font-size: 0.72rem !important; color: #9CA3AF !important; line-height: 1.3 !important; }
.habit-step-btn { justify-content: flex-start !important; text-align: left !important; font-size: 0.85rem !important; margin-bottom: 2px !important; }
.habit-step-btn.active { border-color: #2563EB !important; font-weight: 700 !important; }

/* ===== Console log ===== */
.habit-console-log textarea { overflow-anchor: none !important; }
.habit-console-log { overflow-anchor: none; }

/* ===== Path picker overlay (preserved) ===== */
.habit-path-picker-clone { display: none !important; pointer-events: none !important; visibility: hidden !important; position: fixed !important; inset: -9999px !important; z-index: -1 !important; }
.habit-path-picker-overlay:not(.habit-path-picker-open):not(.habit-path-picker-clone) {
    display: none !important; pointer-events: none !important; visibility: hidden !important;
    position: fixed !important; inset: 0 !important; z-index: 99999 !important;
    width: 100vw !important; max-width: 100vw !important; min-height: 100vh !important;
    margin: 0 !important; padding: 0 !important; box-sizing: border-box !important;
    background: rgba(8,8,10,0.82) !important; backdrop-filter: blur(4px);
    overflow-y: auto !important; border: none !important; border-radius: 0 !important;
    align-items: center !important; justify-content: center !important;
}
.habit-path-picker-overlay.habit-path-picker-open { display: flex !important; visibility: visible !important; pointer-events: auto !important; }
#habit-path-picker-overlay.habit-path-picker-open > .habit-path-picker-card { display: flex !important; visibility: visible !important; pointer-events: auto !important; }
.habit-path-picker-open .habit-path-picker-overlay,
.habit-path-picker-open .gr-group.habit-path-picker-overlay {
    display: flex !important; visibility: visible !important; pointer-events: auto !important;
    position: relative !important; inset: auto !important; width: 100% !important;
    max-width: 100% !important; min-height: auto !important; height: auto !important;
    margin: 0 !important; padding: 0 !important; background: transparent !important;
    backdrop-filter: none !important; z-index: auto !important; overflow: visible !important;
    align-items: stretch !important; justify-content: flex-start !important;
}
.habit-path-picker-card {
    width: min(920px,96vw) !important; max-width: 920px !important; min-width: min(720px, 96vw) !important;
    max-height: 88vh !important; margin: 0 auto !important; padding: 18px 22px 20px !important;
    border-radius: 14px !important; background: var(--background-fill-primary, #262626) !important;
    border: 1px solid var(--border-color-primary, #444) !important;
    box-shadow: 0 24px 64px rgba(0,0,0,0.45) !important; box-sizing: border-box !important;
    overflow: hidden !important; display: flex !important; flex-direction: column !important;
    gap: 0.35rem !important; flex: 0 1 auto !important;
}
.habit-path-explorer, .habit-path-explorer.block, .habit-path-explorer > div,
.habit-path-explorer .wrap, .habit-path-explorer label + div,
.habit-path-explorer [role="tree"], .habit-path-explorer .file-wrap {
    width: 100% !important; min-width: min(680px, 92vw) !important; max-width: 100% !important; box-sizing: border-box !important;
}
.habit-path-explorer [role="treeitem"], .habit-path-explorer label, .habit-path-explorer span {
    white-space: nowrap !important; text-overflow: ellipsis !important; overflow: hidden !important; gap: 8px !important;
}
.habit-path-explorer [aria-label="expand directory"] {
    display: inline-flex !important; visibility: visible !important; opacity: 1 !important;
    width: 1rem !important; flex-shrink: 0 !important; overflow: visible !important;
}
.habit-path-explorer [aria-label="expand directory"].hidden { display: inline-flex !important; visibility: visible !important; }
.habit-path-preview textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important; white-space: nowrap !important; }
.habit-path-actions { margin-top: 0.35rem !important; width: 100% !important; }
"""


def _step_title_md(step_num: int) -> str:
    if not (1 <= step_num <= _NUM_STEPS):
        return ""
    label = STEP_LABELS[WORKFLOW_STEPS[step_num - 1]]
    return f"**Step {step_num} of {_NUM_STEPS}: {label}**"


def _workspace_bundle(root: str, step_num: int = 1) -> Tuple[str, str, str]:
    """Build workspace header HTML, stepper HTML, and step indicator markdown."""
    ctx = ProjectContext.load(root)
    header = ctx.workspace_header_html() if ctx else ""
    stepper = render_stepper_html(root, step_num) if root else ""
    indicator = _step_title_md(step_num)
    return header, stepper, indicator


def main(inbrowser: bool = False) -> None:
    """Main app runner: project home page + workflow workspace."""
    apply_gradio_patches()

    with gr.Blocks(title="HABIT \u2014 Habitat Analysis Toolkit", fill_width=True) as demo:
        project_root_state = gr.State(value="")
        active_step_state = gr.State(value=1)

        # ===== HOME PAGE =====
        with gr.Column(visible=True) as home_col:
            gr.HTML(
                "<div class='habit-home-title'>HABIT</div>"
                "<div class='habit-home-subtitle'>"
                "Habitat Analysis: Biomedical Imaging Toolkit"
                "</div>"
            )
            with gr.Accordion("How to use HABIT (click to expand)", open=False):
                gr.Markdown(
                    "1. **Create or open a project** — standard folders (`01_raw` … `05_ml`) are created automatically.\n"
                    "2. Work through the **6 steps** in the left sidebar (steps 4–6 are optional for some studies).\n"
                    "3. On each step click **Fill paths from project** (or paths auto-fill when you open the project).\n"
                    "4. Use **Quick presets** where available, then **Validate and run**.\n"
                    "5. Progress and completion status are saved in `habit_project.json`.\n\n"
                    f"*Documentation: [{DOCS_SITE_LABEL}]({DOCS_BASE_URL})*"
                )
            with gr.Row():
                with gr.Group(elem_classes=["habit-home-card"]):
                    gr.Markdown("### Open Existing Project")
                    with gr.Row():
                        project_root_input = gr.Textbox(
                            label="Project folder", scale=4,
                            placeholder="/path/to/project or click Browse",
                        )
                        home_browse_btn = gr.Button("Browse", scale=1)
                    open_btn = gr.Button("Open Project", variant="primary")
                    recent_projects = gr.Dropdown(
                        label="Recent projects", choices=[], value=None, interactive=True,
                    )

                with gr.Group(elem_classes=["habit-home-card"]):
                    gr.Markdown("### Create New Project")
                    project_name_input = gr.Textbox(
                        label="Project name", value="MyHabitatStudy",
                        placeholder="e.g. BrainTumorStudy",
                    )
                    with gr.Row():
                        new_project_root = gr.Textbox(
                            label="Project folder", scale=4,
                            placeholder="/path/to/new/project",
                        )
                        new_browse_btn = gr.Button("Browse", scale=1)
                    create_btn = gr.Button("Create Project", variant="primary")

            project_status = gr.Textbox(
                label="Status", interactive=False, placeholder="Ready."
            )

        path_picker = PathPickerRegistry()
        path_picker.add(home_browse_btn, project_root_input, pick="folder")
        path_picker.add(new_browse_btn, new_project_root, pick="folder")
        if should_use_web_path_picker():
            path_picker.build_overlay()

        # ===== WORKSPACE =====
        with gr.Column(visible=False) as workspace_col:
            workspace_header = gr.HTML("")

            with gr.Row():
                with gr.Column(scale=1, min_width=220):
                    stepper_html = gr.HTML("")
                    gr.Markdown("---")
                    step_buttons: List[Any] = []
                    for i in range(_NUM_STEPS):
                        btn = gr.Button(
                            f"{i + 1}. {STEP_LABELS[WORKFLOW_STEPS[i]]}",
                            size="sm", elem_classes=["habit-step-btn"],
                        )
                        step_buttons.append(btn)
                    gr.Markdown("---")
                    back_btn = gr.Button("\u2190 Back to Home", size="sm")

                with gr.Column(scale=4):
                    step_cols: List[Any] = []
                    with gr.Column(visible=True) as col_1:
                        render_dicom_sort_tab(
                            demo=demo, path_picker=path_picker,
                            project_root_state=project_root_state,
                        )
                    step_cols.append(col_1)
                    with gr.Column(visible=False) as col_2:
                        render_preprocess_tab(
                            demo=demo, path_picker=path_picker,
                            project_root_state=project_root_state,
                        )
                    step_cols.append(col_2)
                    with gr.Column(visible=False) as col_3:
                        render_habitat_tab(
                            demo=demo, path_picker=path_picker,
                            project_root_state=project_root_state,
                        )
                    step_cols.append(col_3)
                    with gr.Column(visible=False) as col_4:
                        render_extract_tab(
                            demo=demo, path_picker=path_picker,
                            project_root_state=project_root_state,
                        )
                    step_cols.append(col_4)
                    with gr.Column(visible=False) as col_5:
                        render_ml_tab(
                            demo=demo, path_picker=path_picker,
                            project_root_state=project_root_state,
                        )
                    step_cols.append(col_5)
                    with gr.Column(visible=False) as col_6:
                        render_compare_tab(
                            demo=demo, path_picker=path_picker,
                            project_root_state=project_root_state,
                        )
                    step_cols.append(col_6)

                    with gr.Row():
                        prev_step_btn = gr.Button("\u2190 Previous Step", size="sm")
                        step_indicator = gr.Markdown("")
                        next_step_btn = gr.Button("Next Step \u2192", size="sm", variant="primary")

        path_picker.finalize()
        _path_fill_outputs = all_path_components()

        def _refresh_recent() -> Any:
            items = list_recent_projects()
            choices = [(Path(p).name, p) for p in items]
            return gr.update(choices=choices, value=items[0] if items else None)

        def _enter_workspace(root: str, status_msg: str) -> Tuple[Any, ...]:
            """Shared handler after open/create project."""
            if not root:
                base = (
                    gr.update(visible=True), gr.update(visible=False),
                    status_msg, "", 1, "", "", "",
                )
                return base + fill_all_registered_paths("")

            meta = load_project(root)
            if meta:
                meta = migrate_project_meta(meta)
            header, stepper, indicator = _workspace_bundle(root, 1)
            path_fills = fill_all_registered_paths(root)
            return (
                gr.update(visible=False), gr.update(visible=True),
                status_msg, root, 1, header, stepper, indicator,
            ) + path_fills

        def _open_project(root: str) -> Tuple[Any, ...]:
            if not root or not str(root).strip():
                return _enter_workspace("", "Please select a project folder first.")
            root = normalize_project_root(str(root).strip())
            meta = load_project(root)
            if not meta:
                return _enter_workspace(
                    "",
                    f"habit_project.json not found in:\n{root}\n"
                    "Use 'Create New Project' to initialize it first.",
                )
            register_recent_project(root)
            name = meta.get("name", root)
            summary = get_progress_summary(root)
            status = f"Opened: {name} ({summary.get('completed', 0)}/{summary.get('total', 6)} steps completed)"
            return _enter_workspace(root, status)

        def _create_project(root: str, name: str) -> Tuple[Any, ...]:
            if not root or not root.strip():
                return _enter_workspace("", "Please specify a project folder path.")
            root = normalize_project_root(str(root).strip())
            try:
                meta = create_project(root, name or Path(root).name)
                register_recent_project(root)
                display_name = meta.get("name", root)
                return _enter_workspace(root, f"Created: {display_name}")
            except OSError as exc:
                return _enter_workspace("", f"Create failed: {exc}")

        def _switch_step(step_num: int, root: str) -> Tuple[Any, ...]:
            visibilities = tuple(
                gr.update(visible=(i + 1 == step_num)) for i in range(_NUM_STEPS)
            )
            ctx = ProjectContext.load(root) if root else None
            header = ctx.workspace_header_html() if ctx else ""
            stepper = render_stepper_html(root, step_num) if root else ""
            indicator = _step_title_md(step_num)
            return (step_num, header, stepper, indicator) + visibilities

        def _go_home() -> Tuple[Any, ...]:
            return (
                gr.update(visible=True), gr.update(visible=False), gr.update(),
            )

        _home_outputs = [
            home_col, workspace_col, project_status,
            project_root_state, active_step_state,
            workspace_header, stepper_html, step_indicator,
        ] + _path_fill_outputs

        open_btn.click(_open_project, [project_root_input], _home_outputs)
        create_btn.click(
            _create_project, [new_project_root, project_name_input], _home_outputs,
        )

        recent_projects.change(
            lambda p: p or "", [recent_projects], [project_root_input],
        )

        _step_nav_outputs = [
            active_step_state, workspace_header, stepper_html, step_indicator,
        ] + step_cols
        for i, btn in enumerate(step_buttons):
            btn.click(
                lambda root, idx=i + 1: _switch_step(idx, root),
                [project_root_state], _step_nav_outputs,
            )

        def _go_next(cur_step: int, root: str) -> Tuple[Any, ...]:
            return _switch_step(min(cur_step + 1, _NUM_STEPS), root)

        def _go_prev(cur_step: int, root: str) -> Tuple[Any, ...]:
            return _switch_step(max(cur_step - 1, 1), root)

        next_step_btn.click(
            _go_next, [active_step_state, project_root_state], _step_nav_outputs,
        )
        prev_step_btn.click(
            _go_prev, [active_step_state, project_root_state], _step_nav_outputs,
        )

        back_btn.click(_go_home, [], [home_col, workspace_col, project_status])

        demo.load(_refresh_recent, inputs=[], outputs=[recent_projects])
        demo.load(None, None, None, js=HABIT_PATH_PICKER_PORTAL_JS)

    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "8501"))
    bind_host = get_gradio_bind_host(server_name)
    browser_url = get_gradio_browser_url(server_name, server_port)
    open_browser: bool = inbrowser or os.environ.get("HABIT_GUI_INBROWSER", "").lower() in (
        "1", "true", "yes",
    )

    if should_use_web_path_picker():
        picker_mode = "web FileExplorer dialog (Docker/WSL)"
    else:
        picker_mode = "native tkinter"
    print(f"[HABIT] Path picker: {picker_mode}")

    if is_wsl():
        print(
            f"[HABIT] WSL detected: Gradio binds {bind_host}:{server_port}; "
            f"{get_wsl_browser_access_hint(server_port)}"
        )

    use_gradio_inbrowser = open_browser and not should_use_host_browser()
    if open_browser and should_use_host_browser():
        schedule_browser_open(browser_url, server_port=server_port)

    launch_kwargs: dict = {
        "server_name": bind_host,
        "server_port": server_port, "share": False,
        "inbrowser": use_gradio_inbrowser, "css": custom_css, "ssr_mode": False,
    }
    if should_use_web_path_picker():
        home_dir = os.path.expanduser("~")
        allowed = {"/", "/home", home_dir, "/mnt"}
        allowed.update(list_docker_browse_roots())
        if is_docker_runtime():
            allowed.update({"/data", "/config", "/output"})
        launch_kwargs["allowed_paths"] = sorted(allowed)

    ensure_localhost_no_proxy()

    try:
        import httpx as _httpx
        from gradio import networking as _gradio_networking

        _orig_url_ok = _gradio_networking.url_ok
        _orig_httpx_get = _httpx.get

        class _MockResp:
            is_success = True
            status_code = 200
            url = "http://127.0.0.1/startup-events"

        def _habit_url_ok(url: str, *args: Any, **kwargs: Any) -> bool:
            try:
                return _orig_url_ok(url, *args, **kwargs)
            except Exception:
                return True

        def _habit_httpx_get(url: str, *args: Any, **kwargs: Any) -> Any:
            try:
                return _orig_httpx_get(url, *args, **kwargs)
            except Exception:
                return _MockResp()

        _gradio_networking.url_ok = _habit_url_ok
        _httpx.get = _habit_httpx_get
    except Exception:
        pass

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    _env_inbrowser = os.environ.get("HABIT_GUI_INBROWSER", "1").lower() in ("1", "true", "yes")
    main(inbrowser=_env_inbrowser)
