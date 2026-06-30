# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Gradio path picker: native tkinter on desktop, centered FileExplorer dialog in WSL/Docker.

Doctors always see Windows-style paths (F:\\...) in text boxes after browsing.
"""

from __future__ import annotations

import os
from typing import Any, List, Literal, Optional, Tuple, TYPE_CHECKING

import gradio as gr

from habit.utils.browser_utils import is_wsl
from habit.utils.docker_path_utils import (
    docker_browse_root,
    is_docker_runtime,
    to_user_visible_path,
)
from habit.utils.gradio_patches import safe_listdir

if TYPE_CHECKING:
    pass

PickKind = Literal["file", "folder"]
_PickerEntry = Tuple[gr.Button, gr.Textbox, PickKind]

# Gradio 6 often ignores fixed-position CSS on Group; apply layout when Browse opens.
# Gradio 6 compiles the js= string via new AsyncFunction("return (" + code + ").apply(...)").
# Multiple top-level function declarations cause "Unexpected token 'function'" in that context.
# Wrap in an arrow function and export helpers to window so other JS callbacks can call them.
HABIT_PATH_PICKER_PORTAL_JS = """() => {
    function habitIsCloneOverlay(node) {
        return !!(node && node.classList && node.classList.contains("habit-path-picker-clone"));
    }
    function habitListPathPickerShells() {
        return Array.from(
            document.querySelectorAll("#habit-path-picker-overlay, .habit-path-picker-overlay")
        ).filter((node) => node.querySelector(".habit-path-explorer"));
    }
    function habitFindPathPickerOverlay() {
        const keeper = document.querySelector(
            "#habit-path-picker-overlay:not(.habit-path-picker-clone)"
        );
        if (keeper) {
            return keeper;
        }
        const live = habitListPathPickerShells().find((node) => !habitIsCloneOverlay(node));
        if (live) {
            return live;
        }
        const card = document.querySelector(".habit-path-picker-card:not(.habit-path-picker-clone .habit-path-picker-card)");
        if (card) {
            return card.closest("#habit-path-picker-overlay")
                || card.closest(".habit-path-picker-overlay")
                || card.closest(".gradio-group")
                || card.parentElement;
        }
        return null;
    }
    function habitHidePathPickerClone(node) {
        if (!node || node === habitFindPathPickerOverlay()) {
            return;
        }
        node.classList.add("habit-path-picker-clone");
        node.classList.remove("habit-path-picker-open");
        node.removeAttribute("id");
        node.style.setProperty("display", "none", "important");
        node.style.setProperty("pointer-events", "none", "important");
        node.style.setProperty("visibility", "hidden", "important");
        node.style.setProperty("position", "fixed", "important");
        node.style.setProperty("inset", "-9999px", "important");
        node.style.setProperty("z-index", "-1", "important");
    }
    function habitDedupePathPickerOverlay() {
        const shells = habitListPathPickerShells();
        if (shells.length <= 1) {
            const only = shells[0] || habitFindPathPickerOverlay();
            if (only && only.parentElement && only.parentElement !== document.body) {
                document.body.appendChild(only);
            }
            if (only) {
                only.classList.remove("habit-path-picker-clone");
                only.id = "habit-path-picker-overlay";
            }
            return only || null;
        }
        let keeper = shells.find((node) => node.parentElement === document.body && !habitIsCloneOverlay(node));
        if (!keeper) {
            keeper = shells.find((node) => !habitIsCloneOverlay(node)) || shells[0];
        }
        if (keeper.parentElement !== document.body) {
            document.body.appendChild(keeper);
        }
        keeper.classList.remove("habit-path-picker-clone");
        keeper.id = "habit-path-picker-overlay";
        shells.forEach((node) => {
            if (node !== keeper) {
                habitHidePathPickerClone(node);
            }
        });
        return keeper;
    }
    function habitResetNestedOverlayStyles(overlay) {
        if (!overlay) {
            return;
        }
        overlay.querySelectorAll(".habit-path-picker-overlay, .gr-group.habit-path-picker-overlay").forEach((el) => {
            if (el === overlay) {
                return;
            }
            [
                "display", "position", "inset", "width", "height", "min-height",
                "background", "z-index", "padding", "margin", "pointer-events", "visibility"
            ].forEach((prop) => el.style.removeProperty(prop));
        });
    }
    function habitPortalPathPicker() {
        habitDedupePathPickerOverlay();
    }
    function habitClosePathPickerDialog() {
        document.querySelectorAll(
            "#habit-path-picker-overlay, .habit-path-picker-overlay, .habit-path-picker-clone"
        ).forEach((overlay) => {
            overlay.classList.remove("habit-path-picker-open");
            if (habitIsCloneOverlay(overlay)) {
                return;
            }
            overlay.style.setProperty("display", "none", "important");
            overlay.style.setProperty("pointer-events", "none", "important");
            habitResetNestedOverlayStyles(overlay);
        });
        document.querySelectorAll(".habit-path-explorer").forEach((el) => {
            const block = el.closest(".block") || el.closest(".gradio-group") || el;
            if (block.closest(".habit-path-picker-clone")) {
                return;
            }
            block.style.removeProperty("display");
        });
    }
    window.habitFindPathPickerOverlay = habitFindPathPickerOverlay;
    window.habitPortalPathPicker = habitPortalPathPicker;
    window.habitClosePathPickerDialog = habitClosePathPickerDialog;
    window.habitDedupePathPickerOverlay = habitDedupePathPickerOverlay;
    if (!window.__habitPathPickerPortal) {
        window.__habitPathPickerPortal = true;
        new MutationObserver(habitPortalPathPicker).observe(
            document.documentElement,
            { childList: true, subtree: true }
        );
    }
    habitPortalPathPicker();
}"""

HABIT_PATH_PICKER_OPEN_JS = """() => {
    function habitLiftPathPickerCard(overlay) {
        if (!overlay) {
            return;
        }
        const card = overlay.querySelector(".habit-path-picker-card");
        if (!card) {
            return;
        }
        // Gradio nests a second .habit-path-picker-overlay (display:none) between shell and card.
        // Reparent the card onto the open shell so Playwright and users can see and click it.
        if (card.parentElement !== overlay) {
            overlay.appendChild(card);
        }
    }
    function habitStylePathPickerOverlay(overlay) {
        if (!overlay) {
            return;
        }
        habitLiftPathPickerCard(overlay);
        overlay.style.setProperty("position", "fixed", "important");
        overlay.style.setProperty("inset", "0", "important");
        overlay.style.setProperty("z-index", "99999", "important");
        overlay.style.setProperty("width", "100vw", "important");
        overlay.style.setProperty("height", "100vh", "important");
        overlay.style.setProperty("margin", "0", "important");
        overlay.style.setProperty("padding", "0", "important");
        overlay.style.setProperty("display", "flex", "important");
        overlay.style.setProperty("align-items", "center", "important");
        overlay.style.setProperty("justify-content", "center", "important");
        overlay.style.setProperty("background", "rgba(8, 8, 10, 0.82)", "important");
        overlay.style.setProperty("overflow-y", "auto", "important");
        overlay.classList.add("habit-path-picker-open");
        overlay.querySelectorAll(".habit-path-picker-overlay, .gr-group.habit-path-picker-overlay").forEach((el) => {
            if (el === overlay) {
                return;
            }
            el.style.setProperty("display", "flex", "important");
            el.style.setProperty("visibility", "visible", "important");
            el.style.setProperty("pointer-events", "auto", "important");
            el.style.setProperty("position", "relative", "important");
            el.style.setProperty("inset", "auto", "important");
            el.style.setProperty("width", "100%", "important");
            el.style.setProperty("height", "auto", "important");
            el.style.setProperty("min-height", "auto", "important");
            el.style.setProperty("background", "transparent", "important");
            el.style.setProperty("z-index", "auto", "important");
            el.style.setProperty("padding", "0", "important");
            el.style.setProperty("margin", "0", "important");
        });
        const card = overlay.querySelector(".habit-path-picker-card");
        if (card) {
            card.style.setProperty("width", "min(920px, 96vw)", "important");
            card.style.setProperty("max-width", "920px", "important");
            card.style.setProperty("min-width", "720px", "important");
            card.style.setProperty("margin", "0 auto", "important");
        }
        overlay.querySelectorAll(
            ".habit-path-explorer, .habit-path-explorer > div, .habit-path-explorer .wrap, .habit-path-explorer [role='tree']"
        ).forEach((el) => {
            el.style.setProperty("width", "100%", "important");
            el.style.setProperty("min-width", "680px", "important");
            el.style.setProperty("max-width", "100%", "important");
        });
    }
    function habitOpenPathPickerDialog() {
        let overlay = null;
        if (typeof window.habitDedupePathPickerOverlay === "function") {
            overlay = window.habitDedupePathPickerOverlay();
        } else if (typeof window.habitPortalPathPicker === "function") {
            window.habitPortalPathPicker();
            overlay = typeof window.habitFindPathPickerOverlay === "function"
                ? window.habitFindPathPickerOverlay()
                : document.getElementById("habit-path-picker-overlay");
        } else {
            overlay = document.getElementById("habit-path-picker-overlay");
        }
        if (!overlay) {
            return;
        }
        overlay.classList.remove("habit-path-picker-clone");
        overlay.style.removeProperty("pointer-events");
        overlay.style.removeProperty("visibility");
        if (overlay.parentElement && overlay.parentElement !== document.body) {
            document.body.appendChild(overlay);
        }
        habitStylePathPickerOverlay(overlay);
        document.querySelectorAll(".habit-path-explorer").forEach((el) => {
            if (!overlay.contains(el)) {
                const block = el.closest(".block") || el.closest(".gradio-group") || el;
                block.style.setProperty("display", "none", "important");
            }
        });
    }
    habitOpenPathPickerDialog();
}"""

HABIT_PATH_PICKER_CLOSE_JS = "() => { if (typeof window.habitClosePathPickerDialog === 'function') window.habitClosePathPickerDialog(); }"


def should_use_web_path_picker() -> bool:
    """
    Return True when Browse should use the in-browser FileExplorer dialog.

    Docker and WSL share the same picker: a centered ``gr.FileExplorer`` overlay
    inside the Gradio page. Native OS dialogs (PowerShell / tkinter) are reserved
    for bare-metal Linux, macOS, and native Windows installs where the GUI process
    can open a local chooser directly.

    Returns:
        bool: True on Docker and WSL; False on other desktop runtimes.
    """
    return is_docker_runtime() or is_wsl()


def web_browse_root() -> str:
    """
    Default starting directory for the in-browser path picker.

    Returns:
        str: ``/mnt`` when Docker or WSL (Windows drive mounts); else user home or ``/``.
    """
    if should_use_web_path_picker():
        return docker_browse_root()
    home = os.path.expanduser("~")
    if home and os.path.isdir(home):
        return _normalize_dir(home)
    return "/"


def _normalize_dir(path: str) -> str:
    """Normalize a directory path for stable comparisons and display."""
    cleaned = str(path or "").strip()
    if not cleaned:
        return cleaned
    return os.path.normpath(cleaned)


def _explorer_glob(pick: PickKind) -> str:
    """Return FileExplorer glob pattern for folder vs file selection."""
    if pick == "folder":
        # "**/" only matches trailing-slash paths in fnmatch; use "**" so drive/folder rows stay selectable.
        return "**"
    return "**/*"


def _extract_explorer_path(raw_value: Any, root_dir: Optional[str] = None) -> str:
    """
    Normalize FileExplorer output into a single runtime path string.

    Gradio FileExplorer emits nested segment lists such as ``["f", "work"]`` relative
    to ``root_dir``. Earlier versions kept only the last segment (``work``), which
    broke folder preview and Confirm.

    Args:
        raw_value: Value emitted by ``gr.FileExplorer`` (path, list, or nested list).
        root_dir: Explorer root; defaults to :func:`web_browse_root`.

    Returns:
        str: Absolute runtime path, or empty string when unset.
    """
    if raw_value is None:
        return ""
    root = _normalize_dir(root_dir or web_browse_root())

    def _deepest_segments(value: Any) -> List[str]:
        """Return the innermost selected path segment list from FileExplorer."""
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if os.path.isabs(text):
                return [text]
            return [part for part in text.replace("\\", "/").split("/") if part]
        if isinstance(value, list):
            if not value:
                return []
            if all(isinstance(item, str) for item in value):
                return [str(item).strip() for item in value if str(item).strip()]
            return _deepest_segments(value[-1])
        text = str(value).strip()
        return [text] if text else []

    segments = _deepest_segments(raw_value)
    if not segments:
        return ""
    if len(segments) == 1 and os.path.isabs(segments[0]):
        return _normalize_dir(segments[0])
    return os.path.normpath(os.path.join(root, *segments))


def _normalize_explorer_event(raw_value: Any, root_dir: Optional[str] = None) -> str:
    """
    Convert any FileExplorer event payload into an absolute runtime path.

    Gradio 6 emits different shapes depending on the event:
    - ``change`` / ``input``: nested segment lists or absolute path strings.
    - ``select``: ``{"index": [...], "value": "f/work", "selected": true}`` where
      ``value`` is slash-separated relative to ``root_dir``.

    Args:
        raw_value: Explorer component value or select-event payload.
        root_dir: Explorer root; defaults to :func:`web_browse_root`.

    Returns:
        str: Absolute runtime path, or empty string when unset.
    """
    if raw_value is None:
        return ""
    if hasattr(raw_value, "value") and not isinstance(raw_value, (str, list, dict)):
        raw_value = getattr(raw_value, "value", raw_value)
    if isinstance(raw_value, dict):
        candidate: Any = raw_value.get("value")
        if candidate is None:
            candidate = raw_value.get("path")
        if candidate is not None:
            raw_value = candidate
    return _extract_explorer_path(raw_value, root_dir=root_dir)


_UNREADABLE_DIR_HINT = " (folder exists but contents cannot be read; you may still select this path)"


def _strip_unreadable_hint(path: str) -> str:
    """Remove the unreadable-directory suffix from a preview string."""
    suffix = _UNREADABLE_DIR_HINT
    if path.endswith(suffix):
        return path[: -len(suffix)].rstrip()
    return path


def _format_folder_preview(raw_path: str) -> str:
    """
    Build the Selected path preview, including a hint for unreadable folders.

    Args:
        raw_path: Absolute runtime directory path.

    Returns:
        str: User-visible path, optionally with an unreadable-directory hint.
    """
    visible = to_user_visible_path(raw_path)
    if safe_listdir(raw_path) is None:
        return f"{visible}{_UNREADABLE_DIR_HINT}"
    return visible


def _resolve_selection(raw_path: str, pick: PickKind) -> str:
    """
    Convert an internal runtime path into a user-visible selection string.

    Args:
        raw_path: Absolute runtime path selected in the browser.
        pick: ``file`` or ``folder`` selection mode.

    Returns:
        str: Normalized user-visible path, or empty string when invalid.
    """
    raw = _normalize_dir(raw_path)
    if not raw:
        return ""
    if pick == "folder":
        if os.path.isfile(raw):
            raw = os.path.dirname(raw)
        if not raw:
            return ""
        try:
            is_directory = os.path.isdir(raw)
        except OSError:
            return ""
        if not is_directory:
            return ""
    elif pick == "file":
        if os.path.isdir(raw):
            return ""
        try:
            is_file = os.path.isfile(raw)
        except OSError:
            return ""
        if not is_file:
            return ""
    return to_user_visible_path(raw)


def attach_path_browse(
    browse_btn: gr.Button,
    target_box: gr.Textbox,
    explorer: Optional[Any],
    *,
    pick: PickKind = "folder",
    select_local_path_fn: Optional[Any] = None,
) -> None:
    """
    Wire a Browse button to tkinter on desktop.

    Args:
        browse_btn: ``Browse`` button component.
        target_box: Textbox to receive the selected path.
        explorer: Unused legacy parameter kept for backward compatibility.
        pick: ``file`` or ``folder``.
        select_local_path_fn: Callable ``(kind, title) -> path`` for native desktop.
    """
    if select_local_path_fn is None:
        from habit.gui.utils import select_local_path as select_local_path_fn

    title = "Select folder" if pick == "folder" else "Select file"

    def _native_pick() -> Any:
        chosen = select_local_path_fn(pick, title)
        return chosen if chosen else gr.update()

    browse_btn.click(_native_pick, outputs=target_box)


class PathPickerRegistry:
    """
    Collect Browse buttons for the GUI and wire them on :meth:`finalize`.

    Docker and WSL use one centered in-browser ``FileExplorer`` dialog.
    Other desktop runtimes use native tkinter dialogs.
    Call :meth:`build_overlay` before tabs so Gradio does not inject the explorer inline.
    """

    def __init__(self) -> None:
        self._entries: List[_PickerEntry] = []
        self._overlay_ready: bool = False
        self.path_panel: Any = None
        self.explorer: Any = None
        self.selection_preview: Any = None
        self.picker_title: Any = None
        self.confirm_btn: Any = None
        self.cancel_btn: Any = None
        self.active_index: Any = None
        self.active_pick: Any = None
        self.selected_runtime_path: Any = None
        self._dialog_pick: PickKind = "folder"

    def add(
        self,
        browse_btn: gr.Button,
        target_box: gr.Textbox,
        *,
        pick: PickKind = "folder",
    ) -> None:
        """Register one Browse button and its destination textbox."""
        self._entries.append((browse_btn, target_box, pick))

    def build_overlay(self) -> None:
        """
        Mount the path picker shell once at the top of Blocks.

        Must run before tabs so the dialog is not rendered beside Browse buttons.
        """
        if self._overlay_ready or not should_use_web_path_picker():
            return

        self.active_index = gr.State(-1)
        self.active_pick = gr.State("folder")
        self.selected_runtime_path = gr.State("")

        with gr.Group(
            visible=True,
            elem_id="habit-path-picker-overlay",
            elem_classes="habit-path-picker-overlay",
        ) as self.path_panel:
            with gr.Column(elem_classes="habit-path-picker-card"):
                self.picker_title = gr.Markdown("### Select path")
                self.explorer = gr.FileExplorer(
                    root_dir=web_browse_root(),
                    glob="**/*",
                    file_count="multiple",
                    label="Select path",
                    height=420,
                    min_width=720,
                    interactive=True,
                    elem_classes="habit-path-explorer",
                )
                self.selection_preview = gr.Textbox(
                    label="Selected path",
                    interactive=False,
                    elem_classes="habit-path-preview",
                )
                with gr.Row(elem_classes="habit-path-actions"):
                    self.confirm_btn = gr.Button("Confirm", variant="primary", scale=2)
                    self.cancel_btn = gr.Button("Cancel", scale=1)

        self._overlay_ready = True

    def finalize(self) -> None:
        """Wire all registered Browse buttons. Call once after all tabs are rendered."""
        if not self._entries:
            return
        if should_use_web_path_picker():
            if not self._overlay_ready:
                self.build_overlay()
            self._wire_web_dialog()
            return
        for browse_btn, target_box, pick in self._entries:
            attach_path_browse(browse_btn, target_box, None, pick=pick)

    def _wire_web_dialog(self) -> None:
        """Wire Browse / Confirm / Cancel for the pre-mounted overlay dialog."""
        if not self._overlay_ready:
            return

        target_boxes: List[gr.Textbox] = [entry[1] for entry in self._entries]
        explorer = self.explorer
        selection_preview = self.selection_preview
        picker_title = self.picker_title
        confirm_btn = self.confirm_btn
        cancel_btn = self.cancel_btn
        active_index = self.active_index
        active_pick = self.active_pick
        selected_runtime_path = self.selected_runtime_path
        browse_root = web_browse_root()

        def _preview_from_value(explorer_value: Any = None, pick: Any = "folder") -> Tuple[str, str]:
            # Tolerate empty inputs (Gradio 6 may fire with 0 args on load/hidden)
            if explorer_value is None:
                return "", ""
            if not pick:
                pick = "folder"
            runtime_path = _normalize_explorer_event(explorer_value, root_dir=browse_root)
            resolved = _resolve_selection(runtime_path, pick)
            if resolved and pick == "folder":
                preview = _format_folder_preview(runtime_path)
            else:
                preview = resolved
            return preview, runtime_path

        def _preview_from_select(evt: gr.SelectData) -> Tuple[str, str]:
            pick = self._dialog_pick
            runtime_path = _normalize_explorer_event(evt, root_dir=browse_root)
            resolved = _resolve_selection(runtime_path, pick)
            if resolved and pick == "folder":
                preview = _format_folder_preview(runtime_path)
            else:
                preview = resolved
            return preview, runtime_path

        def _open_dialog(index: int, pick: PickKind) -> Tuple[int, PickKind, Any, str, str, Any]:
            title = "Select folder" if pick == "folder" else "Select file"
            self._dialog_pick = pick
            file_count: Literal["single", "multiple"] = "single" if pick == "folder" else "multiple"
            return (
                index,
                pick,
                gr.update(
                    root_dir=browse_root,
                    glob=_explorer_glob(pick),
                    file_count=file_count,
                    interactive=True,
                    value=[],
                ),
                "",
                "",
                gr.update(value=f"### {title}"),
            )

        open_outputs = [
            active_index,
            active_pick,
            explorer,
            selection_preview,
            selected_runtime_path,
            picker_title,
        ]

        for index, (browse_btn, _, pick) in enumerate(self._entries):
            browse_btn.click(
                fn=lambda idx=index, pick_mode=pick: _open_dialog(idx, pick_mode),
                outputs=open_outputs,
                js=HABIT_PATH_PICKER_OPEN_JS,
            )

        explorer.change(
            _preview_from_value,
            inputs=[explorer, active_pick],
            outputs=[selection_preview, selected_runtime_path],
        )
        explorer.select(
            _preview_from_select,
            outputs=[selection_preview, selected_runtime_path],
        )
        explorer.input(
            _preview_from_value,
            inputs=[explorer, active_pick],
            outputs=[selection_preview, selected_runtime_path],
        )

        def _close_dialog() -> Tuple[Any, str, str, int]:
            return gr.update(value=[]), "", "", -1

        def _confirm_selection(
            index: int,
            preview: str,
            raw_value: Any,
            pick: PickKind,
            runtime_path: str,
        ) -> Tuple[Any, ...]:
            noop = gr.update()
            box_updates: List[Any] = [noop] * len(target_boxes)
            resolved = _strip_unreadable_hint((preview or "").strip())
            if not resolved:
                candidate = runtime_path or _normalize_explorer_event(raw_value, root_dir=browse_root)
                resolved = _resolve_selection(candidate, pick)
            if 0 <= index < len(target_boxes) and resolved:
                box_updates[index] = resolved
            return (
                *box_updates,
                gr.update(value=[]),
                "",
                "",
                -1,
            )

        confirm_btn.click(
            _confirm_selection,
            inputs=[active_index, selection_preview, explorer, active_pick, selected_runtime_path],
            outputs=[*target_boxes, explorer, selection_preview, selected_runtime_path, active_index],
            js=HABIT_PATH_PICKER_CLOSE_JS,
        )
        cancel_btn.click(
            _close_dialog,
            outputs=[explorer, selection_preview, selected_runtime_path, active_index],
            js=HABIT_PATH_PICKER_CLOSE_JS,
        )
