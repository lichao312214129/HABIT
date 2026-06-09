# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
Reusable Gradio editor for feature-preprocessing method pipelines.

Renders a visible ordered list with per-method Accordion parameter panels,
inline reorder buttons, and an add-method dropdown (not used for navigation).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gradio as gr

MAX_PREP_METHOD_SLOTS: int = 8

ALL_PREP_METHODS: List[str] = [
    "winsorize",
    "minmax",
    "zscore",
    "robust",
    "log",
    "binning",
    "variance_filter",
    "correlation_filter",
]

PREP_METHOD_LABELS: Dict[str, str] = {
    "winsorize": "Winsorize",
    "minmax": "Min-max scaling",
    "zscore": "Z-score",
    "robust": "Robust scaling",
    "log": "Log transform",
    "binning": "Binning",
    "variance_filter": "Variance filter",
    "correlation_filter": "Correlation filter",
}

METHOD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "winsorize": {
        "method": "winsorize",
        "winsor_limits": [0.05, 0.05],
        "global_normalize": False,
    },
    "minmax": {"method": "minmax", "global_normalize": False},
    "zscore": {"method": "zscore", "global_normalize": False},
    "robust": {"method": "robust", "global_normalize": False},
    "log": {"method": "log", "global_normalize": False},
    "binning": {
        "method": "binning",
        "n_bins": 10,
        "bin_strategy": "quantile",
        "global_normalize": False,
    },
    "variance_filter": {
        "method": "variance_filter",
        "variance_threshold": 0.01,
        "global_normalize": False,
    },
    "correlation_filter": {
        "method": "correlation_filter",
        "corr_threshold": 0.9,
        "corr_method": "spearman",
        "global_normalize": False,
    },
}

DROPPING_PREP_METHODS: frozenset[str] = frozenset({"variance_filter", "correlation_filter"})


def default_method_entry(method: str) -> Dict[str, Any]:
    """
    Build a fresh default config dict for one preprocessing method.

    Args:
        method: Canonical method key.

    Returns:
        Dict[str, Any]: Deep-copied default parameter mapping.
    """
    if method not in METHOD_DEFAULTS:
        raise ValueError(f"Unknown preprocessing method: {method}")
    return copy.deepcopy(METHOD_DEFAULTS[method])


def normalize_methods_list(raw: Any) -> List[Dict[str, Any]]:
    """
    Normalize YAML-loaded or GUI state into a list of method config dicts.

    Args:
        raw: ``{"methods": [...]}`` mapping, bare list, or None.

    Returns:
        List[Dict[str, Any]]: Sanitized method entries (truncated to max slots).
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        items = raw.get("methods", []) or []
    elif isinstance(raw, list):
        items = raw
    else:
        return []
    methods: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        method_name = str(item.get("method", "")).strip()
        if method_name not in ALL_PREP_METHODS:
            continue
        entry = default_method_entry(method_name)
        entry.update(item)
        entry["method"] = method_name
        methods.append(entry)
        if len(methods) >= MAX_PREP_METHOD_SLOTS:
            break
    return methods


def methods_to_prep_config(methods: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Serialize GUI method list to FeatureConstruction preprocessing block.

    Args:
        methods: Ordered method config dicts from editor state.

    Returns:
        Optional[Dict[str, Any]]: ``{"methods": [...]}`` or None when empty.
    """
    cleaned: List[Dict[str, Any]] = []
    for entry in methods or []:
        method_name = str(entry.get("method", "")).strip()
        if method_name not in ALL_PREP_METHODS:
            continue
        cleaned.append(_strip_none_values(copy.deepcopy(entry)))
    if not cleaned:
        return None
    return {"methods": cleaned}


def _strip_none_values(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keys whose values are None so YAML matches schema defaults."""
    return {key: value for key, value in entry.items() if value is not None}


def _winsor_text(entry: Dict[str, Any]) -> str:
    """Format winsor_limits as comma-separated text for the GUI."""
    limits = entry.get("winsor_limits")
    if not limits or not isinstance(limits, (list, tuple)) or len(limits) < 2:
        return "0.05, 0.05"
    return f"{limits[0]}, {limits[1]}"


def _parse_winsor_text(text: str) -> List[float]:
    """Parse comma-separated winsor limits from the GUI textbox."""
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) < 2:
        return [0.05, 0.05]
    return [float(parts[0]), float(parts[1])]


def _move_method(methods: List[Dict[str, Any]], index: int, direction: int) -> List[Dict[str, Any]]:
    """
    Move one method up (-1) or down (+1) in the pipeline order.

    Args:
        methods: Current ordered method list.
        index: Index of the method to move.
        direction: -1 for up, +1 for down.

    Returns:
        List[Dict[str, Any]]: Reordered list (unchanged when move is invalid).
    """
    methods = list(methods or [])
    if index < 0 or index >= len(methods):
        return methods
    new_index = index + direction
    if 0 <= new_index < len(methods):
        methods[index], methods[new_index] = methods[new_index], methods[index]
    return methods


def _method_panel_visibility(method_name: str) -> Tuple[bool, bool, bool, bool]:
    """
    Return visibility flags for method-specific parameter rows.

    Args:
        method_name: Canonical preprocessing method key.

    Returns:
        Tuple[bool, bool, bool, bool]: winsor, binning, variance, correlation panels.
    """
    return (
        method_name == "winsorize",
        method_name == "binning",
        method_name == "variance_filter",
        method_name == "correlation_filter",
    )


def _refresh_slot_updates(
    index: int,
    methods: List[Dict[str, Any]],
) -> Tuple[Any, ...]:
    """
    Build gr.update payloads for one editor slot.

    Args:
        index: Zero-based slot index.
        methods: Current pipeline method list held in gr.State.

    Returns:
        Tuple[Any, ...]: Updates for row, accordion, labels, and parameter widgets.
    """
    if index < len(methods):
        entry = methods[index]
        method_name = str(entry.get("method", ""))
        winsor_vis, bin_vis, var_vis, corr_vis = _method_panel_visibility(method_name)
        label = PREP_METHOD_LABELS.get(method_name, method_name)
        return (
            gr.update(visible=True),
            gr.update(visible=True, label=f"{index + 1}. {label}"),
            gr.update(value=index + 1),
            gr.update(value=label),
            gr.update(value=bool(entry.get("global_normalize", False))),
            gr.update(value=_winsor_text(entry), visible=winsor_vis),
            gr.update(value=int(entry.get("n_bins") or 10), visible=bin_vis),
            gr.update(
                value=str(entry.get("bin_strategy") or "quantile"),
                visible=bin_vis,
            ),
            gr.update(value=float(entry.get("variance_threshold") or 0.01), visible=var_vis),
            gr.update(value=float(entry.get("corr_threshold") or 0.9), visible=corr_vis),
            gr.update(value=str(entry.get("corr_method") or "spearman"), visible=corr_vis),
        )
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=None),
        gr.update(value=""),
        gr.update(value=False),
        gr.update(value="0.05, 0.05", visible=False),
        gr.update(value=10, visible=False),
        gr.update(value="quantile", visible=False),
        gr.update(value=0.01, visible=False),
        gr.update(value=0.9, visible=False),
        gr.update(value="spearman", visible=False),
    )


def refresh_all_slot_updates(methods: List[Dict[str, Any]]) -> Tuple[Any, ...]:
    """
    Build gr.update payloads for every slot in the editor.

    Args:
        methods: Current pipeline method list.

    Returns:
        Tuple[Any, ...]: Flattened updates for all slots (11 widgets per slot).
    """
    chunks: List[Tuple[Any, ...]] = [
        _refresh_slot_updates(slot_idx, methods) for slot_idx in range(MAX_PREP_METHOD_SLOTS)
    ]
    return tuple(item for chunk in chunks for item in chunk)


@dataclass
class PrepMethodsEditor:
    """
    Gradio widget bundle for one preprocessing method pipeline editor.

    Attributes:
        state: gr.State holding List[Dict[str, Any]] method configs.
        allowed_methods: Method keys allowed in the add-method dropdown.
        slot_rows: Per-slot gr.Row containers.
        slot_accordions: Per-slot parameter accordions.
    """

    state: Any
    allowed_methods: List[str]
    add_dropdown: Any = None
    add_btn: Any = None
    slot_rows: List[Any] = field(default_factory=list)
    slot_accordions: List[Any] = field(default_factory=list)
    slot_orders: List[Any] = field(default_factory=list)
    slot_labels: List[Any] = field(default_factory=list)
    slot_global_norms: List[Any] = field(default_factory=list)
    slot_winsor: List[Any] = field(default_factory=list)
    slot_n_bins: List[Any] = field(default_factory=list)
    slot_bin_strategy: List[Any] = field(default_factory=list)
    slot_var_thresh: List[Any] = field(default_factory=list)
    slot_corr_thresh: List[Any] = field(default_factory=list)
    slot_corr_method: List[Any] = field(default_factory=list)
    slot_up_btns: List[Any] = field(default_factory=list)
    slot_down_btns: List[Any] = field(default_factory=list)
    slot_remove_btns: List[Any] = field(default_factory=list)

    def refresh_outputs(self) -> List[Any]:
        """Return all components updated when the pipeline state changes."""
        outputs: List[Any] = []
        for slot_idx in range(MAX_PREP_METHOD_SLOTS):
            outputs.extend([
                self.slot_rows[slot_idx],
                self.slot_accordions[slot_idx],
                self.slot_orders[slot_idx],
                self.slot_labels[slot_idx],
                self.slot_global_norms[slot_idx],
                self.slot_winsor[slot_idx],
                self.slot_n_bins[slot_idx],
                self.slot_bin_strategy[slot_idx],
                self.slot_var_thresh[slot_idx],
                self.slot_corr_thresh[slot_idx],
                self.slot_corr_method[slot_idx],
            ])
        return outputs

    def refresh_from_methods(self, methods: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        """
        Produce UI updates from a method list without mutating state.

        Args:
            methods: Pipeline method configs.

        Returns:
            Tuple[Any, ...]: gr.update values for ``refresh_outputs()`` components.
        """
        return refresh_all_slot_updates(methods)

    def _apply_slot_param(
        self,
        methods: List[Dict[str, Any]],
        slot_idx: int,
        field_name: str,
        value: Any,
    ) -> List[Dict[str, Any]]:
        """Update one parameter on a single method entry."""
        methods = list(methods or [])
        if slot_idx < 0 or slot_idx >= len(methods):
            return methods
        methods[slot_idx] = copy.deepcopy(methods[slot_idx])
        methods[slot_idx][field_name] = value
        return methods

    def _read_slot_params(
        self,
        methods: List[Dict[str, Any]],
        slot_idx: int,
        global_norm: bool,
        winsor_text: str,
        n_bins: float,
        bin_strategy: str,
        var_thresh: float,
        corr_thresh: float,
        corr_method: str,
    ) -> List[Dict[str, Any]]:
        """Merge current widget values into one slot before a structural change."""
        methods = list(methods or [])
        if slot_idx < 0 or slot_idx >= len(methods):
            return methods
        entry = copy.deepcopy(methods[slot_idx])
        method_name = str(entry.get("method", ""))
        entry["global_normalize"] = bool(global_norm)
        if method_name == "winsorize":
            entry["winsor_limits"] = _parse_winsor_text(winsor_text)
        elif method_name == "binning":
            entry["n_bins"] = int(n_bins)
            entry["bin_strategy"] = bin_strategy
        elif method_name == "variance_filter":
            entry["variance_threshold"] = float(var_thresh)
        elif method_name == "correlation_filter":
            entry["corr_threshold"] = float(corr_thresh)
            entry["corr_method"] = corr_method
        methods[slot_idx] = entry
        return methods

    def wire_handlers(self) -> None:
        """Attach Gradio event handlers for add, remove, reorder, and param edits."""
        refresh_outputs = self.refresh_outputs()

        def _emit(methods: List[Dict[str, Any]]) -> Tuple[Any, ...]:
            return self.refresh_from_methods(methods)

        def on_add(methods: List[Dict[str, Any]], new_method: str) -> Tuple[Any, ...]:
            methods = list(methods or [])
            if not new_method or new_method not in self.allowed_methods:
                return _emit(methods)
            if len(methods) >= MAX_PREP_METHOD_SLOTS:
                return _emit(methods)
            methods.append(default_method_entry(new_method))
            return (methods, *_emit(methods))

        self.add_btn.click(
            on_add,
            inputs=[self.state, self.add_dropdown],
            outputs=[self.state, *refresh_outputs],
        )

        for slot_idx in range(MAX_PREP_METHOD_SLOTS):
            slot_inputs = [
                self.state,
                self.slot_global_norms[slot_idx],
                self.slot_winsor[slot_idx],
                self.slot_n_bins[slot_idx],
                self.slot_bin_strategy[slot_idx],
                self.slot_var_thresh[slot_idx],
                self.slot_corr_thresh[slot_idx],
                self.slot_corr_method[slot_idx],
            ]

            def on_move(
                methods: List[Dict[str, Any]],
                global_norm: bool,
                winsor_text: str,
                n_bins: float,
                bin_strategy: str,
                var_thresh: float,
                corr_thresh: float,
                corr_method: str,
                direction: int,
                slot_index: int = slot_idx,
            ) -> Tuple[Any, ...]:
                methods = self._read_slot_params(
                    methods,
                    slot_index,
                    global_norm,
                    winsor_text,
                    n_bins,
                    bin_strategy,
                    var_thresh,
                    corr_thresh,
                    corr_method,
                )
                methods = _move_method(methods, slot_index, direction)
                return (methods, *_emit(methods))

            def on_remove(
                methods: List[Dict[str, Any]],
                global_norm: bool,
                winsor_text: str,
                n_bins: float,
                bin_strategy: str,
                var_thresh: float,
                corr_thresh: float,
                corr_method: str,
                slot_index: int = slot_idx,
            ) -> Tuple[Any, ...]:
                methods = self._read_slot_params(
                    methods,
                    slot_index,
                    global_norm,
                    winsor_text,
                    n_bins,
                    bin_strategy,
                    var_thresh,
                    corr_thresh,
                    corr_method,
                )
                if 0 <= slot_index < len(methods):
                    methods.pop(slot_index)
                return (methods, *_emit(methods))

            def on_param_change(
                methods: List[Dict[str, Any]],
                global_norm: bool,
                winsor_text: str,
                n_bins: float,
                bin_strategy: str,
                var_thresh: float,
                corr_thresh: float,
                corr_method: str,
                slot_index: int = slot_idx,
            ) -> Tuple[Any, ...]:
                methods = self._read_slot_params(
                    methods,
                    slot_index,
                    global_norm,
                    winsor_text,
                    n_bins,
                    bin_strategy,
                    var_thresh,
                    corr_thresh,
                    corr_method,
                )
                return (methods, *_emit(methods))

            self.slot_up_btns[slot_idx].click(
                lambda *args, sk=slot_idx: on_move(*args, direction=-1, slot_index=sk),
                inputs=slot_inputs,
                outputs=[self.state, *refresh_outputs],
            )
            self.slot_down_btns[slot_idx].click(
                lambda *args, sk=slot_idx: on_move(*args, direction=1, slot_index=sk),
                inputs=slot_inputs,
                outputs=[self.state, *refresh_outputs],
            )
            self.slot_remove_btns[slot_idx].click(
                lambda *args, sk=slot_idx: on_remove(*args, slot_index=sk),
                inputs=slot_inputs,
                outputs=[self.state, *refresh_outputs],
            )

            for widget in (
                self.slot_global_norms[slot_idx],
                self.slot_winsor[slot_idx],
                self.slot_n_bins[slot_idx],
                self.slot_bin_strategy[slot_idx],
                self.slot_var_thresh[slot_idx],
                self.slot_corr_thresh[slot_idx],
                self.slot_corr_method[slot_idx],
            ):
                widget.change(
                    lambda *args, sk=slot_idx: on_param_change(*args, slot_index=sk),
                    inputs=slot_inputs,
                    outputs=[self.state, *refresh_outputs],
                )


def render_prep_methods_editor(
    section_title: str,
    default_methods: Sequence[Dict[str, Any]],
    allowed_methods: Optional[Sequence[str]] = None,
) -> PrepMethodsEditor:
    """
    Render a visible preprocessing pipeline editor inside the current Gradio context.

    Args:
        section_title: Markdown heading shown above the editor.
        default_methods: Initial ordered method list.
        allowed_methods: Keys shown in the add-method dropdown; defaults to all methods.

    Returns:
        PrepMethodsEditor: Widget bundle with state and event wiring helpers.
    """
    allowed = list(allowed_methods or ALL_PREP_METHODS)
    initial_methods = normalize_methods_list(list(default_methods))

    editor = PrepMethodsEditor(
        state=gr.State(value=initial_methods),
        allowed_methods=allowed,
    )

    gr.Markdown(section_title)
    gr.Markdown(
        "Methods run top-to-bottom. Use **⬆ / ⬇** to reorder, **✕** to remove, "
        "and expand each row to edit parameters."
    )
    with gr.Row():
        gr.Markdown("**#**", scale=0)
        gr.Markdown("**Method**", scale=4)
        gr.Markdown("**Actions**", scale=0)

    initial_refresh = refresh_all_slot_updates(initial_methods)
    refresh_iter = iter(initial_refresh)

    for slot_idx in range(MAX_PREP_METHOD_SLOTS):
        slot_updates = [
            next(refresh_iter),
            next(refresh_iter),
            next(refresh_iter),
            next(refresh_iter),
            next(refresh_iter),
            next(refresh_iter),
            next(refresh_iter),
            next(refresh_iter),
            next(refresh_iter),
            next(refresh_iter),
            next(refresh_iter),
        ]
        row_visible = slot_updates[0].get("visible", False) if isinstance(slot_updates[0], dict) else False
        acc_visible = slot_updates[1].get("visible", False) if isinstance(slot_updates[1], dict) else False

        with gr.Row(visible=row_visible) as slot_row:
            slot_order = gr.Number(
                label="",
                value=slot_updates[2].get("value") if isinstance(slot_updates[2], dict) else None,
                interactive=False,
                precision=0,
                scale=0,
            )
            slot_label = gr.Textbox(
                label="",
                value=slot_updates[3].get("value", "") if isinstance(slot_updates[3], dict) else "",
                interactive=False,
                scale=4,
            )
            with gr.Row(scale=0):
                up_btn = gr.Button("⬆")
                down_btn = gr.Button("⬇")
                remove_btn = gr.Button("✕")

        with gr.Accordion(
            label=slot_updates[1].get("label", f"{slot_idx + 1}. Method") if isinstance(slot_updates[1], dict) else f"{slot_idx + 1}. Method",
            open=False,
            visible=acc_visible,
        ) as slot_acc:
            global_norm = gr.Checkbox(
                label="global_normalize",
                value=slot_updates[4].get("value", False) if isinstance(slot_updates[4], dict) else False,
            )
            winsor = gr.Textbox(
                label="winsor_limits (lower, upper)",
                value=slot_updates[5].get("value", "0.05, 0.05") if isinstance(slot_updates[5], dict) else "0.05, 0.05",
                visible=slot_updates[5].get("visible", False) if isinstance(slot_updates[5], dict) else False,
            )
            with gr.Row():
                n_bins = gr.Number(
                    label="n_bins",
                    value=slot_updates[6].get("value", 10) if isinstance(slot_updates[6], dict) else 10,
                    precision=0,
                    visible=slot_updates[6].get("visible", False) if isinstance(slot_updates[6], dict) else False,
                )
                bin_strategy = gr.Dropdown(
                    label="bin_strategy",
                    choices=["uniform", "quantile", "kmeans"],
                    value=slot_updates[7].get("value", "quantile") if isinstance(slot_updates[7], dict) else "quantile",
                    visible=slot_updates[7].get("visible", False) if isinstance(slot_updates[7], dict) else False,
                )
            var_thresh = gr.Number(
                label="variance_threshold",
                value=slot_updates[8].get("value", 0.01) if isinstance(slot_updates[8], dict) else 0.01,
                visible=slot_updates[8].get("visible", False) if isinstance(slot_updates[8], dict) else False,
            )
            with gr.Row():
                corr_thresh = gr.Number(
                    label="corr_threshold",
                    value=slot_updates[9].get("value", 0.9) if isinstance(slot_updates[9], dict) else 0.9,
                    visible=slot_updates[9].get("visible", False) if isinstance(slot_updates[9], dict) else False,
                )
                corr_method = gr.Dropdown(
                    label="corr_method",
                    choices=["pearson", "spearman", "kendall"],
                    value=slot_updates[10].get("value", "spearman") if isinstance(slot_updates[10], dict) else "spearman",
                    visible=slot_updates[10].get("visible", False) if isinstance(slot_updates[10], dict) else False,
                )

        editor.slot_rows.append(slot_row)
        editor.slot_accordions.append(slot_acc)
        editor.slot_orders.append(slot_order)
        editor.slot_labels.append(slot_label)
        editor.slot_global_norms.append(global_norm)
        editor.slot_winsor.append(winsor)
        editor.slot_n_bins.append(n_bins)
        editor.slot_bin_strategy.append(bin_strategy)
        editor.slot_var_thresh.append(var_thresh)
        editor.slot_corr_thresh.append(corr_thresh)
        editor.slot_corr_method.append(corr_method)
        editor.slot_up_btns.append(up_btn)
        editor.slot_down_btns.append(down_btn)
        editor.slot_remove_btns.append(remove_btn)

    add_choices = [(PREP_METHOD_LABELS.get(m, m), m) for m in allowed]
    with gr.Row():
        editor.add_dropdown = gr.Dropdown(
            label="Add method",
            choices=add_choices,
            value=allowed[0] if allowed else None,
            scale=4,
        )
        editor.add_btn = gr.Button("Add to pipeline", scale=1)

    editor.wire_handlers()
    return editor
