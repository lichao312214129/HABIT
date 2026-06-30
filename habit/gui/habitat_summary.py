# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Human-readable labels and review text for the habitat wizard."""

from __future__ import annotations

from typing import List

PREP_PRESET_LABELS: dict[str, str] = {
    "minimal": "Minimal (min-max only)",
    "standard": "Standard (recommended)",
    "full": "Full filtering (radiomics-heavy)",
}

SELECTION_METHOD_LABELS: dict[str, str] = {
    "elbow": "Elbow",
    "silhouette": "Silhouette",
    "bic": "BIC",
    "aic": "AIC",
    "davies_bouldin": "Davies-Bouldin",
    "calinski_harabasz": "Calinski-Harabasz",
    "inertia": "Inertia",
    "kneedle": "Kneedle",
    "gap": "Gap statistic",
}

_CLUSTERING_MODE_LABELS: dict[str, str] = {
    "two_step": "Two-step (supervoxel → habitat)",
    "one_step": "One-step (voxel → habitat)",
    "direct_pooling": "Direct pooling (cohort-level)",
}

_RUN_MODE_LABELS: dict[str, str] = {
    "train": "Train",
    "predict": "Predict",
}


def wizard_step_title(step: int) -> str:
    """
    Return the Markdown title for the active wizard step.

    Args:
        step: Current wizard step index (1–5).

    Returns:
        str: Markdown heading for ``gr.Markdown``.
    """
    titles = {
        1: "**Step 1/4 — Choose template**",
        2: "**Step 2/4 — Data and sequences**",
        3: "**Step 3/4 — Parameters**",
        4: "**Step 4/4 — Review and run**",
        5: "**Step 5 — Results**",
    }
    return titles.get(int(step), titles[1])


def build_habitat_review_summary(
    template_name: str,
    data_dir: str,
    out_dir: str,
    n_subjects: int,
    n_ok: int,
    modalities: List[str],
    clustering_mode: str,
    sv_n_clusters: int,
    habitat_auto: bool,
    habitat_n: int,
    habitat_selection: str,
    feature_type: str,
    prep_preset: str,
    run_mode: str,
) -> str:
    """
    Build a Chinese review summary shown before running habitat analysis.

    Args:
        template_name: Human-readable template name.
        data_dir: Input data directory.
        out_dir: Output directory.
        n_subjects: Total subjects detected.
        n_ok: Subjects passing layout checks.
        modalities: Selected modality names.
        clustering_mode: ``one_step``, ``two_step``, or ``direct_pooling``.
        sv_n_clusters: Supervoxel partition count (two-step).
        habitat_auto: Whether habitat count is auto-selected.
        habitat_n: Fixed habitat count when not auto.
        habitat_selection: Cluster selection method key.
        feature_type: Voxel feature type key.
        prep_preset: Preprocessing preset key.
        run_mode: ``train`` or ``predict``.

    Returns:
        str: Multi-line review text in Chinese.
    """
    mod_text = ", ".join(modalities) if modalities else "—"
    cluster_label = _CLUSTERING_MODE_LABELS.get(clustering_mode, clustering_mode)
    feature_label = feature_type
    prep_label = PREP_PRESET_LABELS.get(prep_preset, prep_preset)
    sel_label = SELECTION_METHOD_LABELS.get(habitat_selection, habitat_selection)
    run_label = _RUN_MODE_LABELS.get(run_mode, run_mode)
    habitat_count = "自动选择" if habitat_auto else str(habitat_n)
    lines = [
        f"模板: {template_name}",
        f"运行模式: {run_label}",
        f"数据目录: {data_dir or '—'}",
        f"输出目录: {out_dir or '—'}",
        f"受试者: {n_ok}/{n_subjects} 通过数据检查",
        f"序列: {mod_text}",
        f"聚类模式: {cluster_label}",
    ]
    if clustering_mode == "two_step":
        lines.append(f"超体素分区数: {sv_n_clusters}")
    lines.extend(
        [
            f"生境数量: {habitat_count}",
            f"生境选择方法: {sel_label}",
            f"体素特征: {feature_label}",
            f"预处理预设: {prep_label}",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "PREP_PRESET_LABELS",
    "SELECTION_METHOD_LABELS",
    "wizard_step_title",
    "build_habitat_review_summary",
]
