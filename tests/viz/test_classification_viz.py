# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Unit tests for binary-classification figures and ML reporting wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure

from habit.api.exceptions import HABITAPIError
from habit.recipes.ml_reporting import (
    visualization_enabled,
    write_classification_figures,
    write_ml_figures_from_config,
)
from habit.viz import (
    net_benefit,
    plot_calibration,
    plot_confusion_matrix,
    plot_decision_curve,
    plot_precision_recall,
    plot_roc,
)

pytestmark = pytest.mark.unit


def _binary_scores(n: int = 80, seed: int = 0):
    """Synthetic separable binary labels and probabilities."""
    rng = np.random.RandomState(seed)
    scores = rng.rand(n)
    y_true = (scores + 0.15 * rng.randn(n) > 0.5).astype(int)
    if y_true.min() == y_true.max():
        y_true[0] = 1 - y_true[0]
    y_prob = np.clip(scores, 0.01, 0.99)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_prob, y_pred


@pytest.mark.parametrize(
    "plotter",
    [plot_roc, plot_precision_recall, plot_calibration, plot_decision_curve],
)
def test_curve_plotters_return_figure(plotter) -> None:
    """Each curve plotter returns a live matplotlib Figure."""
    y_true, y_prob, _ = _binary_scores()
    fig = plotter(y_true, y_prob, title="Unit Test")
    assert isinstance(fig, Figure)
    # All drawn text must stay ASCII (journal-safe).
    for text in fig.axes[0].texts + [fig.axes[0].title]:
        assert str(text.get_text()).isascii()


def test_plot_confusion_matrix_returns_figure() -> None:
    """Confusion matrix accepts hard labels."""
    y_true, _, y_pred = _binary_scores()
    fig = plot_confusion_matrix(y_true, y_pred, title="Confusion")
    assert isinstance(fig, Figure)


def test_plot_confusion_matrix_string_ticks_keep_integer_counts() -> None:
    """Display names like ``(\"0\", \"1\")`` must not zero-out integer labels."""
    y_true = np.array([0, 0, 1, 1, 1], dtype=int)
    y_pred = np.array([0, 1, 1, 1, 0], dtype=int)
    fig = plot_confusion_matrix(
        y_true, y_pred, title="Confusion", class_names=("0", "1")
    )
    image = fig.axes[0].images[0]
    counts = np.asarray(image.get_array(), dtype=np.float64)
    assert counts.sum() == float(y_true.size)
    assert counts[0, 0] == 1.0
    assert counts[1, 1] == 2.0
    assert image.get_clim()[0] == 0.0
    ticks = [text.get_text() for text in fig.axes[0].get_xticklabels()]
    assert ticks == ["0", "1"]


def test_plot_roc_rejects_length_mismatch() -> None:
    """Mismatched arrays raise a clear API error."""
    with pytest.raises(HABITAPIError, match="same length"):
        plot_roc(np.array([0, 1]), np.array([0.1]))


def test_net_benefit_treat_none_is_zero() -> None:
    """Treat-none strategy has zero net benefit at every threshold."""
    y_true = np.array([0, 1, 0, 1], dtype=float)
    assert net_benefit(y_true, np.zeros_like(y_true), 0.3) == 0.0


def test_write_classification_figures_writes_requested_files(tmp_path: Path) -> None:
    """Reporting helper persists only the requested plot types."""
    y_true, y_prob, y_pred = _binary_scores()
    paths = write_classification_figures(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        destination=tmp_path,
        plot_types=["roc", "dca", "calibration"],
        image_format="png",
        dpi=72,
        prefix="test_",
    )
    names = sorted(path.name for path in paths)
    assert names == [
        "test_calibration_curve.png",
        "test_decision_curve.png",
        "test_roc_curve.png",
    ]
    for path in paths:
        assert path.is_file() and path.stat().st_size > 0


def test_write_classification_figures_skips_when_empty_types(tmp_path: Path) -> None:
    """Empty plot_types writes nothing."""
    y_true, y_prob, y_pred = _binary_scores()
    paths = write_classification_figures(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        destination=tmp_path,
        plot_types=[],
    )
    assert paths == []
    assert list(tmp_path.iterdir()) == []


def test_visualization_enabled_respects_flags() -> None:
    """Both top-level and nested enabled switches gate reporting."""
    assert visualization_enabled(is_visualize=False, visualization=None) is False
    assert visualization_enabled(is_visualize=True, visualization=None) is True
    assert (
        visualization_enabled(
            is_visualize=True, visualization=SimpleNamespace(enabled=False)
        )
        is False
    )
    assert (
        visualization_enabled(
            is_visualize=True, visualization={"enabled": True}
        )
        is True
    )


def test_write_ml_figures_from_config_disabled(tmp_path: Path) -> None:
    """Config with is_visualize=false writes no figures."""
    from habit.spec import MLSpec, Spec
    from habit.datasets import make_synthetic_feature_table
    from habit.recipes.modeling import train_model

    table = make_synthetic_feature_table(n_rows=60, n_features=6, rng=0)
    spec = MLSpec(
        name="viz_off",
        classifier=Spec("LogisticRegression", {"max_iter": 300}),
        metrics=(Spec("accuracy"), Spec("auc")),
    )
    result = train_model(table, spec, test_size=0.3, seed=0)
    config = SimpleNamespace(
        is_visualize=False,
        visualization=SimpleNamespace(
            enabled=True,
            plot_types=["roc", "dca", "calibration"],
            dpi=72,
            format="png",
        ),
        n_splits=3,
        random_state=0,
    )
    paths = write_ml_figures_from_config(
        result,
        table,
        config,
        destination=tmp_path,
        mode="holdout",
    )
    assert paths == []
    assert list(tmp_path.iterdir()) == []


def test_write_ml_figures_from_config_holdout(tmp_path: Path) -> None:
    """Hold-out train result produces train_ and test_ curve suites."""
    from habit.spec import MLSpec, Spec
    from habit.datasets import make_synthetic_feature_table
    from habit.recipes.modeling import train_model

    table = make_synthetic_feature_table(n_rows=60, n_features=6, rng=1)
    spec = MLSpec(
        name="viz_on",
        classifier=Spec("LogisticRegression", {"max_iter": 300}),
        metrics=(Spec("accuracy"), Spec("auc")),
    )
    result = train_model(table, spec, test_size=0.3, seed=1)
    config = SimpleNamespace(
        is_visualize=True,
        visualization=SimpleNamespace(
            enabled=True,
            plot_types=["roc", "dca", "calibration"],
            dpi=72,
            format="png",
            explainability=None,
        ),
        n_splits=3,
        random_state=1,
    )
    paths = write_ml_figures_from_config(
        result,
        table,
        config,
        destination=tmp_path,
        mode="holdout",
    )
    # v0.1-aligned coverage: both train and test splits.
    assert len(paths) == 6
    for prefix in ("train_", "test_"):
        assert (tmp_path / f"{prefix}roc_curve.png").is_file()
        assert (tmp_path / f"{prefix}decision_curve.png").is_file()
        assert (tmp_path / f"{prefix}calibration_curve.png").is_file()


def test_shap_plot_family_returns_figures() -> None:
    """Each public SHAP plotter returns a Figure when shap is installed."""
    shap = pytest.importorskip("shap")
    from habit.viz.classification import (
        plot_shap_bar,
        plot_shap_decision,
        plot_shap_dependence,
        plot_shap_force,
        plot_shap_heatmap,
        plot_shap_summary,
        plot_shap_violin,
        plot_shap_waterfall,
    )

    rng = np.random.RandomState(0)
    features = rng.randn(40, 5)
    shap_values = 0.3 * features + 0.05 * rng.randn(40, 5)
    names = [f"feat_{i}" for i in range(5)]
    plotters = [
        lambda: plot_shap_summary(shap_values, features, feature_names=names),
        lambda: plot_shap_bar(shap_values, features, feature_names=names),
        lambda: plot_shap_violin(shap_values, features, feature_names=names),
        lambda: plot_shap_heatmap(shap_values, features, feature_names=names),
        lambda: plot_shap_dependence(
            shap_values, features, 0, feature_names=names
        ),
        lambda: plot_shap_waterfall(
            shap_values, features, 0, feature_names=names, base_value=0.1
        ),
        lambda: plot_shap_decision(
            shap_values,
            features,
            feature_names=names,
            sample_indices=[0, 1, 2],
            base_value=0.1,
        ),
        lambda: plot_shap_force(
            shap_values, features, 0, feature_names=names, base_value=0.1
        ),
    ]
    for plotter in plotters:
        fig = plotter()
        assert isinstance(fig, Figure)
    _ = shap


def test_rank_and_permutation_plotters() -> None:
    """Explainability helpers return figures / ranked indices."""
    from habit.viz.classification import (
        plot_permutation_importance,
        rank_shap_feature_indices,
        select_representative_sample_indices,
    )

    values = np.array(
        [[0.1, -0.5, 0.2], [0.2, -0.4, 0.0], [0.0, -0.6, 0.1]], dtype=float
    )
    ranked = rank_shap_feature_indices(values, top_k=2)
    assert list(ranked) == [1, 0] or list(ranked)[0] == 1
    samples = select_representative_sample_indices(values.sum(axis=1), n_samples=2)
    assert len(samples) == 2
    fig = plot_permutation_importance(
        ["a", "b", "c"],
        np.array([0.1, 0.4, 0.2]),
        importance_std=np.array([0.01, 0.02, 0.01]),
        top_k=2,
    )
    assert isinstance(fig, Figure)
