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
"""Tests for the consensus-clustering habitat fitter and its figures."""

from __future__ import annotations

import numpy as np
import pytest

from habit.exceptions import HABITAPIError
from habit.habitat_model import (
    ConsensusHabitatModelFitter,
    HabitatModelFitterRegistry,
)
from habit._protocols import HabitatModelFitter, Seedable
from habit.viz import (
    plot_consensus_cdf,
    plot_consensus_cluster_stability,
    plot_consensus_delta,
    plot_consensus_item_stability,
    plot_consensus_matrices,
    plot_consensus_matrix,
)

from .conftest import two_cluster_units


def _fitter(
    n_habitats: int | None = None,
    max_items: int = 2000,
) -> ConsensusHabitatModelFitter:
    """Build a small fitter; defaults keep the unit test to a few resamples."""
    return ConsensusHabitatModelFitter(
        n_habitats=n_habitats,
        min_habitats=2,
        max_habitats=3,
        n_resamples=4,
        resample_proportion=0.8,
        n_init=2,
        max_iter=50,
        max_items=max_items,
    )


@pytest.mark.unit
def test_consensus_fitter_is_registered_and_seedable() -> None:
    """The fitter is a habitat-model fitter and is constructed by name."""
    fitter = _fitter(n_habitats=2)
    assert isinstance(fitter, HabitatModelFitter)
    assert isinstance(fitter, Seedable)
    created = HabitatModelFitterRegistry.create(
        "consensus",
        n_habitats=2,
        n_resamples=4,
        resample_proportion=0.8,
        n_init=2,
    )
    assert isinstance(created, ConsensusHabitatModelFitter)
    assert created.spec.name == "consensus"


@pytest.mark.unit
def test_consensus_fit_is_seed_reproducible() -> None:
    """The same seed yields the same centroids and the same selected k."""
    units = two_cluster_units(supervoxels_per_subject=8)
    models = []
    for _ in range(2):
        fitter = _fitter()
        fitter.set_random_state(7)
        models.append(fitter.fit(units))
    np.testing.assert_array_equal(models[0].centroids, models[1].centroids)
    assert models[0].model_id == models[1].model_id
    report = models[0].preprocessing_state["consensus"]
    assert report["license"] == "GPL-3.0-or-later"
    assert report["n_items"] == 24
    assert int(report["selected_k"]) in (2, 3)
    matrices = np.asarray(report["matrices"])
    assert matrices.shape[0] == len(report["candidates"])
    assert matrices.shape[1] == 24
    delta = np.asarray(report["delta_k"])
    # Vendored rule: one delta entry per step, stopping short of max k.
    assert delta.shape == (len(report["delta_k_x"]),)
    assert len(report["delta_k_x"]) == len(report["candidates"]) - 1


@pytest.mark.unit
def test_consensus_fixed_k_separates_two_blobs() -> None:
    """A fixed K=2 on two blobs places one centroid near each blob."""
    units = two_cluster_units(supervoxels_per_subject=8)
    fitter = _fitter(n_habitats=2)
    fitter.set_random_state(3)
    model = fitter.fit(units)
    assert model.n_habitats == 2
    assert model.provenance.produced_by == "habitat_model_fitter.consensus"
    low = model.centroids.min(axis=0)
    high = model.centroids.max(axis=0)
    assert np.all(low < 2.0)
    assert np.all(high > 8.0)


@pytest.mark.unit
def test_consensus_refuses_voxel_scale_item_counts() -> None:
    """An n x n consensus matrix is refused above max_items."""
    units = two_cluster_units(supervoxels_per_subject=8)
    fitter = _fitter(max_items=10)
    fitter.set_random_state(0)
    with pytest.raises(HABITAPIError, match="max_items"):
        fitter.fit(units)


@pytest.mark.unit
def test_consensus_figures_draw_from_the_report() -> None:
    """The public plot helpers draw the stored report without a second fit."""
    units = two_cluster_units(supervoxels_per_subject=8)
    fitter = _fitter()
    fitter.set_random_state(1)
    model = fitter.fit(units)
    report = model.preprocessing_state["consensus"]
    for plot in (
        plot_consensus_delta,
        plot_consensus_cdf,
        plot_consensus_matrix,
        plot_consensus_matrices,
        plot_consensus_cluster_stability,
        plot_consensus_item_stability,
    ):
        figure = plot(report)
        assert figure.axes
        figure.clear()
