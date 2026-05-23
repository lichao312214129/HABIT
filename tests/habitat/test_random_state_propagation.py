"""Tests for random_state propagation in habitat analysis config."""

from __future__ import annotations

from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig


def _minimal_habitat_config(**overrides: object) -> HabitatAnalysisConfig:
    base = {
        "data_dir": "./data",
        "out_dir": "./out",
        "run_mode": "train",
        "random_state": 100,
        "FeatureConstruction": {
            "voxel_level": {"method": "raw(T2)", "params": {}},
        },
        "HabitatSegmentation": {
            "clustering_mode": "two_step",
            "supervoxel": {"algorithm": "kmeans", "n_clusters": 5},
            "habitat": {"algorithm": "kmeans", "max_clusters": 4},
        },
    }
    base.update(overrides)
    return HabitatAnalysisConfig.from_dict(base)


def test_effective_supervoxel_inherits_top_level() -> None:
    config = _minimal_habitat_config()
    assert config.effective_supervoxel_random_state() == 100


def test_effective_supervoxel_yaml_override() -> None:
    config = _minimal_habitat_config(
        HabitatSegmentation={
            "clustering_mode": "two_step",
            "supervoxel": {
                "algorithm": "kmeans",
                "n_clusters": 5,
                "random_state": 7,
            },
            "habitat": {"algorithm": "kmeans", "max_clusters": 4},
        }
    )
    assert config.effective_supervoxel_random_state() == 7


def test_effective_habitat_yaml_override() -> None:
    config = _minimal_habitat_config(
        HabitatSegmentation={
            "clustering_mode": "two_step",
            "supervoxel": {"algorithm": "kmeans", "n_clusters": 5},
            "habitat": {
                "algorithm": "kmeans",
                "max_clusters": 4,
                "random_state": 9,
            },
        }
    )
    assert config.effective_habitat_random_state() == 9
    assert config.effective_supervoxel_random_state() == 100


def test_one_step_individual_prefers_habitat_random_state() -> None:
    config = _minimal_habitat_config(
        HabitatSegmentation={
            "clustering_mode": "one_step",
            "supervoxel": {
                "algorithm": "kmeans",
                "n_clusters": 5,
                "random_state": 7,
            },
            "habitat": {
                "algorithm": "kmeans",
                "max_clusters": 4,
                "random_state": 9,
            },
        }
    )
    assert config.effective_individual_clustering_random_state() == 9
    assert config.effective_habitat_random_state() == 9


def test_one_step_individual_falls_back_to_supervoxel() -> None:
    config = _minimal_habitat_config(
        HabitatSegmentation={
            "clustering_mode": "one_step",
            "supervoxel": {
                "algorithm": "kmeans",
                "n_clusters": 5,
                "random_state": 7,
            },
            "habitat": {"algorithm": "kmeans", "max_clusters": 4},
        }
    )
    assert config.effective_individual_clustering_random_state() == 7


def test_two_step_individual_uses_supervoxel_only() -> None:
    config = _minimal_habitat_config(
        HabitatSegmentation={
            "clustering_mode": "two_step",
            "supervoxel": {
                "algorithm": "kmeans",
                "n_clusters": 5,
                "random_state": 7,
            },
            "habitat": {
                "algorithm": "kmeans",
                "max_clusters": 4,
                "random_state": 9,
            },
        }
    )
    assert config.effective_individual_clustering_random_state() == 7
    assert config.effective_habitat_random_state() == 9


def test_plot_random_state_scopes() -> None:
    config = _minimal_habitat_config(
        HabitatSegmentation={
            "clustering_mode": "two_step",
            "supervoxel": {
                "algorithm": "kmeans",
                "n_clusters": 5,
                "random_state": 7,
            },
            "habitat": {
                "algorithm": "kmeans",
                "max_clusters": 4,
                "random_state": 9,
            },
        }
    )
    assert config.effective_clustering_plot_random_state("individual") == 7
    assert config.effective_clustering_plot_random_state("group") == 9
