"""Manual multi-process preprocessing + habitat smoke script.

On Windows, ``multiprocessing`` uses spawn: child processes re-import this
module. Keep executable logic under ``if __name__ == "__main__"`` so workers
do not re-start the batch pipeline.

To debug with ``processes > 1`` in Cursor/VS Code, launch via the
``HABIT: debug preprocess multi-process`` configuration in ``.vscode/launch.json``
(``subProcess: true``). Without that, the debugger only follows the parent and
may appear deadlocked at 0% progress.

Preferred clinical API usage:
    prepared = ClinicalPreprocessor(preprocessing_config).fit_transform()
    result = HabitatSegmenter(habitat_config).fit_transform(prepared)

``preprocessing_config["data_dir"]`` is the only input-directory declaration for
preprocessing. ``habitat_config`` may omit ``data_dir`` when a prepared cohort
is passed to ``fit_transform``.
"""

from habit import ClinicalPreprocessor, HabitatSegmenter

preprocessing_config = {
    "data_dir": "F:/work/habit_project/config/preprocessing/files_preprocessing_demo.yaml",
    "out_dir": "F:/work/habit_project/demo_data/results/preprocessed_api",
    "processes": 4,
    "preprocessing": {
        "resample": {
            "images": ["delay2", "delay3", "delay5"],
            "target_spacing": [1.0, 1.0, 1.0],
        }
    },
}

# data_dir is taken from prepared.data_dir when fit_transform(prepared) is used.
# Remaining fields match config_habitat_one_step_raw_concat_train.yaml.
habitat_config = {
    "run_mode": "train",
    "out_dir": "F:/work/habit_project/demo_data/results/habitat_api",
    "feature_construction": {
        "voxel_level": {
            "method": "concat(raw(delay2), raw(delay3), raw(delay5))",
            "params": {},
        },
        "preprocessing_for_subject_level": {
            "methods": [
                {
                    "method": "winsorize",
                    "winsor_limits": [0.05, 0.05],
                    "global_normalize": False,
                },
                {
                    "method": "minmax",
                    "global_normalize": False,
                },
            ],
        },
        "preprocessing_for_group_level": {
            "methods": [
                {
                    "method": "binning",
                    "n_bins": 10,
                    "bin_strategy": "uniform",
                    "global_normalize": False,
                },
            ],
        },
    },
    "habitat_segmentation": {
        "clustering_mode": "two_step",
        "supervoxel": {
            "algorithm": "kmeans",
            "n_clusters": 50,
            "max_iter": 300,
            "n_init": 10,
            "one_step_settings": {
                "min_clusters": 2,
                "max_clusters": 10,
                "fixed_n_clusters": None,
                "selection_method": "elbow",
                "plot_validation_curves": True,
            },
        },
        "habitat": {
            "algorithm": "kmeans",
            "max_clusters": 10,
            "habitat_cluster_selection_method": ["elbow"],
            "fixed_n_clusters": None,
            "max_iter": 300,
            "n_init": 10,
        },
    },
    "processes": 2,
    "cap_processes_to_gpu_pool": False,
    "individual_subject_timeout_sec": 900,
    "oom_backoff": False,
    "resume": True,
    "strict_checkpoint_hash": False,
    "checkpoint_dir": None,
    "force_rerun_subjects": [],
    "retry_failed_subjects": False,
    "individual_subject_auto_retry_rounds": 2,
    "individual_subject_parallel_mode": "persistent",
    "persistent_worker_max_consecutive_failures": 1,
    "persistent_worker_recycle_after_tasks": 0,
    "clear_checkpoint_on_success": False,
    "plot_curves": True,
    "save_results_csv": True,
    "random_state": 42,
    "debug": False,
}


if __name__ == "__main__":
    prepared = ClinicalPreprocessor(preprocessing_config).fit_transform()
    result = HabitatSegmenter(habitat_config).fit_transform(prepared)
    print(prepared.data_dir)
    print(result.pipeline_path)
    print(result.table.shape)
