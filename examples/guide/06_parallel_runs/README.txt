6. Parallel runs
================

Voxel texture (four DCE series, radius 3, ``binWidth`` 12) is the
workload. Feature preprocessing changes the numbers. A backend only
changes how subjects are scheduled. GPU and ``voxel_batch`` are set on
the extractor; they do not change the texture definition.

Preprocessing, one method per page. Scheduling on these pages is
:class:`~habit.execution.SerialBackend`.

* **Subject z-score** (and the matching ``HabitatSpec``) —
  :doc:`/auto_examples/06_parallel_runs/plot_08_subject_zscore`.
* **Subject robust scaling** —
  :doc:`/auto_examples/06_parallel_runs/plot_09_subject_robust`.
* **Subject winsorizing** —
  :doc:`/auto_examples/06_parallel_runs/plot_10_subject_winsorize`.
* **Cohort z-score** —
  :doc:`/auto_examples/06_parallel_runs/plot_11_cohort_zscore`.
* **Cohort binning** —
  :doc:`/auto_examples/06_parallel_runs/plot_12_cohort_binning`.

Scheduling uses the subject z-score chain unless the page is about a
failure. A fixed ``random_seed`` keeps finished maps aligned with a
serial run.

* **One machine, in order** —
  :doc:`/auto_examples/06_parallel_runs/plot_01_serial`.
* **One bad case should not stop the batch** —
  :doc:`/auto_examples/06_parallel_runs/plot_02_skip_failed`.
* **A bad case should stop the batch** —
  :doc:`/auto_examples/06_parallel_runs/plot_03_stop_on_failure`.
* **A crashed run should skip finished subjects** —
  :doc:`/auto_examples/06_parallel_runs/plot_04_resume`.
* **Several CPU workers** —
  :doc:`/auto_examples/06_parallel_runs/plot_05_process_pool`.
* **Do not pay worker startup twice** —
  :doc:`/auto_examples/06_parallel_runs/plot_06_reuse_workers`.
* **One fresh process per subject** —
  :doc:`/auto_examples/06_parallel_runs/plot_13_isolated`.
* **Per-subject wall-clock limit** —
  :doc:`/auto_examples/06_parallel_runs/plot_14_wall_clock`.
* **One worker per GPU** (recorded timings, not re-run here) —
  :doc:`/auto_examples/06_parallel_runs/plot_07_several_gpus`.
