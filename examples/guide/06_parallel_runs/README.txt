6. Parallel runs
================

:class:`~habit.spec.HabitatSpec` is what to compute.
:class:`~habit.spec.RunPolicy` is how to schedule it.
The same cohort and the same ``fit_predict`` call take a different
backend. A fixed ``random_seed`` keeps the maps aligned with a serial run.

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
* **One worker per GPU** (recorded timings, not re-run here) —
  :doc:`/auto_examples/06_parallel_runs/plot_07_several_gpus`.

A per-subject time limit is enforced only by the process backend.
It is set on the process-pool page.
