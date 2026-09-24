6. Parallel runs
================

These pages only change how subjects are scheduled. The study is the one
from the Python quickstart, so a pooled run gives the same habitat maps
as a serial run; the first page checks that voxel by voxel.

* **Running a cohort in parallel.** Serial versus a pool of four worker
  processes, with real timings, and which backend to use on CPU, one GPU
  or several GPUs.
  :doc:`/auto_examples/06_parallel_runs/plot_01_run_a_cohort`.
* **When a subject fails.** Continue or stop at the first failure,
  resume from a checkpoint, and a per-subject time limit.
  :doc:`/auto_examples/06_parallel_runs/plot_02_when_a_subject_fails`.
* **Several GPUs.** One worker per card. The timing table is a recorded
  5-GPU run.
  :doc:`/auto_examples/06_parallel_runs/plot_03_several_gpus`.
