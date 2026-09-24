6. Parallel runs
================

These pages only change how subjects are scheduled. The texture, the
preprocessing and the habitat model stay the same, so a finished map
matches a serial run of the same chain. That check is on the process-pool
page. Preprocessing of the texture itself is
:doc:`/auto_examples/02_voxel/plot_06_texture_preprocessing`.

Which backend
-------------

* **One process, subjects in order.** Use this to debug a single case.
  :doc:`/auto_examples/06_parallel_runs/plot_01_serial`.
* **A pool of worker processes.** On a CPU machine the subjects run side
  by side. On one GPU only the first worker gets the card and the others
  run on CPU, so the pool is slower; ``HABIT_GPU_OVERSUBSCRIBE=wrap``
  shares that card instead. Several GPUs are one worker per card.
  :doc:`/auto_examples/06_parallel_runs/plot_05_process_pool`.
* **Keep those workers up between passes.** A cold pool pays spawn and
  imports on every ``map``. ``with backend.reuse_workers():`` pays it once.
  :doc:`/auto_examples/06_parallel_runs/plot_06_reuse_workers`.
* **One fresh process per subject.** Same results, higher spawn cost.
  :doc:`/auto_examples/06_parallel_runs/plot_07_isolated`.
* **One worker per GPU.** The call is on the page. The timing table is a
  recorded 5-GPU run, because this machine has one GPU.
  :doc:`/auto_examples/06_parallel_runs/plot_09_several_gpus`.

When a subject fails
--------------------

* Keep the batch and record the exception:
  :doc:`/auto_examples/06_parallel_runs/plot_02_skip_failed`.
* Stop at the first failure:
  :doc:`/auto_examples/06_parallel_runs/plot_03_stop_on_failure`.
* Skip subjects that already finished:
  :doc:`/auto_examples/06_parallel_runs/plot_04_resume`.
* Give each subject a wall-clock limit. Only the process backend enforces
  it:
  :doc:`/auto_examples/06_parallel_runs/plot_08_wall_clock`.
