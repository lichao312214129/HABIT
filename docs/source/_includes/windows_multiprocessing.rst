.. important::
   **Windows + process pool:** spawning workers re-imports your script.
   Put any call that starts :class:`~habit.execution.ProcessPoolBackend`
   (``RunPolicy(backend="process")``, ``workers > 1``, or a positive
   ``subject_timeout_sec`` that forces a child process) inside::

      if __name__ == "__main__":
          ...

   Running the same code at module top level (or pasting it into a ``.py``
   file without this guard) raises
   ``RuntimeError: ... bootstrapping phase`` on Windows.
   The ``habit`` CLI entry point is already safe; this applies to **scripts /
   notebooks converted to scripts / pure-Python recipes**.
   For a quick serial check, use ``RunPolicy(workers=1, backend="serial",
   subject_timeout_sec=None)`` (no spawn).
