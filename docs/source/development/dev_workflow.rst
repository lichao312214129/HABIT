Contributor Workflow
============

This page provides practical contribution guidance: environment setup, the
test system, and three common implementation scenarios—adding a CLI command,
adding a configuration-step schema, and using the Web GUI bridge. See
:doc:`contributing` for general pull-request and commit conventions.

Environment setup
--------

.. code-block:: bash

   conda activate habit          # Python 3.10
   pip install -r requirements.txt
   pip install -e .              # editable install
   pytest tests/                 # Run the tests

Code conventions
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Convention
     - Description
   * - Type annotations
     - Annotate function inputs and outputs explicitly, for example
       ``x_train: np.ndarray``.
   * - Comment language
     - Write detailed code comments in **English**.
   * - Plot text
     - All text generated inside programmatic plots must be **English**.
   * - Utility organization
     - Put shared utilities in ``habit/utils/`` and use
       ``habit/utils/progress_utils.py`` for progress bars.
   * - Code style
     - Follow PEP 8 and add docstrings to public functions.

Testing
--------

Tests are located in the repository-level ``tests/`` directory and configured
by ``tests/pytest.ini``. There are two main categories:

1. **pytest unit and CLI tests** (``test_*.py``), using
   ``click.testing.CliRunner`` or direct API calls.
2. **Executable demo scripts**, such as
   ``tests/habitat/habitat_two_step_voxel_radiomics_train.py``. These use demo
   YAML files under ``config/`` and provide end-to-end smoke coverage.

Select tests by marker (markers are defined in ``pytest.ini``):

.. code-block:: bash

   pytest tests/ -m unit                 # Unit tests only
   pytest tests/ -m "habitat and not slow"
   pytest tests/ -m cli                  # CLI tests only
   pytest tests/test_architecture_contracts.py -m unit  # Architecture contracts

Available markers: ``slow`` / ``integration`` / ``unit`` / ``preprocessing`` /
``habitat`` / ``ml`` / ``utils`` / ``cli``.

Architecture contract tests
~~~~~~~~~~~~

``tests/test_architecture_contracts.py`` protects two cross-subsystem
contracts. It **must pass** after adding a factory or orchestrator:

.. mermaid::

   flowchart TD
     TST["test_architecture_contracts.py"]
     TST --> R1["test_registry_subclasses_class_registry"]
     TST --> R2["test_registry_exposes_uniform_contract"]
     TST --> R3["test_registries_do_not_share_storage"]
     TST --> O1["test_orchestrator_exposes_terminal_methods"]

     R1 & R2 & R3 --> CR["ClassRegistry + 6 factories"]
     O1 --> OC["ORCHESTRATOR_CONTRACT + 7 orchestrators"]

For a new **class-based factory**, subclass ``ClassRegistry`` and add the
factory to the ``CLASS_REGISTRIES`` dictionary in the test file.
For a new **orchestrator**, append an
``(import_path, class_name, terminal_methods)`` tuple to
``ORCHESTRATOR_CONTRACT``.
Tests for missing optional dependencies are automatically skipped, so the
minimal installation remains usable.

Demo data flow
~~~~~~~~~~~

``tests/conftest.py`` provides fixtures such as ``project_root`` and
``demo_data_dir``; example data is stored in ``demo_data/``.
End-to-end examples are in ``tests/integration/`` (for example,
``workflow_preprocess_to_compare.py`` runs preprocess → get-habitat → extract
→ model → compare). Configuration paths are resolved relative to the YAML
file; see ``PathResolver`` in :doc:`configuration_system`.

Scenario 1: Add a CLI command
-------------------------

Keep the command declaration separate from its implementation:
``habit/cli.py`` declares options and forwards the call, while the
implementation lives in ``habit/commands/cmd_*.py``.

**1. Declare the command in ``habit/cli.py``** (use lazy imports to keep
startup fast):

.. code-block:: python

   @cli.command('my-task')
   @config_option()
   def my_task(config):
       """One-line help shown in `habit --help`."""
       from habit.commands.cmd_my_task import run_my_task
       run_my_task(config)

**2. Implement it in ``habit/commands/cmd_my_task.py``** (follow the
structure of ``cmd_preprocess.py``):

.. code-block:: python

   from habit.commands.common import (
       echo_success, exit_with_error, load_config_or_exit,
   )
   from habit.core.my_subsystem.config_schemas import MyConfig
   from habit.core.my_subsystem.run import run_my_task_from_config

   def run_my_task(config_path: str) -> None:
       """Run my task from a config file."""
       config = load_config_or_exit(MyConfig, config_path)
       try:
           run_my_task_from_config(config)
       except Exception as exc:  # noqa: BLE001
           exit_with_error(f"Error: {exc}")
       echo_success("My task completed successfully!")

**3. Put the core logic** in a ``run_*_from_config`` function under
``habit/core/.../run.py``. Keep the command layer thin and the core layer
focused; Python API users can then call the core function directly.

**4. Add a CLI test** in ``tests/.../test_cli_my_task.py`` using ``CliRunner``.

Scenario 2: Add a configuration-step schema
-------------------------------

If a new algorithm has configurable ``params``, define and register a
parameter schema to obtain **type validation** and **GUI form generation**.

**1. Define the parameter model** under ``habit/core/schemas/steps/``:

.. code-block:: python

   from pydantic import BaseModel, Field

   class MyStepParams(BaseModel):
       # Each field: type + default + constraint gives validation + GUI widget.
       n_clusters: int = Field(3, ge=2, description="Number of clusters")
       metric: str = Field("euclidean", description="Distance metric")

**2. Register it with ``ParamSchemaRegistry``** (see the initialization logic
in ``schemas/registry.py``):

.. code-block:: python

   ParamSchemaRegistry.register("habitat", "my_step", MyStepParams)

**3. Result**: ``validate_step_params()`` validates the step ``params`` while
loading configuration, and the GUI reflects the corresponding form fields from
``schemas/reflect.py``. See :doc:`configuration_system` for details.

.. tip::

   ``BaseConfig`` uses ``extra='forbid'``. Declare every new field in the
   schema; otherwise the configuration will report an unknown field.

Scenario 3: Web GUI and bridge
-------------------------

``habit-gui/`` is an optional, separate Web GUI (not checked out by default;
see :doc:`repo_layout`). It reuses the core through two channels and **does
not duplicate business logic**:

.. mermaid::

   flowchart LR
     WEB["React UI (habit-gui/web)"] -->|HTTP| API["FastAPI (habit-gui/api)"]
     API -->|subprocess| CLI["habit CLI (habit &lt;cmd&gt; --config)"]
     API -->|bridge worker| BR["habit-gui/bridge"]
     BR -->|import| SCH["habit.core.schemas (reflect)"]

- **Execution channel**: The GUI generates YAML configuration and invokes the
  ``habit`` CLI in a **subprocess**, producing the same behavior as the CLI.
- **Schema channel**: ``habit-gui/bridge`` reflects parameter models from
  ``habit.core.schemas`` and sends field descriptors to the frontend for
  dynamic form rendering. **Form fields therefore remain aligned with the
  CLI YAML schema.**

This design means that adding a core component and parameter schema usually
adds the corresponding GUI inputs without frontend changes.

.. seealso::

   - See :doc:`architecture` for the architecture overview and
     :doc:`configuration_system` for configuration details.
   - See :doc:`extension_points` for extension points and
     :doc:`../customization/index` for code templates.
