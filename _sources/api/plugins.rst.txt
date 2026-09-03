Plugin introspection API
========================

.. automodule:: habit.api.plugins
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. currentmodule:: habit.api.plugins

**User guide:** :doc:`../how_to/habitat_components` · :doc:`registry`.
Discover built-in and entry-point components. Parameter order is always
``(name, domain)``.

Classes
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   PluginInfo
   PluginParamInfo
   PluginCatalogEntry
   PluginLoadReport

Functions
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   list_plugins
   get_plugin_info
   get_param_schema
   plugin_catalog
   format_plugin_catalog_rst
   load_plugins

.. currentmodule:: habit.api.utils

.. autosummary::
   :toctree: generated
   :nosignatures:

   setup_logger
   is_available
   show_versions
   check_component

.. code-block:: python

   from habit.api.plugins import get_plugin_info, list_plugins, load_plugins
   from habit.supervoxel import SlicSupervoxelizer, SupervoxelizerRegistry

   report = load_plugins(strict=False)
   print(report.loaded, report.failures)

   all_plugins = list_plugins()
   habitat_only = list_plugins("habitat_model_fitter")

   info = get_plugin_info("slic", "supervoxelizer")
   print(info.name, info.domain, info.implementation, info.provider)

   # Parameters live on the component constructor, not a Params model.
   print(SupervoxelizerRegistry.available())
   print(SupervoxelizerRegistry.constructor_signature("slic"))
   slic = SupervoxelizerRegistry.create("slic", n_supervoxels=50)
   slic = SlicSupervoxelizer(n_supervoxels=50)

``load_plugins`` resilience
---------------------------

* ``strict=False`` (default) — broken entry points are recorded in
  ``PluginLoadReport.failures`` and logged as warnings; discovery continues
* ``strict=True`` — the first load error is re-raised immediately

.. code-block:: python

   report = load_plugins(strict=False)
   if report.failures:
       for name, message in report.failures.items():
           print("plugin failure:", name, message)

   # CI / production gate: abort if any third-party plugin is broken
   load_plugins(strict=True)

Legacy plural domains (e.g. ``habitat_features``, ``models``) still resolve
but emit ``HabitDeprecationWarning``; prefer the v1 singular domains below.

v1 protocol domains
-------------------

Use these domain strings with ``list_plugins`` / ``get_plugin_info``:

.. code-block:: python

   V1_DOMAINS = [
       "voxel_feature_extractor",
       "supervoxelizer",
       "supervoxel_feature_extractor",
       "habitat_model_fitter",
       "habitat_assigner",
       "habitat_feature_extractor",
       "combiner",
       "image_perturbation",
       "preprocessor",
       "table_preprocessor",
       "feature_selector",
       "classifier",
       "metric",
   ]
   for domain in V1_DOMAINS:
       print(domain, [p.name for p in list_plugins(domain)])

Enumerate registered names::

   for plugin in list_plugins():
       print(plugin.domain, plugin.name)

Which ``Spec`` / ``create`` names exist
---------------------------------------

The component ``__init__`` is the only parameter contract: types, defaults,
allowed values or ranges, validation, and same-named public attributes.
``Registry.create("name", **kwargs)`` forwards those constructor
arguments. Do not copy parameter tables into YAML or notebooks — they
rot. Look names up with ``list_plugins`` / ``Registry.available()``, and
read the constructor (or ``Registry.constructor_signature``) for
parameters.

.. code-block:: python

   from habit.api.plugins import list_plugins
   from habit.table_preprocessing import TablePreprocessorRegistry

   print([info.name for info in list_plugins("table_preprocessor")])
   print(TablePreprocessorRegistry.constructor_signature("minmax"))
   scaler = TablePreprocessorRegistry.create("minmax")

A bad name on ``Registry.create`` / ``get_plugin_info`` lists the names
that *are* registered in that domain.

Live catalog
~~~~~~~~~~~~

Each entry is one line of purpose, required vs optional constructor
parameters, one ``Spec`` / ``Registry.create`` example, and a
parameter table (default, allowed values / type, meaning). YAML
``name:`` / ``params:`` blocks use the same constructor names.

Habitat stages organised by scientific module:
:doc:`../how_to/habitat_components`.

.. include:: _generated_plugin_catalog.rst

Types
-----

* ``PluginInfo`` — name, domain, implementation, provider
* ``PluginLoadReport`` — result of ``load_plugins``

Registering a third-party plugin
--------------------------------

External packages make components discoverable by declaring HABIT's entry
point groups in their own packaging metadata. The group name is ``habit.``
followed by the domain string; v1.0 domains are the snake_case singular
protocol names listed above (older plural group names remain honoured for
legacy factories). HABIT itself declares no empty groups — setuptools strips
them — the authoritative list is
``habit/api/plugins.py::_ENTRY_POINT_GROUPS``.

In the plugin's ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."habit.supervoxelizer"]
   my_slic_variant = "my_package.my_module:MySupervoxelizer"

   [project.entry-points."habit.voxel_feature_extractor"]
   t1_t2_contrast = "my_package.features:register"

The registered object must implement the matching protocol from the
capability package (for example
:class:`~habit.supervoxel.Supervoxelizer`) and expose a ``spec``
(see :doc:`spec`). For
``voxel_feature_extractor``, the entry point commonly points at a callable
that performs ``@VoxelFeatureExtractorRegistry.register(...)`` (same pattern
as other domains). After ``pip install`` of the plugin package,
``load_plugins()`` surfaces it and ``list_plugins("supervoxelizer")`` /
``list_plugins("voxel_feature_extractor")`` report
``provider="<distribution>"``.

Runnable DIY example: :doc:`../examples/custom_voxel_features`.
