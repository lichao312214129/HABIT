Plugin introspection API
========================

Discover built-in and entry-point components. Parameter order is always
``(name, domain)``.

.. code-block:: python

   from habit import (
       get_param_schema,
       get_plugin_info,
       list_plugins,
       load_plugins,
   )

   report = load_plugins(strict=False)
   print(report.loaded, report.failures)

   all_plugins = list_plugins()
   habitat_only = list_plugins("habitat_model_fitter")

   info = get_plugin_info("slic", "supervoxelizer")
   print(info.name, info.domain, info.implementation, info.provider)

   schema = get_param_schema("slic", "supervoxelizer")
   if schema is not None:
       json_schema = schema.model_json_schema()

v1 protocol domains
-------------------

Use these domain strings with ``list_plugins`` / ``get_plugin_info`` /
``get_param_schema``:

.. code-block:: python

   V1_DOMAINS = [
       "voxel_feature_extractor",
       "supervoxelizer",
       "supervoxel_feature_extractor",
       "habitat_model_fitter",
       "habitat_assigner",
       "habitat_feature_extractor",
       "preprocessor",
       "table_preprocessor",
       "feature_selector",
       "classifier",
       "metric",
   ]
   for domain in V1_DOMAINS:
       print(domain, [p.name for p in list_plugins(domain)])

Enumerate every component and fetch its schema::

   for plugin in list_plugins():
       schema = get_param_schema(plugin.name, plugin.domain)
       print(plugin.domain, plugin.name, schema)

Types
-----

* ``PluginInfo`` — name, domain, implementation, params_schema, provider
* ``PluginLoadReport`` — result of ``load_plugins``

Registering a third-party plugin
--------------------------------

External packages make components discoverable by declaring HABIT's entry
point groups in their own packaging metadata. The group name is ``habit.``
followed by the domain string; v1.0 domains are the snake_case singular
protocol names listed above (the plural v0.1 groups remain honoured for
legacy factories). HABIT itself declares no empty groups — setuptools strips
them — the authoritative list is
``habit/api/plugins.py::_ENTRY_POINT_GROUPS``.

In the plugin's ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."habit.supervoxelizer"]
   my_slic_variant = "my_package.my_module:MySupervoxelizer"

The registered object must implement the matching protocol from
``habit.domain.protocols`` and expose a ``spec`` (see :doc:`spec`). After
``pip install`` of the plugin package, ``load_plugins()`` surfaces it and
``list_plugins("supervoxelizer")`` reports ``provider="<distribution>"``.
