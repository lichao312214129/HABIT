Plugin discovery API
====================

Inspect built-in and entry-point plugins (preprocessors, models, feature
extractors, metrics, and related domains).

::

   from habit import list_plugins, get_plugin_info, get_param_schema, load_plugins

   plugins = list_plugins("models")
   info = get_plugin_info("models", plugins[0].name)
   schema = get_param_schema("models", info.name)

Module reference
----------------

.. automodule:: habit.api.plugins
   :members:
   :undoc-members:
   :show-inheritance:
