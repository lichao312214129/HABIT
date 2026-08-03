Plugin discovery API
====================

Inspect built-in and entry-point plugins (preprocessors, models, feature
extractors, metrics, and related domains).

::

   from habit import list_plugins, get_plugin_info, get_param_schema, load_plugins

   plugins = list_plugins("models")
   info = get_plugin_info(plugins[0].name, "models")
   schema = get_param_schema(info.name, "models")

Module reference
----------------

.. automodule:: habit.api.plugins
   :members:
   :undoc-members:
   :show-inheritance:
