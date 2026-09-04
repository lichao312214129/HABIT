.. _api-registry:

Component registry
==================

.. automodule:: habit.registry
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. currentmodule:: habit.registry

**User guide:** :doc:`plugins` · :doc:`domain_habitat` ·
:doc:`domain_table`.

``ComponentRegistry`` is the base for every capability registry. Built-in
domains subclass it; plugins register via entry points ``habit.<domain>``.

Classes
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ComponentRegistry

.. code-block:: python

   from habit.registry import ComponentRegistry

   class MyOp:
       def __init__(self, alpha: float = 1.0) -> None:
           self.alpha = alpha

       def __call__(self, subject: object) -> object:
           return subject


   class MyRegistry(ComponentRegistry[MyOp]):
       domain = "my_op"


   MyRegistry.register("demo", MyOp)

   op = MyRegistry.create("demo", alpha=2.0)
   print(MyRegistry.available())
   print(MyRegistry.constructor_signature("demo"))

Standard methods on every registry:

* ``register(name, cls)``
* ``create(name, **params)``
* ``available()``
* ``constructor_signature(name)``

See :doc:`domain_habitat` and :doc:`domain_table` for the built-in registries.
