Component registry
==================

``ComponentRegistry`` is the base for every domain registry. Built-in domains
subclass it; plugins register via entry points ``habit.<domain>``.

.. code-block:: python

   from habit.registry import ComponentRegistry
   from pydantic import BaseModel, Field


   class MyParams(BaseModel):
       alpha: float = Field(1.0, gt=0)


   class MyOp:
       def __init__(self, alpha: float = 1.0) -> None:
           self.alpha = alpha

       def __call__(self, subject):
           return subject


   class MyRegistry(ComponentRegistry[MyOp]):
       domain = "my_op"


   MyRegistry.register("demo", MyOp)
   MyRegistry.register_params_model("demo", MyParams)

   op = MyRegistry.create("demo", alpha=2.0)
   print(MyRegistry.available())
   print(MyRegistry.params_model("demo"))

Standard methods on every registry:

* ``register(name, cls)``
* ``register_params_model(name, model)``
* ``create(name, **params)``
* ``available()``
* ``params_model(name)``

See :doc:`domain_habitat` and :doc:`domain_table` for the built-in registries.
