Python API (v1.0)
=================

HABIT v1.0 is **API-first**. The documented surface is the layered core:

``contracts`` → ``domain`` → ``spec`` / ``execution`` → ``kernels`` / ``compat``.

CLI and YAML are thin shells over the same objects. The former ``run_*`` /
clinical-facade documentation from v0.1 is **retired** from this section.

.. toctree::
   :maxdepth: 2

   python_api
   data_model
   adapters
   domain_habitat
   domain_table
   spec
   execution
   kernels
   compat
   plugins
   registry
   image_io
   exceptions
