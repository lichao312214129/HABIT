Provenance and methods paragraphs
=================================

**Level:** persistence / reporting · **Data:** synthetic · **Extras:** none · **Time:** ~10–30 s

Provenance is part of the data model:

* :meth:`~habit.spec.HabitatSpec.fingerprint` changes when stage params change
* :class:`~habit.contracts.RunManifest` records what **actually ran**
* :meth:`~habit.contracts.RunManifest.describe_methods` drafts a manuscript
  methods paragraph from that record

Script
------

.. literalinclude:: scripts/provenance_methods_demo.py
   :language: python

Run::

   python docs/source/examples/scripts/provenance_methods_demo.py

What to read next
-----------------

* :doc:`persistence` — ``StudyResult.save`` writes ``run_manifest.json``
* :doc:`apply_saved_model` — reusable ``.habitatmodel`` archives
* :doc:`../api/contracts` — ``Provenance`` / ``RunManifest`` reference
