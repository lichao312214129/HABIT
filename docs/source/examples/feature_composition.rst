Feature composition: trees, combiners, and statistics
=====================================================

Every feature extraction stage — voxel, supervoxel, habitat — accepts the
same **node** abstraction: a leaf (one extraction form over one or more
modalities) or a **combiner** node that merges its children column-wise.
Nodes nest into trees, so multi-modality designs compose single-modality
extractors with explicit combination logic instead of hiding it inside one
monolithic extractor.

Two spellings, one Spec
-----------------------

Any component entry accepts either the **expression form** (a strict string
parsed by :func:`~habit.spec.parse_feature_expression`) or the **structured
form** (nested ``name`` / ``params`` mappings). Both produce the identical
``Spec`` — and therefore the identical fingerprint:

.. code-block:: python

   from habit import Spec, parse_feature_expression

   expression = 'concat(raw("T1"), local_entropy("T2", kernel_size=3))'
   parsed = parse_feature_expression(expression)
   structured = Spec(
       "concat",
       {
           "children": [
               {"name": "raw", "params": {"modality": "T1"}},
               {"name": "local_entropy", "params": {"modality": "T2", "kernel_size": 3}},
           ],
       },
   )
   assert parsed.fingerprint() == structured.fingerprint()

Expression grammar (strict; ambiguous input is rejected loudly):

* **Modalities are quoted strings** — ``raw("T1")``. Bare identifiers are
  refused (``raw(T1)`` is a v0.1 spelling handled only by the legacy YAML
  adapter, never by this parser).
* **Parameters are explicit** — ``key=value`` with JSON-like literals
  (numbers, booleans, strings, lists).
* **Children are nested calls** — a quoted string among children becomes an
  implicit ``raw(...)`` leaf, preserving argument order.

Built-in combiners (domain ``combiner``): ``concat``, ``weighted_concat``
(``weights=[...]``), ``average``, ``ratio``, ``difference``, ``kinetic``,
``expression``. Leaves and combiners can nest arbitrarily.

Column naming and ``as_``
-------------------------

* Single-column leaves keep their **source label** as the column name
  (``raw("T1")`` → ``T1``); multi-column leaves prefix each feature with
  the family name (``local_entropy("T2")`` → ``local_entropy-T2``).
* ``as_="label"`` overrides the source label — use it to disambiguate two
  branches that would otherwise collide. ``as_`` renames only single-output
  nodes; aliasing a multi-column node raises immediately.

Supervoxel statistics
---------------------

``mean`` / ``std`` / ``percentile`` are single-modality supervoxel
extractors that aggregate the (optionally preprocessed) voxel signal of one
modality per supervoxel. Inside ``SubjectPipeline`` the voxel fields are
bound automatically; ``source="original"`` selects the pre-preprocessing
signal. They compose with the same combiners:

.. code-block:: yaml

   supervoxel_feature_extractor:
     name: concat
     params:
       children:
         - {name: mean, params: {modality: T1}}
         - {name: std, params: {modality: T1, as_: t1_spread}}
         - {name: percentile, params: {modality: T2, q: 90}}

Runnable demo
-------------

.. literalinclude:: scripts/feature_composition_demo.py
   :language: python

Output::

   === expression -> Spec ===
     expression: concat(raw("T1"), local_entropy("T2", kernel_size=3))
     parsed == structured: True
     fingerprint equal: True

   === voxel tree (atomic) ===
     columns: ['T1', 'local_entropy-T2']

   === weighted_concat + ratio with as_ aliases ===
     columns: ['t1w', 't2w', 't1_over_t2']

   === supervoxel statistics tree (pipeline) ===
     columns: ['T1', 'std-t1_spread', 'p90-T2']
     units: 6 supervoxels

   === YAML dual form ===
     fingerprint equal: True

What to read next
-----------------

* :doc:`../api/domain` — registries and protocols, including ``Combiner``
* :doc:`../api/spec` — Spec trees, fingerprints, YAML isomorphism
* :doc:`habitat_feature_routes` — the classic per-route extraction designs
