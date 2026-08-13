Design Philosophy
=================

This page explains why HABIT has its current architecture. The architecture
page describes what the system is; the extension guide describes how to
change it. The section below is written as **limitations of the research
setting HABIT is built for**, not as a feature list.

Limitations
-----------

Imaging papers that survive peer review have to be **repeatable** (same
operator, same protocol, similar numbers) and **reproducible** (independent
group, same scientific conclusion). Habitat analysis sits where both fail
easily: voxel intensities depend on scanner, reconstruction, and
preprocessing; clustering depends on :math:`k`, the feature vector, and the
validation rule; a one-voxel shift of a mask can rewrite a radiomic map.
HABIT does not remove that physics. It makes the choices that *do* change
the map **explicit, typed, and carried with the result**
(:class:`~habit.contracts.habitat.HabitatModel`, :class:`~habit.spec.specs.Spec`,
:meth:`~habit.contracts.RunManifest.describe_methods`) so a reviewer can see what was fixed
and a second lab can run the same chain. A green unit test is not a
multi-centre replication.

A second limitation is who has to implement the chain. Many
clinician-researchers cannot program, or can write code that runs but is
wrong in ways that never raise: a leakage-prone split, a silhouette search
that is re-tuned after seeing the outcome, a YAML key that is silently
ignored. HABIT's answer is not "everyone should become a software engineer".
It is to put the scientific objects in a **library API** that can be copied,
and to keep CLI / YAML as **shells** over that API (see below). The
limitation remains: a GUI or a config file cannot invent a clinical
question, an endpoint, or an external validation set.

A third limitation is the state of typical pipelines. Preprocessing,
habitat definition, feature extraction, and modelling are often four
unrelated scripts with parameters chosen because they "looked reasonable"
or because a previous paper used them on a different organ. HABIT ships
**one ordered chain** with schema-checked parameters and registry-selected
algorithms, and it documents community-used defaults (for k-means habitat
count: inertia **elbow**, :math:`k\in[2,10]`). Defaults are a starting
protocol, not a claim that elbow-2–10 is optimal for every tumour. If a
study changes :math:`k`, the feature set, or the preprocessor after
unblinding the endpoint, HABIT will still run — it cannot police that
scientific error.

API-first (CLI and YAML are not the architecture)
-------------------------------------------------

HABIT is designed so that **the Python API is the product**. CLI subcommands
and YAML files are interfaces that assemble the same operators. The test
is: if ``habit/cli.py`` and ``habit/commands/`` were deleted, the scientific
capability should still be callable as ``op(subject)`` or
``Study(spec).fit_predict(cohort)``. Configuration is not a layer of the
architecture; it is a serialisation of :class:`~habit.spec.specs.Spec`.

That choice has costs. Contributors must keep schemas, registries, and
public symbols in sync. Third parties who only want a one-line CLI still
have to accept that the meaning of a run lives in the spec and the
manifest, not in the filename of a YAML. The gain is that a notebook, a
MONAI loop, or another product can embed one habitat step without HABIT's
directory layout.

Five design pillars
-------------------

These pillars are how HABIT responds to the limitations above; they are
not a claim that the limitations are solved.

.. mermaid::

   flowchart LR
     P1["Reproducible choices"] --> S1["Config as interface"]
     P1 --> S2["Typed schema"]
     P2["Cohort execution"] --> S5["Thin command, thick operators"]
     P3["Replaceable algorithms"] --> S3["Registry decoupling"]
     P4["Train = predict"] --> S4["Unified contracts"]
     P5["CLI = API shells"] --> S5

1. **Configuration is an interface, not the core.** YAML and Python dicts
   share one validation path so a CLI user and an API user are not running
   two products.
2. **Schemas fail fast.** Pydantic validates before computation. Unknown
   fields are rejected where the schema uses ``extra='forbid'``.
3. **Registries decouple algorithms.** Clustering, features, and models are
   selected by name so a study can swap an implementation without rewriting
   the workflow.
4. **Train and predict share a contract.** Inference must reuse the
   processing fitted on the training cohort, including cohort-level
   preprocessing state on :class:`~habit.contracts.habitat.HabitatModel`.
5. **Commands remain thin.** CLI entry points delegate to
   ``habit.api`` / ``habit.recipes``. They must not grow a second copy of
   the science.

Trade-offs
----------

Typed schemas, registries, and frozen specs constrain implementation
freedom in exchange for a result that can be named in a methods section.
They do not replace a protocol, an endpoint, or an external test set. The
non-negotiable engineering rules are documented in :doc:`invariants`.

See also
--------

* :doc:`configuration_system` for configuration assembly.
* :doc:`request_lifecycle` for a complete command lifecycle.
* :doc:`mental_model` for the core vocabulary.
* :doc:`../tutorial/precise_screening` for morphology-aware feature
  screening before clustering (Prior et al., 2024).
