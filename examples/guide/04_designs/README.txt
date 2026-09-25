4. Three habitat designs
========================

**Background.** A habitat design decides which rows are clustered and
whether one model is shared by the whole cohort or fitted per subject.
**Purpose.** Run the same data through each design and see what changes:
shared vs. per-subject habitat ids, supervoxels vs. voxels.

The complete analysis is two-step: ``partition``, then ``pool``, then
``fit``. These pages change that stage list and nothing else.

====================  =============  ========  =====================
Design                ``partition``  ``pool``  ``fit`` runs on
====================  =============  ========  =====================
two-step              yes            yes       supervoxels, cohort
inside each subject   no             no        voxels, one subject
pooling voxels        no             yes       voxels, cohort
====================  =============  ========  =====================

``two_step_habitat``, ``one_step_habitat`` and ``direct_pooling_habitat``
build those lists. Habitat ids match across subjects only when ``fit``
ran once on the cohort. Otherwise match labels first
(:doc:`/auto_examples/06_matching/index`).

* **Two steps** — :doc:`/auto_examples/04_designs/plot_01_two_step`.
* **Inside each subject** —
  :doc:`/auto_examples/04_designs/plot_02_inside_each_subject`.
* **Pooling voxels** — :doc:`/auto_examples/04_designs/plot_03_pool_voxels`.

Applying a saved model is the next section. Opening each stage is
:doc:`/auto_examples/02_stages/index`.
