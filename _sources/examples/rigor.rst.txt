Rigor
=====

Acceleration and parallelism must not change what a column means. A faster
implementation of the same formula may stay under the old name. A different
discretisation, binning, interpolation, or null model is a different
scientific object — give it a new option name (and document it), do not
reuse the old column as if the definition were unchanged.

IBSI
----

When HABIT and PyRadiomics claim the same IBSI quantity, the default
reference data is the IBSI-1 Phase 1 digital phantom (not ``demo_data/``).
Texture aggregation is IBSI **3-D averaged**. Kurtosis is Pearson kurtosis:
HABIT / PyRadiomics = IBSI excess kurtosis + 3. Phantom files, published
refs, and the comparison table: :doc:`../reference/features/traditional`.
Regression: ``tests/kernels/test_ibsi_digital_phantom.py``.

PyRadiomics and GPU matrices
----------------------------

ROI-level, voxel-level, and 3-D shape radiomics match PyRadiomics
``execute()`` under the same discretisation. GPU count matrices
(GLCM / GLRLM / GLSZM / GLDM / NGTDM) are bit-identical to the C
extension; see :doc:`voxel_texture` (``use_gpu_matrices``).

Provenance
----------

:meth:`~habit.spec.HabitatSpec.fingerprint` changes when stage params
change. :class:`~habit.contracts.RunManifest` records what actually ran.
:meth:`~habit.contracts.RunManifest.describe_methods` drafts a methods
paragraph from that record.

.. literalinclude:: scripts/provenance_methods_demo.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

.. figure:: ../_static/images/examples/provenance_methods_overlay.png
   :alt: Habitat overlay from the provenance demo study
   :width: 420

   :func:`~habit.viz.plot_habitat_overlay` on the fingerprinted run.

Golden tests
------------

``tests/golden/`` locks voxel-wise habitat maps; floating-point features
use ``rtol=1e-6``.

What to read next
-----------------

* :doc:`feature_extraction` — volume, MSI, ITH, graph
* :doc:`visualization` — publication overlay
* :doc:`../reference/features/traditional` — IBSI phantom table
