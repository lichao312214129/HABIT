:mod:`habit.combiners`: compose feature trees
=============================================

.. automodule:: habit.combiners
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. currentmodule:: habit.combiners

**User guide:** :doc:`domain_habitat`. Component names:
:doc:`../how_to/habitat_components`.

Concat, ratio, kinetic, and expression trees over extractors
(``concat(raw("T1"), voxel_radiomics("T2"))``).

Classes
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Combiner
   ConcatCombiner
   WeightedConcatCombiner
   AverageCombiner
   RatioCombiner
   DifferenceCombiner
   ExpressionCombiner
   KineticCombiner
   CombinerRegistry

Functions
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   block_sources
   check_blocks
   concat_blocks
