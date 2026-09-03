One-step habitat analysis
=========================

**Level:** recipe · **Data:** ``demo_data/preprocessed`` · **Extras:** optional
``[view]`` · **Time:** ~20–60 s

When to use: **one_step** = neither ``partition`` nor ``pool``. Habitats
are defined **per subject**. Integer ids are permuted — align them before
comparing patients (:doc:`apply_saved_model`, :doc:`habitat_label_match`).
Two-step / direct pooling: :doc:`two_step_habitat`,
:doc:`direct_pooling_habitat`.

The one-step design clusters **voxels inside each subject independently**.
Declare stages with **neither** ``partition`` **nor** ``pool``, then call
:meth:`~habit.recipes.Study.fit_predict`. There is **no cohort-level preprocessing
chain** at train time — per-subject state is frozen into
``StudyResult.subject_models`` rather than a single
:class:`~habit.contracts.HabitatModel`.

The factory :func:`~habit.recipes.one_step_habitat` remains for convenience.

For a long cohort, pass a :class:`~habit.report.Report` so each subject's
map, ``.habitatmodel``, and figures land on disk **as that subject
completes**. ``retain="tables"`` drops voxel-level units and maps from
the parent process. ``Report`` is not a :class:`~habit.spec.HabitatSpec`
stage — changing a figure does not invalidate scientific checkpoints.
Streaming is currently wired for ``one_step`` only.

Script
------

Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed tree.

.. literalinclude:: scripts/one_step_habitat_demo.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

Draw the figures
----------------

Paste this after the Script block (it uses ``cohort``, ``result``, and
``ROI``). Writes ``out/one_step_*.png``.

.. literalinclude:: scripts/one_step_habitat_demo.py
   :language: python
   :start-after: # BEGIN figures
   :end-before: # END figures

Stream per subject (``report=``)
--------------------------------

Same science, different run object. Construct a :class:`~habit.report.Report`
and pass it to :meth:`~habit.recipes.Study.fit_predict`. Built-in figure
atoms call :mod:`habit.viz` (pure ``Figure`` out); the Report writes PNGs
atomically under ``<writer.root>/figures``. Set
``figure_layout="by_subject"`` to nest each subject's PNGs in a
subdirectory (``figures/<subject_id>/<kind>.png``); the default
``"flat"`` keeps ``figures/<subject_id>_<kind>.png``. Graph figures
(:class:`~habit.report.GraphSlice`, :class:`~habit.report.GraphNetwork2D`)
take the same :class:`~habit.kernels.HabitatGraphFeatureOptions` as
``Spec("graph")``. Those PNGs are a representative 2D slice
(display-only); graph metrics use the full 3D volume.

.. literalinclude:: scripts/one_step_report_demo.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

``writer=`` / ``retain=`` without a Report still work: they build the same
object. Prefer ``report=`` when you want figures. Custom figures implement
the :class:`~habit.report.FigureAtom` protocol (``stem`` + ``draw``).

Run from the repository root::

   python docs/source/examples/scripts/one_step_report_demo.py

Change the Spec (other methods and parameters)
----------------------------------------------

The script above is **one worked recipe**, not the only recipe. Each
``Stage`` is a slot: keep the scaffolding (``cohort_from_directory`` →
``HabitatSpec`` → :meth:`~habit.recipes.Study.fit_predict`) and swap the
``Spec("name", {params})`` in that slot.

**Slots in this one-step example**

* ``extract_voxel_features`` — voxel field (here ``raw``)
* ``fit`` — clustering (here ``kmeans`` with auto-K)
* ``assign`` — label voxels (``nearest_centroid``)
* ``quantify`` / ``quantify2`` / … — summaries that do not change the map

**Concrete swaps** (paste over the matching ``Stage(...)``)::

   # 1) GMM instead of k-means (fixed K)
   Stage("fit", Spec("gmm", {"n_habitats": 4, "covariance_type": "full"}))

   # 2) Local entropy instead of raw intensity
   Stage(
       "extract_voxel_features",
       Spec("local_entropy", {"modalities": list(MODALITIES), "kernel_size": 3, "bins": 32}),
   )

   # 3) Scale voxels before clustering (insert after extract, before fit)
   Stage("preprocess1", Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}))
   Stage("preprocess2", Spec("minmax", {"across_features": False}))

   # 4) Mixed families — needs two series (raw T1 + entropy T2)
   from habit.spec import parse_feature_expression
   Stage(
       "extract_voxel_features",
       parse_feature_expression(
           'concat(raw("T1"), local_entropy("T2", kernel_size=3, bins=32))'
       ),
   )

Full list of names, every parameter, allowed values, and the YAML twin:
:doc:`../how_to/habitat_components` (section 1A is one series; 1B is
``concat`` / ``voxel_radiomics`` trees).

Discover the same facts in a running interpreter::

   from habit.api.plugins import list_plugins

   print([p.name for p in list_plugins("habitat_model_fitter")])

Output
------

Illustrative::

   Cohort: 2 subjects
   Cohort-level habitat_model: None
   Per-subject models: 2
   Habitat maps: 2

The Script block prints these lines. The **Draw the figures** block writes
``out/one_step_*.png``. Edit those paths if you want a different folder.

``HABIT_NO_VIEW=1`` skips the optional napari window when you run the
full script from the repository root (maintainers then copy ``out/*.png``
into the docs gallery)::

   python docs/source/examples/scripts/one_step_habitat_demo.py

Same ``demo_data/preprocessed`` + ``random_seed=42`` reproduces the habitat
**labels** (numerical contract). PNG **pixels** can differ slightly across
matplotlib / OS / DPI.

Figures
-------

These are the same files the **Draw the figures** block writes under ``out/``. The site
gallery is a copy of those PNGs (same composition; not a cropped re-plot).
Overlay uses the public 3-D default: three orthogonal panels through the
densest habitat slices. Pass ``ImageVolume`` / ``HabitatMap`` (not
``.data``) so orientation follows the volume.

.. figure:: ../_static/images/examples/one_step_overlay.png
   :alt: One-step habitat overlay
   :width: 720

   Per-subject habitats (:func:`~habit.viz.plot_habitat_overlay`, three
   orthogonal panels).

.. figure:: ../_static/images/examples/one_step_volume_fractions.png
   :alt: One-step volume fractions
   :width: 420

   Volume fractions (:func:`~habit.viz.plot_habitat_volume_fractions`).

.. figure:: ../_static/images/examples/one_step_msi_matrix.png
   :alt: One-step MSI heatmap
   :width: 420

   MSI matrix (:func:`~habit.viz.plot_msi_matrix`).

.. figure:: ../_static/images/examples/one_step_ith_summary.png
   :alt: One-step ITH summary
   :width: 520

   ITH summary (:func:`~habit.viz.plot_ith_summary`).

.. figure:: ../_static/images/examples/one_step_cluster_validation.png
   :alt: One-step cluster validation curves
   :width: 520

   Auto-K curves from this subject's ``selection_report``
   (:func:`~habit.viz.plot_cluster_validation_from_report`).

2D graph figures (node lattice + network) are on :doc:`graph_features`
(:func:`~habit.viz.plot_habitat_graph_slice`,
:func:`~habit.viz.plot_habitat_graph_network_2d`). Stream every atom
above plus those two graphs with :class:`~habit.report.Report` — see
**Stream per subject** and
``docs/source/examples/scripts/one_step_report_demo.py``.

What to read next
-----------------

* :doc:`feature_extraction` — quantify these maps
* :doc:`two_step_habitat` — shared cohort definition (typical paper pipeline)
* :doc:`direct_pooling_habitat` — pool voxels before clustering
* :doc:`persistence` — ``StudyResult.save`` vs streaming ``Report``
* :doc:`../how_to/habitat_components` — registered ``Spec`` names (Reference)
