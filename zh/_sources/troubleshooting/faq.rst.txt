FAQ
===

Install
-------

**``habit`` not found** — you are not in the conda env. On Windows open
**Anaconda Prompt** (not plain CMD), run ``conda activate habit`` until
the prompt shows ``(habit)``, then ``habit --version``.
See :doc:`../tutorial/installation`.

**``pip install`` fails / mirror missing package** — use official PyPI::

   pip install habitat-analysis -i https://pypi.org/simple

**PyRadiomics on Windows** — do **not** use bare ``pip install pyradiomics``
(broken sdist). Install the wheel from Release ``v1.0.2`` matching your
Python, e.g. 3.10::

   pip install https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/pyradiomics-3.1.0-cp310-cp310-win_amd64.whl

**napari / ``PluginManifest`` / pydantic errors**::

   pip install "pydantic>=2.12,<3"
   pip install "napari[pyqt5]" "npe2>=0.8.2" "pydantic!=2.11.*,>=2.8,<3" -i https://pypi.org/simple

**napari window flashes then closes** — keep ``show=True`` (default); the
call blocks until you close the window.

**``OptionalDependencyError``** — the message includes the
``pip install ...`` command for the missing package.

**Parquet / pyarrow** — ``pip install pyarrow``, or set
``habitats_results_format: csv`` in YAML.

Run
---

**Path / file errors** — run from the project root (folder that contains
``config/``). Check ``data_dir`` / ``out_dir`` (and absolute paths if unsure).

**Command failed** — ``habit <cmd> --help``; read ``processing.log`` in the
output folder; validate with ``habit check-config -c <yaml>``.

**YAML changes ignored** — confirm the file passed with ``-c``; CLI
``--mode`` overrides YAML ``run_mode``.

Data
----

ROI: NIfTI aligned with images, or resample the mask
(:doc:`/auto_examples/06_manipulating_images/plot_02_nifti`).
Directory, loose files, SimpleITK, or NumPy:
:doc:`/auto_examples/06_manipulating_images/index`.
DICOM: :doc:`../how_to/preprocess`.

Habitat maps
------------

**Why is my habitat map one big block?** Usually the features were
clustered across patients without subject-level normalization. MR
intensities are arbitrary units, so a patient whose scan is globally
darker or brighter than the training patients falls almost entirely
into the darkest or brightest centroid. Put subject-level preprocessing
before ``pool`` -- for example ``Spec("winsorize", {"winsor_limits":
[0.05, 0.05]})`` then ``Spec("minmax")`` -- as in
:doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`. The reason,
and what cohort-level preprocessing does and does not fix, is on
:doc:`/tutorial/concepts`. Other causes: too few habitats (``fit`` chose
2), or a very small ROI.

**Why do two runs or two pages give different habitat maps?** Two cases.
(1) The same regions with different numbers: clustering numbers its
habitats in arbitrary order, so "habitat 2" of one fit can be "habitat
3" of another. The maps agree after matching the labels (Hungarian
prototype matching):
:doc:`/auto_examples/05_validation_and_reuse/plot_02_match_same_subject`,
:doc:`/auto_examples/05_validation_and_reuse/plot_03_match_two_subjects`,
or :doc:`/auto_examples/05_validation_and_reuse/plot_04_match_group`.
Two-step and pooled-voxel fits already share one model and do not need matching.
(2) Genuinely different regions: the two runs used a different
definition (other stages, preprocessing, number of supervoxels, seed) or
a different training cohort. Compare the two stage lists and the
``cohort digest`` in ``model.summary()``. With the same spec, seed and
training cohort, HABIT gives identical labels voxel by voxel, whatever
the backend.

**Can HABIT analyse a peritumoral region?** Yes, if you supply the
peritumoral mask yourself; HABIT has no peritumoral-ring generator.
Masks are looked up by name: a :class:`~habit.contracts.Subject` holds
``masks`` (ROI name to mask), and ``subject.mask(roi_name)`` returns one.
Save your ring as another ROI folder, for example
``masks/<subject>/peritumoral/<file>``, load it with
``cohort_from_directory(DATA, modalities=MODALITIES, roi="peritumoral")``,
and name the same ROI in the extract stage
(``Spec("raw", {"modalities": [...], "roi": "peritumoral"})``). Keep the
ring and the tumour as separate analyses unless you merge the masks
yourself. Loading routes: :doc:`/auto_examples/06_manipulating_images/index`.

Support: |link_github_issues| · lichao19870617@163.com
