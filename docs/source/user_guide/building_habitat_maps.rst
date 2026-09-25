Building habitat maps
=====================

Two-step, inside-each-subject, and pooled-voxel designs are covered in
the Examples gallery under *Introductory tutorials* and *Building
habitat maps*. Spec pages declare the analysis; atomic pages check
voxel agreement with the matching Spec.

Reading a figure
----------------

Habitat overlays use ``ImageVolume`` + ``HabitatMap`` (not raw arrays
and not ``direction=``). Same colour means the same habitat id when a
cohort model was fitted.

Choosing the number of habitats
-------------------------------

When ``n_habitats`` is omitted, k-means tries every candidate in
``min_habitats`` … ``max_habitats`` and keeps the count preferred by
``validation`` (schema default ``elbow``). Report the rule and the
random seed with the results.

HABIT validation ``elbow`` is the same rule as ``kneedle``. It runs
``KneeLocator`` on k-means inertia (within-cluster sum of squares),
curve convex, direction decreasing
(Satopaa, Albrecht, Irwin, and Raghavan, 2011, *Finding a "Kneedle" in
a Haystack*, IEEE ICDCS Workshops). In two sentences: normalize ``k``
and inertia to the unit square, draw the chord from the first point to
the last, and take the ``k`` farthest from that chord. A smooth curve
often places this knee to the right of the bend a person sees.

In the clustering literature, *elbow* usually means looking at
within-cluster dispersion versus ``k`` and choosing where extra
clusters stop helping (Thorndike RL, 1953, *Who belongs in the
family?*, Psychometrika 18(4):267–276, as that early statement). Do
**not** claim Thorndike published a second-difference formula.

HABIT also documents a **discrete-curvature elbow** (visual elbow,
computed): the ``k`` that maximizes the second difference of inertia
(``argmax(diff2) + 1`` into the candidate list). That was the rule
HABIT used for ``elbow`` before v1.0. Label it *discrete-curvature
elbow*, not “Thorndike formula”.

Other k-means criteria in
:mod:`habit.kernels.cluster_selection` — maximize: ``silhouette``,
``calinski_harabasz``, ``gap``; minimize: ``davies_bouldin``; knee:
``elbow`` / ``kneedle`` / ``inertia``. GMM also supports minimize
``bic`` / ``aic`` and ``bic_elbow`` (Prior 2024 BIC-slope rule).
Criteria disagree on the same curve; the gallery page
:doc:`/auto_examples/03_clustering/plot_03_clustering_algorithm`
prints a comparison table and marks both the Kneedle ``k`` and the
discrete-curvature ``k`` on the inertia panel.
