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

Examples use k-means with an elbow (or other real metrics) over a
candidate range. Report the rule and the seed with the results.
