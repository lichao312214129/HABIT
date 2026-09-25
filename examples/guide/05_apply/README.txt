5. Apply a saved model
======================

**Background.** A fitted habitat model can be saved and reused, so later or
external subjects get the same habitat ids as the training cohort.
**Purpose.** Save the model once, reload it, and label new subjects with the
training centroids without refitting.

Train the habitat definition, write a ``.habitatmodel`` file, and label
later subjects without fitting again.

:doc:`/auto_examples/05_apply/plot_04_apply_saved_model`
