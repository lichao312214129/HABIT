# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Contract tests for ``MLSpec.steps``, the single ordered table-step list.

``MLSpec`` used to express step order through three fixed fields
(``pre_preprocessing_feature_selectors`` -> ``table_preprocessors`` ->
``feature_selectors``), which allowed a selector to sit only before all
preprocessing or after all of it. ``steps`` replaces them with one ordered
list; the three are deprecated aliases for the whole of v1.x.

Two things must hold at once and are what these tests lock:

* the deprecated layout keeps producing BYTE-IDENTICAL payloads and therefore
  byte-identical fingerprints, because every provenance record and golden
  baseline HABIT has published hashes that payload;
* the new layout can express an order the old one could not.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import pytest

from habit.exceptions import HABITAPIError
from habit.spec.specs import MLSpec, Spec


def _legacy_spec() -> MLSpec:
    """
    Build a spec through the deprecated three-chain layout.

    Returns:
        MLSpec: The chains fold into ``steps`` as
        ``variance -> zscore -> correlation``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return MLSpec(
            name="ml_cv",
            classifier=Spec(name="LogisticRegression", params={"C": 1.0}),
            pre_preprocessing_feature_selectors=(
                Spec(name="variance", params={"top_k": 20}),
            ),
            table_preprocessors=(Spec(name="zscore"),),
            feature_selectors=(Spec(name="correlation", params={"threshold": 0.8}),),
            random_seed=42,
        )


def _interleaved_spec() -> MLSpec:
    """
    Build a spec whose order the deprecated layout cannot express.

    Returns:
        MLSpec: ``zscore -> variance -> minmax -> lasso`` -- two
        preprocessors with a selector between them, which no assignment to
        three fixed slots can produce.
    """
    return MLSpec(
        name="interleaved",
        classifier=Spec(name="LogisticRegression"),
        steps=(
            Spec(name="zscore"),
            Spec(name="variance", params={"threshold": 0.01}),
            Spec(name="minmax"),
            Spec(name="lasso"),
        ),
        random_seed=42,
    )


# ---------------------------------------------------------------------------
# The deprecated layout still works, and says so
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_deprecated_chains_fold_into_steps_in_the_documented_order() -> None:
    """
    The three chains concatenate as pre -> preprocessors -> post.

    That order is the v0.1 execution order, and it is the whole reason the
    deprecated layout can be translated at all: the slots encoded a fixed
    sequence, so the sequence is recoverable.
    """
    assert [entry.name for entry in _legacy_spec().steps] == [
        "variance",
        "zscore",
        "correlation",
    ]


@pytest.mark.unit
def test_declaring_a_deprecated_chain_warns() -> None:
    """The deprecation is announced, not silent."""
    with pytest.warns(DeprecationWarning, match="deprecated"):
        MLSpec(
            name="warns",
            classifier=Spec(name="SVM"),
            table_preprocessors=(Spec(name="zscore"),),
        )


@pytest.mark.unit
def test_declaring_steps_does_not_warn() -> None:
    """The supported layout is quiet."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        MLSpec(
            name="quiet",
            classifier=Spec(name="SVM"),
            steps=(Spec(name="zscore"),),
        )


@pytest.mark.unit
def test_a_spec_with_no_table_steps_stays_quiet_and_legacy_shaped() -> None:
    """
    Declaring nothing declares nothing deprecated.

    The empty spec keeps the deprecated payload shape (see the fingerprint
    test below) but must not be accused of using a deprecated field it never
    named.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        spec = MLSpec(name="bare", classifier=Spec(name="SVM"))
    assert spec.steps == ()
    assert spec.declares_deprecated_chains is True


# ---------------------------------------------------------------------------
# Fingerprint stability (hard constraint: no published fingerprint moves)
# ---------------------------------------------------------------------------


#: The payload a legacy-shaped ``MLSpec`` produced before ``steps`` existed,
#: transcribed from v1.0.4 (commit ``ced2e583``). Frozen here rather than
#: recomputed, so a change to ``to_dict`` cannot quietly redefine "unchanged".
_FROZEN_LEGACY_PAYLOAD: Dict[str, Any] = {
    "name": "ml_cv",
    "version": "1.0",
    "pre_preprocessing_feature_selectors": [
        {"name": "variance", "params": {"top_k": 20}, "version": "1.0"}
    ],
    "table_preprocessors": [{"name": "zscore", "params": {}, "version": "1.0"}],
    "feature_selectors": [
        {"name": "correlation", "params": {"threshold": 0.8}, "version": "1.0"}
    ],
    "classifier": {
        "name": "LogisticRegression",
        "params": {"C": 1.0},
        "version": "1.0",
    },
    "metrics": [],
    "random_seed": 42,
}


@pytest.mark.unit
def test_legacy_payload_is_byte_identical_to_the_pre_steps_shape() -> None:
    """
    Hard constraint: adding ``steps`` moves no existing fingerprint.

    ``to_dict`` must emit the three deprecated keys and NOT a ``steps`` key
    for a legacy-declared spec. Every ``run_manifest.json`` in
    ``tests/golden/baseline/`` records this payload and the hash of it, so an
    extra key here is a silent baseline change.
    """
    payload = _legacy_spec().to_dict()
    assert payload == _FROZEN_LEGACY_PAYLOAD
    assert "steps" not in payload


@pytest.mark.unit
def test_empty_spec_payload_keeps_the_deprecated_keys() -> None:
    """
    A spec with no table steps also keeps the old shape.

    Otherwise the majority of published specs -- classifier-only ones --
    would change fingerprint for a feature they do not use.
    """
    payload = MLSpec(name="bare", classifier=Spec(name="SVM")).to_dict()
    assert set(payload) == {
        "name",
        "version",
        "pre_preprocessing_feature_selectors",
        "table_preprocessors",
        "feature_selectors",
        "classifier",
        "metrics",
        "random_seed",
    }


@pytest.mark.unit
def test_steps_payload_carries_steps_and_no_deprecated_keys() -> None:
    """
    The new layout serialises as one ordered list.

    Emitting both layouts would make the payload ambiguous about which one
    is the pipeline, and would put every step in the fingerprint twice.
    """
    payload = _interleaved_spec().to_dict()
    assert set(payload) == {
        "name",
        "version",
        "steps",
        "classifier",
        "metrics",
        "random_seed",
    }
    assert [entry["name"] for entry in payload["steps"]] == [
        "zscore",
        "variance",
        "minmax",
        "lasso",
    ]


@pytest.mark.unit
def test_both_layouts_round_trip_through_the_dict_form() -> None:
    """A spec reconstructed from its payload is the same spec."""
    for spec in (_legacy_spec(), _interleaved_spec(), MLSpec("bare", Spec("SVM"))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            restored = MLSpec.from_dict(spec.to_dict())
        assert restored == spec
        assert restored.fingerprint() == spec.fingerprint()


@pytest.mark.unit
def test_step_order_changes_the_fingerprint() -> None:
    """
    Order is scientific, so it is part of the hash.

    ``variance`` before ``zscore`` and after it compute different things
    (z-scoring makes every variance 1.0); a provenance record that could not
    tell them apart would be useless.
    """
    forward = _interleaved_spec()
    swapped = MLSpec(
        name=forward.name,
        classifier=forward.classifier,
        steps=(forward.steps[1], forward.steps[0], *forward.steps[2:]),
        random_seed=forward.random_seed,
    )
    assert [entry.name for entry in swapped.steps][:2] == ["variance", "zscore"]
    assert swapped.fingerprint() != forward.fingerprint()


@pytest.mark.unit
def test_the_two_layouts_fingerprint_differently_even_when_equivalent() -> None:
    """
    Declaring the same pipeline both ways yields two different hashes.

    This is intentional. The alternative -- normalising both layouts onto one
    payload -- would have to pick a shape, and picking ``steps`` moves every
    fingerprint ever published. The pipelines are identical; the DOCUMENTS
    are not, and a fingerprint identifies a document.
    """
    legacy = _legacy_spec()
    same_order = MLSpec(
        name=legacy.name,
        classifier=legacy.classifier,
        steps=legacy.steps,
        random_seed=legacy.random_seed,
    )
    assert [entry.name for entry in same_order.steps] == [
        entry.name for entry in legacy.steps
    ]
    assert same_order.fingerprint() != legacy.fingerprint()


# ---------------------------------------------------------------------------
# Mixing the layouts
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_contradictory_layouts_are_rejected() -> None:
    """
    Declaring two different pipelines is an error, not a precedence puzzle.

    Resolving it by preferring one field would run a pipeline the document
    also says it is not running.
    """
    with pytest.raises(HABITAPIError, match="disagree"):
        MLSpec(
            name="ambiguous",
            classifier=Spec(name="SVM"),
            steps=(Spec(name="zscore"),),
            table_preprocessors=(Spec(name="minmax"),),
        )


@pytest.mark.unit
def test_a_consistent_pair_is_accepted_so_replace_keeps_working() -> None:
    """
    ``dataclasses.replace`` re-supplies every field, including both layouts.

    A legacy spec passed through ``replace`` (which the recipes do to fold a
    seed override in) therefore arrives carrying chains AND the ``steps``
    they folded into. That is consistent, not contradictory, and must not
    raise.
    """
    import dataclasses

    original = _legacy_spec()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        reseeded = dataclasses.replace(original, random_seed=7)
    assert reseeded.random_seed == 7
    assert [entry.name for entry in reseeded.steps] == [
        entry.name for entry in original.steps
    ]
    assert "steps" not in reseeded.to_dict()


@pytest.mark.unit
def test_from_dict_rejects_a_document_declaring_both_layouts() -> None:
    """A hand-written YAML cannot smuggle two pipelines past the reader."""
    with pytest.raises(HABITAPIError):
        MLSpec.from_dict(
            {
                "name": "ambiguous",
                "classifier": {"name": "SVM"},
                "steps": [{"name": "zscore"}],
                "table_preprocessors": [{"name": "minmax"}],
            }
        )


@pytest.mark.unit
def test_every_step_must_be_a_spec() -> None:
    """A bare string in ``steps`` is rejected like one in the old chains."""
    with pytest.raises(HABITAPIError, match="steps"):
        MLSpec(
            name="broken",
            classifier=Spec(name="SVM"),
            steps=("zscore",),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Methods prose
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_methods_paragraph_states_the_ordered_chain() -> None:
    """The prose names every step, in execution order."""
    text = _interleaved_spec().describe_methods()
    assert "an ordered table pipeline of" in text
    assert (
        text.index("zscore")
        < text.index("variance")
        < text.index("minmax")
        < text.index("lasso")
    )


@pytest.mark.unit
def test_methods_paragraph_of_a_legacy_spec_is_unchanged() -> None:
    """The deprecated layout still renders its three stage phrases."""
    text = _legacy_spec().describe_methods()
    assert (
        text.index("pre-preprocessing feature selection")
        < text.index("table preprocessing")
        < text.index("feature selection with correlation")
    )
    assert "an ordered table pipeline of" not in text
