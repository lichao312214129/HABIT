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
"""Strict parser for the v1 feature-tree expression language.

The expression language is the compact spelling of a feature spec tree --
the same tree the structured YAML form (``name`` / ``params`` / nested
``children``) describes. A string like::

    concat(raw("T1"), ratio(raw("T1"), raw("T2"), as_="t1_over_t2"))

parses into exactly the :class:`~habit.spec.specs.Spec` tree the structured
form would produce, so the two spellings share one canonical representation
(and one fingerprint).

Strictness is the point: the v0 mini-language guessed whether a bare token
was a modality name or a parameter reference, which made typos silently
change results. Here modality names are QUOTED strings and parameters are
explicit ``key=value`` pairs; a bare identifier where a value is expected
fails with a message that says so.

Grammar::

    node     := IDENT [ "(" [ arg ("," arg)* ] ")" ]
    arg      := node                    # nested call  -> child node
              | STRING                  # quoted       -> modality (leaf) or raw child (combiner)
              | IDENT "=" literal       # kwarg        -> node parameter
    literal  := STRING | NUMBER | BOOLEAN | NULL | "[" [literal ("," literal)*] "]"]

Positional rules:

- A node whose arguments include nested calls is a combiner node; every
  quoted string among its arguments becomes an implicit ``raw`` leaf child
  (``concat("T1", "T2")`` is the short form of
  ``concat(raw("T1"), raw("T2"))``).
- A node with only quoted positional strings is a leaf: one string maps to
  ``modality``, several to ``modalities``.
- ``weights=[...]`` on a combiner node is positional sugar: the parser keys
  the list by each child's source label (``as_`` > ``modality`` > name), so
  the canonical spec always carries the dict form.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from habit.exceptions import HABITAPIError
from habit.spec.specs import Spec

__all__ = ["parse_feature_expression"]

# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

_IDENT_START = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
_IDENT_CHARS = _IDENT_START + "0123456789"
_NUMBER_CHARS = "0123456789+-.eE"

_Token = Tuple[str, Any, int]  # (kind, value, position)


def _tokenize(text: str) -> List[_Token]:
    """
    Split an expression into tokens.

    Args:
        text: The expression source.

    Returns:
        Token list of ``(kind, value, position)`` triples; kinds are
        ``ident``, ``string``, ``number`` and the single-character
        punctuation tokens.

    Raises:
        HABITAPIError: On an unterminated string or an unexpected character.
    """
    tokens: List[_Token] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char in "(),=[]":
            tokens.append((char, char, index))
            index += 1
            continue
        if char == "=":
            tokens.append(("=", char, index))
            index += 1
            continue
        if char in "\"'":
            quote = char
            start = index
            index += 1
            pieces: List[str] = []
            while index < length and text[index] != quote:
                if text[index] == "\\" and index + 1 < length:
                    # Minimal escapes: keep the escaped character verbatim.
                    pieces.append(text[index + 1])
                    index += 2
                else:
                    pieces.append(text[index])
                    index += 1
            if index >= length:
                raise HABITAPIError(
                    f"Unterminated string starting at position {start} in "
                    f"expression {text!r}."
                )
            index += 1  # closing quote
            tokens.append(("string", "".join(pieces), start))
            continue
        if char in _IDENT_START:
            start = index
            while index < length and text[index] in _IDENT_CHARS:
                index += 1
            tokens.append(("ident", text[start:index], start))
            continue
        if char.isdigit() or (
            char in "+-" and index + 1 < length and text[index + 1].isdigit()
        ):
            start = index
            while index < length and text[index] in _NUMBER_CHARS:
                index += 1
            tokens.append(("number", text[start:index], start))
            continue
        raise HABITAPIError(
            f"Unexpected character {char!r} at position {index} in "
            f'expression {text!r}. Quote modality names, e.g. raw("T1").'
        )
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _Cursor:
    """One-token lookahead over the token stream."""

    def __init__(self, tokens: List[_Token], source: str) -> None:
        self._tokens = tokens
        self._source = source
        self._at = 0

    def peek(self) -> Optional[_Token]:
        """Return the current token, or None at end of input."""
        return self._tokens[self._at] if self._at < len(self._tokens) else None

    def next(self) -> _Token:
        """Consume and return the current token."""
        token = self.peek()
        if token is None:
            raise HABITAPIError(f"Unexpected end of expression {self._source!r}.")
        self._at += 1
        return token

    def expect(self, kind: str) -> _Token:
        """Consume one token of ``kind`` or raise a positioned error."""
        token = self.peek()
        if token is None or token[0] != kind:
            got = "end of expression" if token is None else repr(token[1])
            where = len(self._source) if token is None else token[2]
            raise HABITAPIError(
                f"Expected {kind!r} at position {where} in expression "
                f"{self._source!r}; got {got}."
            )
        return self.next()


def _coerce_number(text: str, position: int, source: str) -> Union[int, float]:
    """Convert a numeric literal to int when integral, else float."""
    try:
        value = float(text)
    except ValueError:
        raise HABITAPIError(
            f"Malformed number {text!r} at position {position} in "
            f"expression {source!r}."
        ) from None
    return (
        int(value)
        if value.is_integer() and "." not in text and "e" not in text.lower()
        else value
    )


_IDENT_LITERALS: Dict[str, Any] = {
    "true": True,
    "false": False,
    "null": None,
}


def _parse_literal(cursor: _Cursor) -> Any:
    """Parse one kwarg literal: string, number, boolean, null or flat list."""
    token = cursor.peek()
    if token is None:
        raise HABITAPIError("Unexpected end of expression in a parameter value.")
    kind, value, position = token
    if kind == "string":
        cursor.next()
        return value
    if kind == "number":
        cursor.next()
        return _coerce_number(value, position, cursor._source)
    if kind == "ident" and str(value).lower() in _IDENT_LITERALS:
        cursor.next()
        return _IDENT_LITERALS[str(value).lower()]
    if kind == "[":
        cursor.next()
        items: List[Any] = []
        token = cursor.peek()
        if token is not None and token[0] != "]":
            items.append(_parse_literal(cursor))
            token = cursor.peek()
            while token is not None and token[0] == ",":
                cursor.next()
                items.append(_parse_literal(cursor))
                token = cursor.peek()
        cursor.expect("]")
        return items
    raise HABITAPIError(
        f"Expected a parameter value (quoted string, number, boolean or "
        f"list) at position {position} in expression {cursor._source!r}; "
        f"got {value!r}. Bare names are not values -- quote modality "
        'names, e.g. raw("T1").'
    )


def _parse_node(cursor: _Cursor) -> Spec:
    """
    Parse one node: a bare leaf name or a call with arguments.

    Args:
        cursor: Token cursor positioned at the node's identifier.

    Returns:
        The node as a Spec; children are nested under ``params['children']``.
    """
    kind, name, position = cursor.next()
    if kind != "ident":
        raise HABITAPIError(
            f"Expected a component name at position {position} in "
            f"expression {cursor._source!r}; got {name!r}."
        )
    token = cursor.peek()
    if token is None or token[0] != "(":
        # Bare name, e.g. ``volume``: a leaf with no parameters.
        return Spec(name=str(name), params={})
    cursor.next()  # consume "("

    positional: List[Tuple[str, Any]] = []  # ("call", Spec) | ("string", str), in order
    kwargs: Dict[str, Any] = {}
    token = cursor.peek()
    if token is not None and token[0] != ")":
        while True:
            token = cursor.peek()
            if token is None:
                raise HABITAPIError(
                    f"Unterminated argument list of {name!r} in expression "
                    f"{cursor._source!r}."
                )
            if token[0] == "ident":
                # Either a nested call or a ``key=value`` kwarg; a bare
                # identifier in any other role is the classic v0 ambiguity
                # and is rejected.
                lookahead = (
                    cursor._tokens[cursor._at + 1]
                    if cursor._at + 1 < len(cursor._tokens)
                    else None
                )
                if lookahead is not None and lookahead[0] == "=":
                    key = str(cursor.next()[1])
                    cursor.next()  # consume "="
                    if key in kwargs:
                        raise HABITAPIError(f"{name!r}: duplicate parameter {key!r}.")
                    kwargs[key] = _parse_literal(cursor)
                elif lookahead is not None and lookahead[0] == "(":
                    positional.append(("call", _parse_node(cursor)))
                else:
                    raise HABITAPIError(
                        f"Bare name {token[1]!r} at position {token[2]} in "
                        f"expression {cursor._source!r}. Modality names "
                        f'must be quoted (raw("{token[1]}")); parameter '
                        "values use key=value."
                    )
            elif token[0] == "string":
                cursor.next()
                positional.append(("string", token[1]))
            else:
                raise HABITAPIError(
                    f"Expected a child node, a quoted modality name or a "
                    f"key=value parameter at position {token[2]} in "
                    f"expression {cursor._source!r}; got {token[1]!r}."
                )
            token = cursor.peek()
            if token is not None and token[0] == ",":
                cursor.next()
                continue
            break
    cursor.expect(")")
    return _assemble_node(str(name), positional, kwargs, cursor._source)


def _source_label(spec: Spec) -> str:
    """Return one child node's source label: ``as_`` > ``modality`` > name."""
    alias = spec.params.get("as_")
    if alias:
        return str(alias)
    modality = spec.params.get("modality")
    if modality:
        return str(modality)
    return spec.name


def _assemble_node(
    name: str,
    positional: List[Tuple[str, Any]],
    kwargs: Dict[str, Any],
    source: str,
) -> Spec:
    """
    Assemble one node's Spec from its positional arguments and kwargs.

    Args:
        name: Component name.
        positional: ``("call", Spec)`` / ``("string", modality)`` entries in
            argument order.
        kwargs: Explicit ``key=value`` parameters.
        source: Full expression text for error messages.

    Returns:
        The assembled Spec.
    """
    params: Dict[str, Any] = {}
    calls = [item[1] for item in positional if item[0] == "call"]
    strings = [item[1] for item in positional if item[0] == "string"]

    if calls:
        # Combiner node: quoted strings become implicit ``raw`` children,
        # preserving the argument order of the whole positional list.
        # Children are stored as bare ``name``/``params`` payloads (no
        # ``version`` key), so the expression spelling and the structured
        # YAML spelling fingerprint identically.
        children: List[Dict[str, Any]] = []
        for kind, value in positional:
            if kind == "call":
                children.append({"name": value.name, "params": value.params})
            else:
                children.append({"name": "raw", "params": {"modality": value}})
        params["children"] = children
    elif strings:
        # Leaf node: one modality or an explicit modality list.
        if len(strings) == 1:
            params["modality"] = strings[0]
        else:
            params["modalities"] = list(strings)

    weights = kwargs.get("weights")
    if calls and isinstance(weights, list):
        # Positional sugar: key the weight list by each child's source
        # label so the canonical spec always carries the dict form.
        if len(weights) != len(calls):
            raise HABITAPIError(
                f"{name!r}: weights list has {len(weights)} entries but "
                f"{len(calls)} children in expression {source!r}."
            )
        kwargs["weights"] = {
            _source_label(child): weight for child, weight in zip(calls, weights)
        }
    params.update(kwargs)
    return Spec(name=name, params=params)


def parse_feature_expression(expression: str) -> Spec:
    """
    Parse one feature-tree expression into its canonical Spec tree.

    This is the strict v1 spelling: quoted modality names, explicit
    ``key=value`` parameters, nested calls for combiner children. The v0
    bare-token mini-language is handled by the legacy config adapter
    instead; this parser deliberately rejects it so ambiguous expressions
    fail loudly rather than silently guessing.

    Args:
        expression: The expression, e.g.
            ``'concat(raw("T1"), ratio(raw("T1"), raw("T2"), as_="r"))'``.

    Returns:
        The root node's Spec; combiner children nest under
        ``params['children']`` as ``Spec.to_dict()`` payloads.

    Raises:
        HABITAPIError: On any malformed or ambiguous input.
    """
    text = expression.strip()
    if not text:
        raise HABITAPIError("Empty feature expression.")
    cursor = _Cursor(_tokenize(text), text)
    spec = _parse_node(cursor)
    leftover = cursor.peek()
    if leftover is not None:
        kind, value, position = leftover
        raise HABITAPIError(
            f"Unexpected {value!r} at position {position} after the "
            f"complete expression {text!r}."
        )
    return spec
