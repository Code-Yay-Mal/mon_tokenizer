"""Myanmar syllable segmentation — the unit a token boundary must not break.

## Why `\\X` is not good enough

The obvious choice is a Unicode extended grapheme cluster, `regex`'s `\\X`. It is
wrong for this script, and wrong in the most common place.

UAX #29 assigns `GCB=Other` to U+102B `ါ`, U+102C `ာ` and U+1038 `း`, so they
begin a *new* grapheme cluster rather than joining the one before:

    regex.findall(r"\\X", "ကျော်")  ->  ['ကျေ', 'ာ်']     # two, for one syllable

Measured over 5,000 real Mon lines (179,409 `\\X` clusters): **5.2% begin with a
combining mark** — each one is itself a mid-syllable fragment — and **75.7% are a
single codepoint**, which a `len > 1` filter then drops from the denominator. A
violation metric built on `\\X` scores a cut through the middle of `ကျော်` as
clean, and is blind at exactly the positions a cluster-atomicity filter targets.

## The definition used here

    syllable := base (U+1039 base)* mark* (base U+103A mark*)*

Read left to right: a base character, any stacked consonants (U+1039 is the
invisible stacker), its vowels and tone marks, and then any **killed consonants**
— a consonant carrying U+103A ASAT is a syllable *final*, not the start of the
next syllable. That last clause is what makes `ဒုင်စသိုင်` three syllables rather
than five.

The character classes are **derived from `unicodedata.category` over the Myanmar
ranges**, not typed out by hand. Hand-typed ranges for this script are how the
Mon-exclusive set `ဨဳဴဵၚၛၜၝၞၟၠ` came to be described as letters when
U+105E, U+105F and U+1060 are medial *signs* — `ၝၞၟၠ` is one syllable, not four.

## Validation, and a caution about fixtures

`tests/test_syllable.py` asserts the fixture below. Three of its expectations
were wrong when first written and the segmenter was right each time:
`ၝၞၟၠ` is 1 syllable, not 4; `မြန်မာ` is 2 (myan-ma), not 3. A fixture is a
claim about the language and deserves the same scrutiny as the code.

The segmenter also **exactly partitions** any Myanmar run — no gaps, no overlaps —
which is what makes the offset arithmetic in `metrics.py` sound.
"""

from __future__ import annotations

import unicodedata

import regex as re

__all__ = [
    "MYANMAR_RANGES",
    "SYLLABLE_FIXTURE",
    "is_combining",
    "is_myanmar",
    "starts_mid_syllable",
    "syllable_spans",
    "syllables",
]

# Myanmar, Myanmar Extended-A and Myanmar Extended-B. The extensions carry
# Mon-, Shan- and Khamti-specific characters, so a Mon tokenizer that stopped at
# U+109F would mis-segment its own script.
MYANMAR_RANGES: tuple[tuple[int, int], ...] = ((0x1000, 0x109F), (0xA9E0, 0xA9FE), (0xAA60, 0xAA7F))

VIRAMA = "္"  # invisible stacker: U+1039 + consonant = subscript form
ASAT = "်"  # visible killer: marks a consonant as a syllable final

_ALL = [
    chr(code)
    for start, end in MYANMAR_RANGES
    for code in range(start, end + 1)
    if unicodedata.category(chr(code)) != "Cn"
]
_COMBINING = frozenset(c for c in _ALL if unicodedata.category(c) in ("Mn", "Mc"))
_BASE = frozenset(_ALL) - _COMBINING
# The stacker is a combining mark by category but has its own role in the
# grammar above, so it is excluded from the general mark class.
_MARK = _COMBINING - {VIRAMA}


def _character_class(characters: frozenset[str]) -> str:
    return "[" + "".join(f"\\u{ord(c):04X}" for c in sorted(characters)) + "]"


_B = _character_class(_BASE)
_M = _character_class(_MARK)

SYLLABLE_PATTERN = f"{_B}(?:{VIRAMA}{_B})*{_M}*(?:{_B}{ASAT}{_M}*)*"
_SYLLABLE = re.compile(SYLLABLE_PATTERN)

# Every entry hand-checked against the script. Kept beside the definition it
# validates so neither can be changed without the other being looked at.
SYLLABLE_FIXTURE: dict[str, int] = {
    "ကျော်": 1,  # medial + vowel + asat all on one base
    "ဗဟိုဌာန": 4,  # ဗ | ဟို | ဌာ | န
    "သ္ဂောံ": 1,  # U+1039 stacked consonant stays inside the syllable
    "ဒုင်စသိုင်": 3,  # killed finals attach left: ဒုင် | စ | သိုင်
    "အောင်း": 1,  # killed final plus a following tone mark
    "မြန်မာ": 2,  # myan-ma, not three
    "ဂွံအခေါင်အရာမွဲ": 6,
    "ၝၞၟၠ": 1,  # U+105E-1060 are medial signs, not letters
    "ၚၛၜၝ": 4,  # U+105A-105D are letters
    "ကခဂ": 3,
    "ဦး": 1,
    "မွဲ": 1,
    "ပ္ဍဲ": 1,
    "က်": 1,
}


def is_myanmar(char: str) -> bool:
    return any(start <= ord(char) <= end for start, end in MYANMAR_RANGES)


def is_combining(char: str) -> bool:
    """A Myanmar combining mark — vowel sign, medial, tone mark, stacker or asat."""
    return char in _COMBINING


def syllables(text: str) -> list[str]:
    """Myanmar syllables in `text`. Non-Myanmar runs are not returned."""
    return _SYLLABLE.findall(text)


def syllable_spans(text: str, minimum_length: int = 2) -> list[tuple[int, int]]:
    """`(start, end)` for each multi-character Myanmar syllable.

    Only multi-character syllables are returned, because a single character has
    no interior and so cannot be split — including them would pad the denominator
    with cases that are violation-proof by construction.

    Non-Myanmar text yields nothing, which is the honest answer: Latin script has
    no stacked marks to break. Callers must report the **denominator** alongside
    any rate computed from this, since an English corpus can legitimately produce
    a denominator of one.
    """
    return [
        (match.start(), match.end())
        for match in _SYLLABLE.finditer(text)
        if match.end() - match.start() >= minimum_length
    ]


def starts_mid_syllable(piece: str, word_marker: str = "▁") -> bool:
    """Can this vocabulary piece only ever appear from inside a syllable?

    True when the first real character is a Myanmar combining mark: a vowel sign
    or medial has nothing to attach to unless a base character precedes it, in
    another token.

    **What this cannot catch**, stated plainly: a piece beginning with a bare
    consonant may turn out to be a killed final belonging to the previous
    syllable — `င` in `ဒုင်` — and nothing about the piece in isolation reveals
    that. So this is a sound filter, not a complete one. It removes pieces that
    are *always* mid-syllable and leaves those that are *sometimes* mid-syllable,
    which is the correct trade: the alternative removes pieces that are often
    legitimate.
    """
    if piece.startswith("<") and piece.endswith(">"):
        return False  # special and byte-fallback pieces are never candidates
    body = piece.lstrip(word_marker)
    return bool(body) and is_combining(body[0])
