"""The Myanmar syllable segmenter, which the violation metric is built on.

Getting this wrong is not a small error. The first version of the metric used
`regex`'s `\\X` (Unicode extended grapheme cluster), and UAX #29 puts a cluster
break *before* U+102C `ာ` — so a cut through the middle of `ကျော်` scored as
clean, and the metric was blind at the script's most common syllable break.
"""

from __future__ import annotations

import json
import unicodedata

import pytest
import regex as re

from mon_tokenizer.syllable import (
    ASAT,
    DOT_BELOW,
    MYANMAR_RANGES,
    SYLLABLE_FIXTURE,
    VIRAMA,
    is_combining,
    starts_mid_syllable,
    syllable_spans,
    syllables,
)
from mon_tokenizer.tokenizer import default_model_path

ARTIFACT = default_model_path()
MYANMAR_RUN = re.compile("[" + "".join(f"\\u{a:04X}-\\u{b:04X}" for a, b in MYANMAR_RANGES) + "]+")
# A piece that opens on `င်္` began inside the previous syllable's final.
KINZI_AT_START = re.compile(rf"^.{DOT_BELOW}?{ASAT}{VIRAMA}")


@pytest.mark.parametrize(("text", "expected"), SYLLABLE_FIXTURE.items())
def test_the_fixture_segments_as_expected(text: str, expected: int):
    """Every entry hand-checked against the script.

    Three of these expectations were wrong when first written and the segmenter
    was right each time — `ၝၞၟၠ` is one syllable because U+105E-1060 are medial
    signs rather than letters, and `မြန်မာ` is two (myan-ma). A fixture is a claim
    about the language and gets the same scrutiny as the code.
    """
    assert len(syllables(text)) == expected, syllables(text)


@pytest.mark.parametrize("text", SYLLABLE_FIXTURE)
def test_segmentation_is_an_exact_partition(text: str):
    """No gaps, no overlaps, nothing dropped.

    `metrics.syllable_violations` compares syllable spans against token
    boundaries computed from the same string. If the segmenter skipped a
    character the two would index different positions and every violation count
    would be quietly wrong.
    """
    assert "".join(syllables(text)) == text


def test_dot_below_before_asat_stays_in_the_final():
    """NFC stores `င့်` as U+1004 U+1037 U+103A: dot below (class 7) sorts first.

    The final clause once required the asat directly after the consonant, so the
    normalized form split as `['ကြော', 'င့်']`.
    """
    word = unicodedata.normalize("NFC", "ကြောင့်")
    assert word.endswith("\u1004" + DOT_BELOW + ASAT)
    assert syllables(word) == [word]


def test_kinzi_stacker_belongs_to_the_killed_final():
    """Kinzi is asat followed by U+1039; the stacker was once in no syllable.

    `မင်္ဂလာ` came back as `['မင်', 'ဂ', 'လာ']`, dropping U+1039 and breaking the
    partition that the violation metric's offsets rely on.
    """
    assert syllables("မင်္ဂလာ") == ["မင်္", "ဂ", "လာ"]
    assert syllables("အၚ်္ဂလိက်") == ["အၚ်္", "ဂ", "လိက်"]


def test_segmentation_partitions_real_text_from_the_vocabulary():
    """The partition holds over text the corpus actually contains.

    The shipped vocabulary is the only corpus-derived text in this repository:
    every piece is a substring of the training data. A run is checked when it
    starts at a syllable start — not on a combining mark and not on a kinzi,
    both of which mean the piece was cut mid-syllable — and does not end on a
    dangling stacker whose consonant is in another piece.
    """
    vocab = json.loads(ARTIFACT.read_text(encoding="utf-8"))["model"]["vocab"]
    runs = [run for piece, _ in vocab for run in MYANMAR_RUN.findall(piece)]
    checked = [
        run
        for run in runs
        if not is_combining(run[0]) and not run.endswith(VIRAMA) and not KINZI_AT_START.match(run)
    ]
    assert len(checked) > 40_000, "the sample must be large enough to mean something"
    assert any(f"{ASAT}{VIRAMA}" in run for run in checked), "the sample must include kinzi"
    broken = [run for run in checked if "".join(syllables(run)) != run]
    assert broken == [], broken[:10]


def test_killed_finals_attach_to_the_preceding_syllable():
    """`ဒုင်` is one syllable, not two.

    A consonant carrying U+103A ASAT is a syllable *final*. Without that rule the
    segmenter returns five syllables for `ဒုင်စသိုင်` instead of three, and the
    violation denominator inflates with boundaries that are not breaks.
    """
    assert syllables("ဒုင်စသိုင်") == ["ဒုင်", "စ", "သိုင်"]


def test_stacked_consonants_stay_inside_one_syllable():
    """U+1039 is the invisible stacker: `သ္ဂောံ` is a single unit."""
    assert syllables("သ္ဂောံ") == ["သ္ဂောံ"]


def test_the_vowel_sign_that_grapheme_clusters_get_wrong():
    """The specific case that motivated replacing `\\X`.

    `regex.findall(r"\\X", "ကျော်")` returns `['ကျေ', 'ာ်']`.
    """
    assert syllables("ကျော်") == ["ကျော်"]


def test_single_character_syllables_are_excluded_from_spans():
    """They have no interior, so they cannot be split.

    Including them would pad the denominator with cases that are violation-proof
    by construction — 75.7% of `\\X` clusters were single-codepoint, which is one
    of the reasons the old metric read low.
    """
    assert syllable_spans("ကခဂ") == []
    assert len(syllable_spans("ကျော်")) == 1


def test_latin_text_yields_no_syllables():
    """Not a failure — Latin has no stacked marks to break.

    Callers must report the denominator with any rate computed from this, since
    an English stratum legitimately produces zero.
    """
    assert syllable_spans("hello world") == []


def test_combining_marks_are_recognised_across_all_three_myanmar_ranges():
    """Medial signs live in Extended-A, which a U+1000-109F check would miss."""
    assert is_combining("ိ") and is_combining("ာ")
    assert is_combining("ၞ"), "U+105E is a Mon medial sign and must count as combining"
    assert not is_combining("က")


def test_starts_mid_syllable_flags_leading_marks_only():
    """The filter's contract, including what it deliberately cannot catch."""
    assert starts_mid_syllable("ာ")
    assert starts_mid_syllable("▁ိုက်")
    assert not starts_mid_syllable("ကျော်")
    assert not starts_mid_syllable("hello")
    # Special and byte pieces are never candidates — removing them would leave
    # text with no representation at all.
    assert not starts_mid_syllable("<unk>")
    assert not starts_mid_syllable("<0x41>")
    # Documented limitation: a bare consonant may turn out to be a killed final,
    # and nothing about the piece in isolation reveals that.
    assert not starts_mid_syllable("င်"), "cannot be detected without surrounding context"
