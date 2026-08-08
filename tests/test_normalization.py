"""One normalization spec, two implementations, asserted equal.

`normalize_text` is pure Python; `hf_normalizer()` builds the Rust pipeline that
is serialized into `tokenizer.json` and actually runs on every `encode`. They
have to agree, and "`\\p{Zs}` in the Rust regex crate matches Python's
`unicodedata.category(c) == "Zs"`" is an assumption worth checking rather than
believing.

Three copies of this logic had already drifted before v2: mon-lm's trainer,
mon_OCR's charset module, and this package's `encode` — which applied none of it.
"""

from __future__ import annotations

import sys
import unicodedata

import pytest

from mon_tokenizer import MonTokenizer
from mon_tokenizer.normalization import INVISIBLE_CHARS, hf_normalizer, normalize_text


@pytest.fixture(scope="module")
def rust():
    return hf_normalizer()


def test_the_two_implementations_agree_on_every_codepoint(rust):
    """All of Unicode, not a sample.

    Cheap enough to be exhaustive, and exhaustive is the only way to be sure
    about a character-class equivalence. Surrogates are excluded because they are
    not valid scalar values.
    """
    disagreements = []
    for code in range(sys.maxunicode + 1):
        if 0xD800 <= code <= 0xDFFF:
            continue
        char = chr(code)
        if normalize_text(char) != rust.normalize_str(char):
            disagreements.append(f"U+{code:04X}")
        if len(disagreements) > 10:
            break
    assert not disagreements, f"python and rust normalizers disagree on {disagreements}"


def test_invisible_characters_are_removed(rust):
    for char in INVISIBLE_CHARS:
        assert normalize_text(f"a{char}b") == "ab", f"U+{ord(char):04X} survived"
        assert rust.normalize_str(f"a{char}b") == "ab"


def test_unicode_space_separators_fold_to_a_plain_space(rust):
    """Adopted from mon_OCR, which measured the cost of not doing it.

    It found 20 U+00A0 in its training labels and 9 in validation, each rendering
    as a visible gap in the image while being dropped from the target.
    """
    separators = [chr(c) for c in range(sys.maxunicode) if unicodedata.category(chr(c)) == "Zs"]
    assert len(separators) > 10, "sanity: Zs should not be nearly empty"
    for char in separators:
        assert normalize_text(f"a{char}b") == "a b"
        assert rust.normalize_str(f"a{char}b") == "a b"


def test_runs_of_spaces_are_preserved(rust):
    """The v1 defect, held open.

    SentencePiece collapses runs of spaces by default, which was the sole cause of
    v1's round-trip loss — 18,603 lost U+0020, zero Mon characters. Whitespace is
    content.
    """
    assert normalize_text("a   b") == "a   b"
    assert rust.normalize_str("a   b") == "a   b"


def test_nfc_composes_the_one_decomposable_myanmar_letter(rust):
    """`ဦ` U+1026 is the only codepoint in the Myanmar block with a canonical
    decomposition, and it is common — it opens the honorific `ဦး`."""
    decomposed = unicodedata.normalize("NFD", "ဦး")
    assert decomposed != "ဦး", "U+1026 no longer decomposes; this test now proves nothing"
    assert normalize_text(decomposed) == "ဦး"
    assert rust.normalize_str(decomposed) == "ဦး"


def test_normalization_is_idempotent():
    """Applied twice must equal applied once.

    The trainer normalizes its input and the artifact normalizes again at encode
    time, so anything else would mean training and inference saw different text.
    """
    for text in ["ဂွံ​အခေါင်", "a ​b", "ဦး", "  spaced  ", "🙏 ภาษา"]:
        once = normalize_text(text)
        assert normalize_text(once) == once


def test_the_shipped_artifact_carries_the_normalizer():
    """The whole point of moving to `tokenizer.json`.

    If the normalizer were absent from the artifact, `encode` would fall back to
    raw input and v1's train/inference skew would return — silently, because
    nothing else would change.
    """
    tokenizer = MonTokenizer()
    assert tokenizer.encode("ဂွံ​အခေါင်")["pieces"] == tokenizer.encode("ဂွံအခေါင်")["pieces"]
    assert tokenizer.normalize("a b") == "a b"
