"""The one definition of what "normalized text" means for Mon.

## Why this file exists

v1 had three copies of this logic and they had drifted:

* `mon-lm/src/train-tokenizer.py` applied it before training, and set
  `normalization_rule_name="identity"` so SentencePiece would never normalize on
  its own.
* `mon_OCR/src/monocr/charset.py` had a fuller version — it also folded Unicode
  space separators to U+0020 — under a comment marking it the single source of
  truth for a different repository.
* `mon_tokenizer`'s `encode()` applied **nothing**, which is the train/inference
  skew that made one ZERO WIDTH SPACE cost five tokens instead of one.

Three definitions of one operation is three ways to disagree, and they did.

## How v2 makes drift impossible rather than unlikely

The normalizer is **serialized into `tokenizer.json`**. A caller loading the
artifact gets the normalization with it; there is no precondition to forget and
no second copy to update. That is the main reason for moving off SentencePiece,
whose `.model` cannot carry a normalizer this expressive.

`normalize_text` below is a pure-Python reference implementation of the same
spec, for callers who want the operation standalone and for
`tests/test_normalization.py`, which asserts the two agree character-for-character
over a corpus sample. If they ever diverge, that test fails rather than a CER
number quietly getting worse.

## The three steps, and why each is there

1. **Strip invisible characters.** Zero-width joiners and friends arrive
   constantly from web pages, Facebook and PDF extraction. They are invisible in
   the image and absent from the training corpus, so leaving them in pushes the
   input off-distribution for something the reader cannot even see.

2. **Fold Unicode space separators to U+0020.** Adopted from mon_OCR, which
   measured the cost of *not* doing it: 20 U+00A0 in its train labels and 9 in
   val, each rendering as a visible gap while being dropped from the target.
   Every `Zs` looks like a space; treating them as one is what a reader does.

3. **NFC.** Narrow but real for this script: measured across U+1000–U+109F,
   exactly **one** codepoint has a canonical decomposition — `ဦ` U+1026 MYANMAR
   LETTER UU, decomposing to U+1025 U+102E. It is common (it opens `ဦး`, the
   honorific), and a decomposed one renders identically while tokenizing
   differently.

## What is deliberately NOT here

**Runs of spaces are preserved.** SentencePiece collapses them by default, which
is the sole cause of v1's round-trip loss: 4.42% of corpus lines did not survive
`decode(encode(x))`, and every one of those failures was collapsed `U+0020` —
18,603 of them, with zero Mon characters lost or gained. Whitespace is content.
"""

from __future__ import annotations

import unicodedata

__all__ = ["INVISIBLE_CHARS", "INVISIBLE_PATTERN", "hf_normalizer", "normalize_text"]

# Exactly the set v1's trainer stripped, kept identical so v1 and v2 differ only
# where the difference was deliberate. Extending it silently moves inference off
# whatever distribution the model was fitted to, so it changes only with a retrain.
INVISIBLE_CHARS: frozenset[str] = frozenset(
    {
        "​",  # ZERO WIDTH SPACE
        "‌",  # ZERO WIDTH NON-JOINER
        "‍",  # ZERO WIDTH JOINER
        "⁠",  # WORD JOINER
        "﻿",  # BOM / ZERO WIDTH NO-BREAK SPACE
        "­",  # SOFT HYPHEN
    }
)

# The same set as a character class, for the Rust regex engine inside the
# artifact. Built from INVISIBLE_CHARS rather than written out a second time —
# a hand-maintained duplicate of a six-element set is still a duplicate.
INVISIBLE_PATTERN: str = "[" + "".join(f"\\u{ord(c):04X}" for c in sorted(INVISIBLE_CHARS)) + "]"


def normalize_text(text: str) -> str:
    """Reference implementation of the artifact's normalizer.

    Not on the hot path — `tokenizer.json` carries the normalizer, so `encode()`
    applies it inside Rust. This exists so the operation is available standalone
    and so the spec is testable against what actually ships.
    """
    without_invisibles = "".join(c for c in text if c not in INVISIBLE_CHARS)
    spaces_folded = "".join(
        " " if unicodedata.category(c) == "Zs" else c for c in without_invisibles
    )
    return unicodedata.normalize("NFC", spaces_folded)


def hf_normalizer():
    """The normalizer that gets baked into `tokenizer.json`.

    Imported lazily so `normalize_text` stays usable without `tokenizers`
    installed, which matters for the corpus-preparation path.

    `\\p{Zs}` is Unicode-property syntax supported by the Rust regex engine; the
    equivalence with Python's `unicodedata.category(c) == "Zs"` is asserted in
    `tests/test_normalization.py` rather than assumed.
    """
    from tokenizers import Regex, normalizers

    return normalizers.Sequence(
        [
            normalizers.Replace(Regex(INVISIBLE_PATTERN), ""),
            normalizers.Replace(Regex(r"\p{Zs}"), " "),
            normalizers.NFC(),
        ]
    )
