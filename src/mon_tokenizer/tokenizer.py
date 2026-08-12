"""Core tokenizer for Mon (mnw), Burmese and English.

## The normalization bug cannot happen here

v1's `encode()` applied no normalization, while its model had been trained on
normalized text with `normalization_rule_name="identity"` — so SentencePiece
never did it either, and one ZERO WIDTH SPACE cost five tokens instead of one.
Fixing that meant remembering to call a function.

v2 does not have that failure available. **The normalizer is serialized inside
`tokenizer.json`**, so it travels with the model and runs on every `encode`.
There is no precondition to forget and no second copy to drift. That is the main
reason for leaving SentencePiece, whose `.model` cannot carry a normalizer this
expressive.

## Round-trip, and its two known limits

The vocabulary carries all 256 `<0xNN>` byte pieces and the model has
`byte_fallback` enabled, so unseen characters reconstruct exactly: Thai, emoji,
IPA, CJK, typographic quotes. Measured on the val split, **100% round-trip on
Mon, Burmese, English and mixed-script text.**

Two inputs do not round-trip, and the loss happens at `encode`, so no decoder can
recover it. `Metaspace(prepend_scheme="always")` gives each of these pairs
identical ids:

    encode(" abc") == encode("abc")      # one leading space is dropped
    encode("a b")  == encode("a▁b")      # a literal U+2581 becomes a space

The val-split figure does not contradict that. `scripts/export_corpus.py` calls
`.strip()` on every line, so no measured line carries leading whitespace, and the
corpus contains no U+2581 at all. The measurement is real, but its corpus
excludes both cases by construction. `tests/test_tokenizer.py` records them as
xfail so the limit stays visible rather than being rediscovered.

Fixing it properly means `prepend_scheme="never"`, which changes every token id
and so needs a retrain and a new Hub artifact. Deferred deliberately: id
stability is worth more than these two cases. Interior whitespace runs, trailing
whitespace, tabs, newlines, NUL and control characters all round-trip today.

That matters for the intended use. `mon-vlm` reads scanned books, so the
tokenizer meets whatever is on the page; the training corpus alone spans 1,458
distinct characters across 24 Unicode blocks. Without byte fallback, unknown
characters are not flagged — they are silently deleted.

Coverage is the other half: the training run guarantees printable ASCII, all
three Myanmar ranges and common book punctuation are each a *single* token, so
**98.74% of the distinct characters in the Mon val split are single-token** —
392 of 397 across all 29,600 lines. The five that are not are one combining
diaeresis below, a Greek capital pi, two Sinhala letters and one emoji, seven
occurrences in 2.28M characters. Those still round-trip, at one to four tokens
per character; they are simply not cheap.

That figure read 100% until 2026-08-12, because the driver measured the first
5,000 Mon lines while its card declared the whole split. The head of that file is
dictionary text and the rarer scripts arrive later, so the cap did not look like
a cap — which is the argument for deriving the line count instead of asserting it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import TypedDict

__all__ = ["Encoding", "MonTokenizer", "default_model_path", "model_card"]


class Encoding(TypedDict):
    """What `encode()` returns."""

    pieces: list[str]
    ids: list[int]
    text: str


def default_model_path() -> Path:
    """Path to the bundled artifact.

    `importlib.resources` rather than `__file__`, so this keeps working when the
    package is loaded from somewhere that is not a plain directory.
    """
    return Path(str(resources.files("mon_tokenizer") / "data" / "mon_tokenizer.json"))


@lru_cache(maxsize=1)
def model_card() -> dict:
    """The measured record shipped beside the artifact.

    Carries the corpus digest, the training config and the per-stratum metrics.
    v1 described its corpus three different ways across README and CHANGELOG and
    published a compression figure that did not reproduce; this is the single
    machine-readable answer, and `tests/test_model_card.py` re-measures against it.
    """
    path = resources.files("mon_tokenizer") / "data" / "model_card.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


@lru_cache(maxsize=4)
def _load(model_path: str):
    """Load and cache by path.

    The artifact is several megabytes and v1 re-read its model on every
    construction. Anything building a tokenizer per request — a web handler, a
    dataloader worker — paid that repeatedly.
    """
    from tokenizers import Tokenizer

    return Tokenizer.from_file(model_path)


class MonTokenizer:
    """Unigram tokenizer for Mon, Burmese and English, with byte fallback.

    Measured on the val split at 64,256 pieces: Mon 4.686 characters per token,
    Burmese 4.117, English 4.112, mixed-script 3.804 — all at 100% round-trip.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        resolved = Path(model_path) if model_path is not None else default_model_path()
        if not resolved.exists():
            raise FileNotFoundError(f"tokenizer artifact not found: {resolved}")
        # Always a Path. v1 annotated this `Optional[str]` while holding a Path on
        # the default branch, and shipped `py.typed`, so every consumer's type
        # checker was told something untrue.
        self.model_path: Path = resolved
        self._tokenizer = _load(str(resolved))

    # -- encoding ---------------------------------------------------------

    def encode(self, text: str) -> Encoding:
        """Tokenize. Returns pieces, ids, and the text that was actually encoded.

        `text` in the result is the **normalized** string — what the model saw —
        not the caller's input. Comparing a decode against it then compares like
        with like; v1 echoed the raw input, which made its own round-trip test
        tautological.
        """
        encoded = self._tokenizer.encode(text, add_special_tokens=False)
        return {"pieces": encoded.tokens, "ids": encoded.ids, "text": self.normalize(text)}

    def encode_ids(self, text: str) -> list[int]:
        """Just the ids, for callers that never look at the piece strings."""
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Ids for many texts, parallelised inside the Rust tokenizer."""
        return [e.ids for e in self._tokenizer.encode_batch(texts, add_special_tokens=False)]

    def normalize(self, text: str) -> str:
        """Apply the artifact's own normalizer.

        Read out of the model rather than reimplemented, so it cannot disagree
        with what `encode` does — which is precisely the disagreement that made
        v1 wrong.
        """
        return str(self._tokenizer.normalizer.normalize_str(text))

    # -- decoding ---------------------------------------------------------

    def decode(self, pieces: list[str]) -> str:
        """Decode from piece strings. Byte pieces are reassembled.

        `decode_ids` is the more direct path; a piece string has to be looked up,
        and a piece absent from the vocabulary raises rather than being dropped.
        """
        ids = [self._tokenizer.token_to_id(piece) for piece in pieces]
        missing = [p for p, i in zip(pieces, ids, strict=True) if i is None]
        if missing:
            raise ValueError(f"not in the vocabulary: {missing[:5]}")
        return self.decode_ids([i for i in ids if i is not None])

    def decode_ids(self, ids: list[int]) -> str:
        return str(self._tokenizer.decode(ids, skip_special_tokens=False))

    # -- vocabulary -------------------------------------------------------

    def get_vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size())

    def get_vocab(self) -> dict[str, int]:
        return dict(self._tokenizer.get_vocab())

    def id_to_piece(self, token_id: int) -> str:
        return str(self._tokenizer.id_to_token(token_id))

    def piece_to_id(self, piece: str) -> int:
        found = self._tokenizer.token_to_id(piece)
        if found is None:
            raise KeyError(f"{piece!r} is not in the vocabulary")
        return int(found)

    # -- special tokens ---------------------------------------------------
    #
    # All four have real ids in v2. v1 declared `<pad>` but never assigned it one,
    # so `pad_id` was -1 and `PieceToId("<pad>")` returned `<unk>` — anyone
    # padding a batch padded with unknown tokens, silently.

    @property
    def unk_id(self) -> int:
        return self.piece_to_id("<unk>")

    @property
    def bos_id(self) -> int:
        return self.piece_to_id("<s>")

    @property
    def eos_id(self) -> int:
        return self.piece_to_id("</s>")

    @property
    def pad_id(self) -> int:
        return self.piece_to_id("<pad>")

    def __repr__(self) -> str:
        return f"MonTokenizer(vocab_size={self.get_vocab_size()}, path={self.model_path.name!r})"
