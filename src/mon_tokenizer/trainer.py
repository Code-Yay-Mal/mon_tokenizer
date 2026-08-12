"""Training a Mon tokenizer, and recording what it was trained on.

`mon-lm/src/train-tokenizer.py:6` has claimed since April that *"The core logic is
identical to mon_tokenizer/trainer.py — keep both in sync"*. That file never
existed. This is it, and mon-lm's copy is deleted rather than synced: a sync
contract between two files is a promise someone has to keep, and this one was
broken from the day it was written.

## Byte fallback is the feature that makes this safe for OCR

`mon-vlm` will read scanned books and photographed pages, so the tokenizer meets
whatever is on the page. The corpus alone carries **1,458 distinct characters
across 24 Unicode blocks** — 23,650 Thai characters from dictionary translations,
64,866 typographic quotes and dashes, 621 emoji, plus Greek, IPA, Devanagari,
Cyrillic, Arabic, Hebrew and CJK.

Without byte fallback those are not marked unknown. They are **silently deleted**:

    'ကျော် page 42 — "quoted" ၏ 🙏 ภาษา ə café ×2 …'
        decodes to
    'ကော် e   oed       '

Fluent-looking output with content removed and no signal anything was lost — the
worst failure mode for a pipeline whose output feeds a corpus.

**And enabling it is not one flag.** `UnigramTrainer` has no byte-fallback
parameter, and `models.Unigram(vocab, byte_fallback=True)` still emits `<unk>`,
because the 256 `<0xNN>` pieces it would fall back to **are not in the
vocabulary**. SentencePiece's trainer adds them; HuggingFace's does not. So the
pieces are injected here, explicitly, and the decoder is a `Sequence` that knows
how to reassemble them. Verified: with those in place all three probe strings
round-trip exactly, on a 31-piece vocabulary — proving it is the byte path doing
the work and not the vocabulary.

## What changed from mon-lm's version, and why

**The `min_mon_ratio=0.3` filter is gone.** It dropped any line less than 30%
Myanmar, so Burmese- and English-dominant lines never entered the fit. Mon is not
written in isolation.

**`user_defined_symbols` for grapheme clusters is gone**, because `tokenizers` has
no equivalent and the obvious substitute is worse — a `Split` pre-tokenizer on
`\\X` measured 1.144 chars/token, 4.7x worse than v1, because a pre-tokenizer can
only bound how far a token *may* span, never forbid a split inside one.
`filter_partial_syllable_pieces` is the replacement, gated on measurement.

**Runs of spaces are preserved.** SentencePiece collapses them by default, the
sole cause of v1's round-trip loss.

**Training is on the train split only.** v1 was fitted on the whole corpus
including text it was later scored on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import string
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .normalization import hf_normalizer, normalize_text
from .syllable import MYANMAR_RANGES, starts_mid_syllable

logger = logging.getLogger(__name__)

__all__ = [
    "CorpusStats",
    "TrainConfig",
    "build_model_card",
    "corpus_fingerprint",
    "filter_partial_syllable_pieces",
    "guaranteed_alphabet",
    "train",
]

WORD_MARKER = "▁"
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]
BYTE_PIECES = [f"<0x{value:02X}>" for value in range(256)]
# Byte pieces must lose to any real vocabulary entry, so they are only reached
# when nothing else matches. Trained Unigram log-probabilities sit far above this.
BYTE_SCORE = -100.0

# Typographic characters that appear in scanned books and are worth a whole token
# rather than three bytes. Measured in the corpus, not guessed: General
# Punctuation alone is 64,866 occurrences.
BOOK_PUNCTUATION = "“”‘’—–…•·×°†‡§¶№½¼¾"


@dataclass
class TrainConfig:
    """Everything that determines the artifact, recorded so a run is repeatable."""

    # The size the shipped artifact was trained at, and the same default
    # `scripts/train_tokenizer.py` passes. It read 48,000 — a size nothing in the
    # project uses — so `train(lines)` with no config quietly built a tokenizer
    # 16,000 pieces smaller than the one this repository ships and measures.
    vocab_size: int = 64_000
    special_tokens: list[str] = field(default_factory=lambda: list(SPECIAL_TOKENS))
    unk_token: str = "<unk>"
    byte_fallback: bool = True
    enforce_syllable_atomicity: bool = False
    # The library default, stated rather than inherited: a silent default is one
    # nobody notices changing. Verified equal to the default in tokenizers 0.23.1.
    max_piece_length: int = 16

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CorpusStats:
    """What the tokenizer was fitted on. Half of reproducibility."""

    lines: int = 0
    chars: int = 0
    by_bucket: dict[str, int] = field(default_factory=dict)
    source_digest: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def corpus_fingerprint(paths: Sequence[Path]) -> str:
    """Digest over the *content* of every source file, order-independent.

    Names and mtimes are not enough — a file edited in place keeps both. Hashing
    sorted per-file digests means the fingerprint changes when the corpus changes
    and only then.
    """
    digests = sorted(
        hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest() for path in paths
    )
    return hashlib.blake2b("".join(digests).encode(), digest_size=16).hexdigest()


def guaranteed_alphabet() -> list[str]:
    """Characters that must each be a single token, whatever the corpus frequency.

    Byte fallback makes everything *representable*; this makes the common cases
    *cheap*. A page number in Myanmar digits or an English caption should not cost
    three tokens per character because the training corpus happened to be thin
    there.

    Deliberately not extended to Thai, emoji or CJK: they are real but rare
    (0.027% and below), and 64,000 vocabulary slots are better spent on Mon.
    Byte fallback carries them correctly, just not cheaply.
    """
    alphabet = set(string.printable) - set("\t\r\x0b\x0c")
    for start, end in MYANMAR_RANGES:
        alphabet.update(chr(code) for code in range(start, end + 1))
    alphabet.update(BOOK_PUNCTUATION)
    return sorted(alphabet)


def filter_partial_syllable_pieces(
    vocab: Sequence[tuple[str, float]],
) -> tuple[list[tuple[str, float]], int]:
    """Drop pieces that can only appear mid-syllable. Returns (kept, dropped).

    Viterbi can only put a boundary inside a Myanmar syllable if some vocabulary
    piece starts with a combining mark. Remove those and the segmentation has to
    respect syllable edges — without the compression cost of a `Split`
    pre-tokenizer, because pieces spanning *several* syllables are untouched.

    Whether it helps is a measurement, not a claim: it trades vocabulary slots
    for syllable integrity, and `TrainConfig.enforce_syllable_atomicity` defaults
    to **off** until the numbers say otherwise.
    """
    kept = [entry for entry in vocab if not starts_mid_syllable(entry[0], WORD_MARKER)]
    return kept, len(vocab) - len(kept)


def _decoder():
    """Reassemble pieces, bytes included.

    Order matters and was determined by testing, not by reading: `ByteFallback`
    must run before `Fuse` so the `<0xNN>` runs become real characters first, and
    the Metaspace marker is replaced afterwards. `Strip` removes the single
    leading space that `prepend_scheme="always"` introduces.
    """
    from tokenizers import decoders

    return decoders.Sequence(
        [
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Replace(WORD_MARKER, " "),
            decoders.Strip(" ", 1, 0),
        ]
    )


def _attach(tokenizer):
    """Normalizer, pre-tokenizer and decoder — the three the model does not carry."""
    from tokenizers import pre_tokenizers

    tokenizer.normalizer = hf_normalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement=WORD_MARKER, prepend_scheme="always"
    )
    tokenizer.decoder = _decoder()
    return tokenizer


def train(lines: Iterable[str], config: TrainConfig | None = None):
    """Fit a Unigram tokenizer. Returns a `tokenizers.Tokenizer`.

    Lines are normalized here as well as by the artifact's own normalizer:
    the trainer must see exactly what inference will see, and relying on the
    pipeline to do it means a pipeline change silently becomes a training-data
    change.

    The caller supplies the lines and is responsible for having restricted them
    to the train split. This does not filter silently — a function that quietly
    drops part of its input is one whose contract nobody can check.
    """
    from tokenizers import Tokenizer, models, trainers

    config = config or TrainConfig()
    prepared = [normalize_text(line) for line in lines]
    if not prepared:
        raise ValueError("no lines to train on")

    tokenizer = _attach(Tokenizer(models.Unigram()))
    logger.info(
        "training unigram vocab=%d on %d lines / %d chars",
        config.vocab_size,
        len(prepared),
        sum(map(len, prepared)),
    )
    tokenizer.train_from_iterator(
        prepared,
        trainers.UnigramTrainer(
            vocab_size=config.vocab_size,
            special_tokens=config.special_tokens,
            unk_token=config.unk_token,
            max_piece_length=config.max_piece_length,
            initial_alphabet=guaranteed_alphabet(),
            show_progress=False,
        ),
    )
    return _finalise(tokenizer, config)


def _finalise(tokenizer, config: TrainConfig):
    """Rebuild the model with byte pieces, and optionally without partial syllables.

    Always a rebuild, never a conditional one. The first version applied byte
    fallback only on the atomicity path, so the gate compared
    `atomicity + byte_fallback` against neither and could not isolate the thing it
    was gating. One path now, both variants.
    """
    from tokenizers import AddedToken, Tokenizer, models

    state = json.loads(tokenizer.to_str())
    vocab: list[tuple[str, float]] = [(piece, score) for piece, score in state["model"]["vocab"]]

    dropped = 0
    if config.enforce_syllable_atomicity:
        vocab, dropped = filter_partial_syllable_pieces(vocab)
        logger.info("syllable atomicity: dropped %d pieces starting with a combining mark", dropped)

    if config.byte_fallback:
        present = {piece for piece, _ in vocab}
        missing = [(piece, BYTE_SCORE) for piece in BYTE_PIECES if piece not in present]
        vocab.extend(missing)
        logger.info("byte fallback: injected %d of 256 byte pieces", len(missing))

    unk_id = next((i for i, (piece, _) in enumerate(vocab) if piece == config.unk_token), None)
    if unk_id is None:
        # Loud, not silent. A missing unk token means every out-of-vocabulary
        # span has nowhere to go, and defaulting to id 0 would hide that behind
        # whatever piece happens to be first.
        raise ValueError(f"{config.unk_token!r} is not in the vocabulary after filtering")

    rebuilt = _attach(
        Tokenizer(models.Unigram(vocab, unk_id=unk_id, byte_fallback=config.byte_fallback))
    )
    # Re-adding these is not optional: a fresh Tokenizer has no added_tokens, so
    # the first implementation silently lost every special token — the exact
    # pad-with-<unk> class of bug this project is fixing in v1.
    rebuilt.add_special_tokens([AddedToken(token, special=True) for token in config.special_tokens])
    return rebuilt


def build_model_card(
    config: TrainConfig,
    corpus: CorpusStats,
    metrics: dict,
    artifact_version: str,
    vocab_size: int,
) -> dict:
    """The record that makes an artifact accountable.

    v1 described its corpus three different ways across README and CHANGELOG
    (41.4M / 92.8M / 177M characters) and published a compression figure that did
    not reproduce. A machine-readable card that a test re-measures against is the
    fix.

    `artifact_version` names the **trained model**, not the package. The two are
    independent by design: a bugfix release of the library ships a byte-identical
    tokenizer, and forcing this card to be regenerated for it would mean
    republishing the Hugging Face artifact to change one string. The field was
    called `version` and sat next to `__version__` with nothing saying which it
    tracked, which is an invitation to couple them by accident.

    It changes when, and only when, the artifact does — which is why
    `tests/test_model_card.py` pins each value to the corpus digest it was
    trained on.
    """
    return {
        "name": "mon-tokenizer",
        "artifact_version": artifact_version,
        "algorithm": "unigram",
        "vocab_size": vocab_size,
        "languages": ["mnw", "mya", "eng"],
        "config": config.as_dict(),
        "corpus": corpus.as_dict(),
        "metrics": metrics,
        "notes": {
            "compression_is_corpus_dependent": (
                "chars_per_token describes this corpus, not the tokenizer. Quote it "
                "with the corpus or not at all."
            ),
            "violation_rate": (
                "Share of multi-character Myanmar syllables split by a token boundary, "
                "using the segmenter in syllable.py rather than Unicode grapheme "
                "clusters, which do not bound a Myanmar syllable. Report the "
                "denominator with the rate."
            ),
            "roundtrip": (
                "Compared after normalization on both sides. Normalization is a "
                "deliberate transform, not a loss."
            ),
            "byte_fallback": (
                "Every input round-trips, including Thai, emoji and CJK, at 1-4 "
                "tokens per character. Only the guaranteed alphabet is cheap."
            ),
        },
    }
