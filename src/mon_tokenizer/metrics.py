"""Measuring a tokenizer: compression, fidelity, coverage, and syllable integrity.

Every number in the README, the CHANGELOG and `model_card.json` comes from here,
and a test re-runs these functions against the shipped artifact and fails if the
recorded numbers have drifted. Numbers maintained by hand in prose go stale — v1
claimed 5.22 chars/token, 100% round-trip and 0.00% byte fallback, and none of
the three reproduced.

## Five measurements, and what each one is for

**Compression** (`chars_per_token`) is the headline, and it is a property of a
*corpus*, not of a tokenizer. Report it with the corpus or not at all.

**Round-trip** is fidelity: does `decode(encode(x))` return `x`? Compared after
normalization on both sides — normalization is a deliberate transform, not a
loss — but **not** after `.strip()`. An earlier version stripped both sides, which
would have hidden leading- and trailing-whitespace loss, the exact class of defect
v2 exists to fix.

**Fallback rate** is how much of the output is spelled out byte by byte.
Measured **by token id**, against the vocabulary's byte pieces and `unk_id` — not
by pattern-matching the piece strings. That matters: HuggingFace Unigram returns
the *original surface substring* for an unknown span rather than the string
`<unk>`, so a string-matching implementation reports 0.00% on text the tokenizer
cannot represent at all. It silently did.

**Coverage** is the complement, and the one that matters for scanning books: what
share of the distinct characters in a text is each representable as a single
token, versus needing byte fallback, versus not representable at all.

**Syllable violation rate** is the one nobody was measuring correctly. A Myanmar
syllable is a base plus its stacked marks and killed finals — one unit a reader
sees. A token boundary inside it corresponds to nothing on the page. This uses
`syllable.syllable_spans`, **not** Unicode grapheme clusters: UAX #29 puts a
cluster break before `ာ`, so `\\X` scores a cut through the middle of `ကျော်` as
clean.

Offsets are reconstructed from the pieces and then **verified against the source
text**; a line whose reconstruction does not match is counted as
`unreconstructable` and excluded from the violation rate rather than guessed at.
A metric that silently guesses is worse than one that reports a gap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass

from .normalization import normalize_text
from .syllable import syllable_spans

__all__ = [
    "BucketMetrics",
    # The callable contract `measure` takes. Exported because callers implement
    # it — `scripts/compare_algorithms.py` builds one per algorithm — and a type
    # that describes a caller's obligation is part of the interface.
    "Decoder",
    "Encoder",
    "coverage",
    "measure",
    "piece_boundaries",
    "syllable_violations",
]

WORD_MARKER = "▁"

# `encode` returns (pieces, ids). Both are needed: pieces to reconstruct offsets,
# ids to identify fallback without pattern-matching strings.
Encoder = Callable[[str], tuple[list[str], list[int]]]
Decoder = Callable[[list[str], list[int]], str]


@dataclass
class BucketMetrics:
    """What one stratum measures."""

    lines: int = 0
    chars: int = 0
    tokens: int = 0
    chars_per_token: float = 0.0
    tokens_per_line: float = 0.0
    roundtrip: float = 0.0
    fallback_rate: float = 0.0
    violation_rate: float = 0.0
    # Reported beside the rate, always. On some strata the denominator is tiny —
    # full English val has exactly ONE multi-character syllable — and a rate over
    # a denominator of one is noise wearing a percentage sign.
    syllables: int = 0
    unreconstructable: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def piece_boundaries(text: str, pieces: Sequence[str]) -> set[int] | None:
    """Character offsets in `text` where a token boundary falls.

    Returns ``None`` when the pieces do not reconstruct the text exactly — which
    happens whenever the tokenizer normalized or lost something. The caller must
    treat that as "cannot measure this line", never as zero violations: counting
    an unmeasurable line as clean is how a violation rate flatters itself.
    """
    rebuilt = [piece.replace(WORD_MARKER, " ") for piece in pieces]
    joined = "".join(rebuilt)

    # Metaspace with prepend_scheme="always" adds a marker to the first piece,
    # which reconstructs as a leading space the source does not have. Guard on
    # the piece rather than the join: an empty first piece would make the slice a
    # no-op and drop the line to unreconstructable for no reason.
    if joined != text and rebuilt and rebuilt[0].startswith(" ") and joined[1:] == text:
        rebuilt[0] = rebuilt[0][1:]
        joined = "".join(rebuilt)
    if joined != text:
        return None

    cuts: set[int] = set()
    position = 0
    for chunk in rebuilt[:-1]:
        position += len(chunk)
        # Offset 0 is the start of the string, not a boundary between tokens.
        if position:
            cuts.add(position)
    return cuts


def syllable_violations(text: str, cuts: set[int]) -> tuple[int, int]:
    """``(syllables split, multi-character syllables)`` for one line."""
    split = 0
    spans = syllable_spans(text)
    for start, end in spans:
        if any(start < cut < end for cut in cuts):
            split += 1
    return split, len(spans)


def coverage(vocabulary: Iterable[str], text: str) -> dict[str, float]:
    """How the distinct characters of `text` are represented.

    The question a book-scanning pipeline actually asks: will this page cost one
    token per character, three, or be lost? `single_token` is the cheap path;
    everything else round-trips through byte fallback but costs more.
    """
    known = set(vocabulary)
    distinct = {c for c in normalize_text(text)}
    if not distinct:
        return {"distinct": 0, "single_token": 0.0}
    single = sum(1 for c in distinct if c in known or f"{WORD_MARKER}{c}" in known)
    return {"distinct": len(distinct), "single_token": round(single / len(distinct), 4)}


def measure(
    encode: Encoder,
    decode: Decoder,
    lines: Iterable[str],
    fallback_ids: frozenset[int] = frozenset(),
) -> BucketMetrics:
    """Measure one stratum.

    Callables rather than a tokenizer object, so two tokenizers can be compared
    without either backend leaking into the metric. That is what made the v1
    baseline like-for-like when v1 was still a SentencePiece `.model` — 1.0.0
    ships `tokenizer.json` and depends on SentencePiece nowhere — and it is what
    lets `scripts/compare_algorithms.py` score unigram, BPE, byte-level BPE and
    WordPiece through this one function.

    `fallback_ids` is the set of byte-piece and unknown ids. Supplying it is the
    caller's job because only the caller knows the vocabulary; supplying nothing
    means fallback is reported as zero, which is honest but uninformative.
    """
    metrics = BucketMetrics()
    split_total = syllable_total = exact = fallback_tokens = 0

    for raw in lines:
        line = normalize_text(raw)
        pieces, ids = encode(line)
        metrics.lines += 1
        metrics.chars += len(line)
        metrics.tokens += len(ids)
        fallback_tokens += sum(1 for token_id in ids if token_id in fallback_ids)

        # No .strip(): whitespace is content, and this metric exists to catch its
        # loss rather than to look good.
        if normalize_text(decode(pieces, ids)) == line:
            exact += 1

        cuts = piece_boundaries(line, pieces)
        if cuts is None:
            metrics.unreconstructable += 1
            continue
        split, total = syllable_violations(line, cuts)
        split_total += split
        syllable_total += total

    if not metrics.lines:
        return metrics

    metrics.chars_per_token = round(metrics.chars / metrics.tokens, 3) if metrics.tokens else 0.0
    metrics.tokens_per_line = round(metrics.tokens / metrics.lines, 1)
    metrics.roundtrip = round(exact / metrics.lines, 4)
    metrics.syllables = syllable_total
    metrics.violation_rate = round(split_total / syllable_total, 5) if syllable_total else 0.0
    metrics.fallback_rate = round(fallback_tokens / metrics.tokens, 5) if metrics.tokens else 0.0
    return metrics
