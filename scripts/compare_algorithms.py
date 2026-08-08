#!/usr/bin/env python3
"""Which subword algorithm is actually best for Mon, Burmese and English?

The choice of Unigram was inherited from v1 and defended twice by argument —
"it is what T5 and Gemma use", "it suits agglutinative morphology". Those are
citations, not evidence about *this* corpus. This script replaces them with a
measurement.

Four candidates, all at the same vocabulary size, on the same train split,
scored on the same full val split through the same harness:

  unigram        the incumbent, with byte fallback
  bpe            byte fallback, otherwise identical treatment
  byte-level-bpe GPT/Llama style: every byte is a token, so nothing is ever
                 unrepresentable and `unk` cannot occur. The interesting
                 question is what that costs on a 3-bytes-per-character script.
  wordpiece      BERT-era. Included because "we checked" beats "we assumed",
                 and it is cheap to run.

    uv run python scripts/compare_algorithms.py --vocab-size 48000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mon_tokenizer.metrics import coverage, measure
from mon_tokenizer.normalization import hf_normalizer
from mon_tokenizer.trainer import (
    BYTE_PIECES,
    BYTE_SCORE,
    SPECIAL_TOKENS,
    WORD_MARKER,
    guaranteed_alphabet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare")

ROOT = Path(__file__).parent.parent
BUCKETS = ("mon", "burmese", "english", "mixed")


def load(path: Path):
    train: list[str] = []
    val: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            if row["split"] == "train":
                train.append(row["text"])
            elif row["split"] == "val":
                val[row["bucket"]].append(row["text"])
                # A fourth stratum: lines carrying anything outside ASCII and
                # Myanmar — Thai, emoji, IPA, typographic quotes. The three
                # language buckets measure none of that, and it is exactly what a
                # scanned book contains.
                if any(
                    not (ord(c) < 0x80 or 0x1000 <= ord(c) <= 0x109F or c.isspace())
                    for c in row["text"]
                ):
                    val["mixed"].append(row["text"])
    return train, dict(val)


def _metaspace():
    from tokenizers import pre_tokenizers

    return pre_tokenizers.Metaspace(replacement=WORD_MARKER, prepend_scheme="always")


def _byte_decoder():
    from tokenizers import decoders

    return decoders.Sequence(
        [
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Replace(WORD_MARKER, " "),
            decoders.Strip(" ", 1, 0),
        ]
    )


def build_unigram(train: list[str], vocab_size: int):
    from tokenizers import AddedToken, Tokenizer, models, trainers

    tok = Tokenizer(models.Unigram())
    tok.normalizer, tok.pre_tokenizer = hf_normalizer(), _metaspace()
    tok.train_from_iterator(
        train,
        trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            unk_token="<unk>",
            initial_alphabet=guaranteed_alphabet(),
            show_progress=False,
        ),
    )
    vocab = [(p, s) for p, s in json.loads(tok.to_str())["model"]["vocab"]]
    present = {p for p, _ in vocab}
    vocab += [(p, BYTE_SCORE) for p in BYTE_PIECES if p not in present]
    unk = next(i for i, (p, _) in enumerate(vocab) if p == "<unk>")
    out = Tokenizer(models.Unigram(vocab, unk_id=unk, byte_fallback=True))
    out.normalizer, out.pre_tokenizer, out.decoder = hf_normalizer(), _metaspace(), _byte_decoder()
    out.add_special_tokens([AddedToken(t, special=True) for t in SPECIAL_TOKENS])
    return out


def build_bpe(train: list[str], vocab_size: int):
    from tokenizers import Tokenizer, models, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
    tok.normalizer, tok.pre_tokenizer, tok.decoder = hf_normalizer(), _metaspace(), _byte_decoder()
    tok.train_from_iterator(
        train,
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=guaranteed_alphabet() + BYTE_PIECES,
            show_progress=False,
        ),
    )
    return tok


def build_byte_level_bpe(train: list[str], vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.normalizer = hf_normalizer()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        train,
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    return tok


def build_wordpiece(train: list[str], vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, trainers

    tok = Tokenizer(models.WordPiece(unk_token="<unk>", max_input_chars_per_word=200))
    tok.normalizer, tok.pre_tokenizer = hf_normalizer(), _metaspace()
    tok.decoder = decoders.WordPiece(prefix="##")
    tok.train_from_iterator(
        train,
        trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=guaranteed_alphabet(),
            show_progress=False,
        ),
    )
    return tok


BUILDERS = {
    "unigram": build_unigram,
    "bpe": build_bpe,
    "byte-level-bpe": build_byte_level_bpe,
    "wordpiece": build_wordpiece,
}


def fallback_ids(tokenizer) -> frozenset[int]:
    """Ids that mean "spelled out" — byte pieces plus unknown."""
    vocab = tokenizer.get_vocab()
    ids = {i for piece, i in vocab.items() if piece.startswith("<0x") and piece.endswith(">")}
    unk = vocab.get("<unk>")
    if unk is not None:
        ids.add(unk)
    return frozenset(ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "build/corpus.jsonl")
    parser.add_argument("--vocab-size", type=int, default=48_000)
    parser.add_argument("--only", nargs="*", choices=list(BUILDERS), default=list(BUILDERS))
    parser.add_argument("--output", type=Path, default=ROOT / "build/algorithms.json")
    args = parser.parse_args()

    train, val = load(args.corpus)
    logger.info("train %d lines; val %s", len(train), {b: len(v) for b, v in val.items()})

    probe = "ကျော် page 42 — “quoted” ၏ 🙏 ภาษา ə café ×2 …"
    results: dict[str, dict] = {}

    for name in args.only:
        started = time.time()
        tokenizer = BUILDERS[name](train, args.vocab_size)
        elapsed = time.time() - started
        fids = fallback_ids(tokenizer)

        per_bucket = {}
        for bucket in BUCKETS:
            lines = val.get(bucket, [])
            if not lines:
                continue
            per_bucket[bucket] = measure(
                lambda t, tk=tokenizer: (tk.encode(t).tokens, tk.encode(t).ids),
                lambda _pieces, ids, tk=tokenizer: tk.decode(ids),
                lines,
                fids,
            ).as_dict()

        encoded = tokenizer.encode(probe)
        results[name] = {
            "vocab_size": tokenizer.get_vocab_size(),
            "train_seconds": round(elapsed, 1),
            "buckets": per_bucket,
            "coverage": coverage(tokenizer.get_vocab().keys(), "".join(val.get("mon", [])[:2000])),
            "probe_tokens": len(encoded.ids),
            "probe_roundtrip": tokenizer.decode(encoded.ids) == probe,
        }
        logger.info("%s done in %.0fs", name, elapsed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 104)
    header = f"{'algorithm':<16}"
    for b in BUCKETS:
        header += f"{b[:3] + ' c/t':>10}"
    header += f"{'mon viol':>10}{'mon rt':>8}{'mon fb':>8}{'cover':>8}{'probe':>7}{'train s':>9}"
    print(header)
    print("-" * 104)
    for name, r in results.items():
        row = f"{name:<16}"
        for b in BUCKETS:
            row += f"{r['buckets'].get(b, {}).get('chars_per_token', 0):>10.3f}"
        mon = r["buckets"].get("mon", {})
        row += (
            f"{mon.get('violation_rate', 0):>9.2%}{mon.get('roundtrip', 0):>8.1%}"
            f"{mon.get('fallback_rate', 0):>8.2%}{r['coverage']['single_token']:>8.1%}"
            f"{r['probe_tokens']:>7}{r['train_seconds']:>9.0f}"
        )
        print(row)
    print("=" * 104)
    print(f"probe round-trip: {[(n, r['probe_roundtrip']) for n, r in results.items()]}")
    print(f"mon syllable denominator: {results[args.only[0]]['buckets']['mon']['syllables']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
