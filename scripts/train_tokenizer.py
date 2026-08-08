#!/usr/bin/env python3
"""Train the Mon tokenizer, measure it, and write the artifact with its model card.

    uv run python scripts/train_tokenizer.py --gate      # compare variants, write nothing
    uv run python scripts/train_tokenizer.py --write     # train and ship

Reads the JSONL produced by `export_corpus.py`. Fits on the **train** split and
measures on the **whole val** split — never a head slice. An earlier version took
`val[bucket][:5000]`, which in file order meant `dict/` supplied 47.5% of the Mon
sample while being 8.1% of the split, and `mon_shards/` — 47.2% of the split —
supplied nothing. Dictionary headwords and running prose compress very
differently, so that number described a sample nobody had chosen deliberately.

`--gate` exists because the syllable-atomicity filter is an open question, not a
decision. It trains both variants, prints both, and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mon_tokenizer.metrics import coverage, measure
from mon_tokenizer.trainer import CorpusStats, TrainConfig, build_model_card, train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_tokenizer")

ROOT = Path(__file__).parent.parent
BUCKETS = ("mon", "burmese", "english", "mixed")


def _is_mixed(text: str) -> bool:
    """Carries anything outside ASCII and Myanmar — Thai, emoji, IPA, curly quotes.

    The three language buckets measure none of that, and it is exactly what a
    scanned book contains: the corpus holds 23,650 Thai characters, 64,866
    typographic punctuation marks and 621 emoji.
    """
    return any(not (ord(c) < 0x80 or 0x1000 <= ord(c) <= 0x109F or c.isspace()) for c in text)


def load(path: Path) -> tuple[list[str], dict[str, list[str]], dict]:
    train_lines: list[str] = []
    val: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            if row["split"] == "train":
                train_lines.append(row["text"])
            elif row["split"] == "val":
                val[row["bucket"]].append(row["text"])
                if _is_mixed(row["text"]):
                    val["mixed"].append(row["text"])
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return train_lines, dict(val), meta


def fallback_ids(tokenizer) -> frozenset[int]:
    """Ids that mean "spelled out byte by byte", plus unknown.

    Identified by id, not by matching piece strings: HuggingFace Unigram returns
    the original surface substring for an unknown span rather than the literal
    `<unk>`, so a string-matching check reports 0.00% on text the tokenizer
    cannot represent. It did exactly that.
    """
    vocab = tokenizer.get_vocab()
    ids = {i for piece, i in vocab.items() if piece.startswith("<0x") and piece.endswith(">")}
    if (unk := vocab.get("<unk>")) is not None:
        ids.add(unk)
    return frozenset(ids)


def evaluate(tokenizer, val: dict[str, list[str]]) -> dict:
    """Every stratum, whole split."""
    fids = fallback_ids(tokenizer)
    out = {}
    for bucket in BUCKETS:
        lines = val.get(bucket, [])
        if not lines:
            continue
        out[bucket] = measure(
            lambda t, tk=tokenizer: (tk.encode(t).tokens, tk.encode(t).ids),
            lambda _pieces, ids, tk=tokenizer: tk.decode(ids),
            lines,
            fids,
        ).as_dict()
    out["coverage"] = coverage(tokenizer.get_vocab().keys(), "".join(val.get("mon", [])[:5000]))
    return out


def show(name: str, metrics: dict) -> None:
    print(f"\n{name}")
    print(
        f"  {'stratum':<9}{'chars/tok':>11}{'tok/line':>10}{'violations':>12}"
        f"{'syllables':>11}{'round-trip':>12}{'fallback':>10}{'unrecon':>9}"
    )
    for bucket in BUCKETS:
        m = metrics.get(bucket)
        if not m:
            continue
        print(
            f"  {bucket:<9}{m['chars_per_token']:>11.3f}{m['tokens_per_line']:>10.1f}"
            f"{m['violation_rate']:>11.2%}{m['syllables']:>11,}{m['roundtrip']:>12.1%}"
            f"{m['fallback_rate']:>10.2%}{m['unreconstructable']:>9,}"
        )
    print(f"  single-token character coverage: {metrics['coverage']['single_token']:.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "build/corpus.jsonl")
    parser.add_argument("--vocab-size", type=int, default=64_000)
    parser.add_argument("--gate", action="store_true", help="compare variants, write nothing")
    parser.add_argument("--atomicity", action="store_true")
    parser.add_argument("--write", action="store_true")
    # Names the trained artifact, not the package — see build_model_card. No
    # default: a retrain that silently reuses the previous artifact's version
    # publishes two different tokenizers under one name, which is the shape of
    # the defect this release fixed on the Hub.
    #
    # Required on --write but not by argparse, because --gate writes nothing and
    # should not make anyone invent a version to compare two variants.
    parser.add_argument(
        "--artifact-version",
        help="version of the artifact being written, e.g. 1.1.0 (not the package version)",
    )
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"{args.corpus} not found — run scripts/export_corpus.py first", file=sys.stderr)
        return 1

    train_lines, val, meta = load(args.corpus)
    logger.info(
        "train %d lines / %d chars; val %s",
        len(train_lines),
        sum(map(len, train_lines)),
        {b: len(val.get(b, [])) for b in BUCKETS},
    )

    variants = [False, True] if args.gate else [args.atomicity]
    results = []
    for atomicity in variants:
        config = TrainConfig(vocab_size=args.vocab_size, enforce_syllable_atomicity=atomicity)
        tokenizer = train(train_lines, config)
        metrics = evaluate(tokenizer, val)
        show(
            f"vocab {args.vocab_size:,}, syllable atomicity {'ON' if atomicity else 'off'} "
            f"(actual vocab {tokenizer.get_vocab_size():,})",
            metrics,
        )
        results.append((tokenizer, config, metrics))

    if args.gate or not args.write:
        print("\n(nothing written)")
        return 0

    if not args.artifact_version:
        print(
            "--write needs --artifact-version. It names the artifact being "
            "shipped, and reusing the previous one would publish two different "
            "tokenizers under one name.",
            file=sys.stderr,
        )
        return 1

    tokenizer, config, metrics = results[0]
    data_dir = ROOT / "src/mon_tokenizer/data"
    data_dir.mkdir(parents=True, exist_ok=True)
    artifact = data_dir / "mon_tokenizer.json"
    tokenizer.save(str(artifact))

    corpus = CorpusStats(
        lines=meta["lines"],
        chars=meta["chars"],
        by_bucket={b: sum(s.values()) for b, s in meta["by_bucket_split"].items()},
        source_digest=meta["source_digest"],
    )
    card = build_model_card(
        config, corpus, metrics, args.artifact_version, tokenizer.get_vocab_size()
    )
    card["eval"] = {"split": "val", "sampling": "whole split, no cap"}
    (data_dir / "model_card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {artifact} ({artifact.stat().st_size / 1e6:.1f}MB)")
    print(f"wrote {data_dir / 'model_card.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
