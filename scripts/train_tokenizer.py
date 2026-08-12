#!/usr/bin/env python3
"""Train the Mon tokenizer, measure it, and write the artifact with its model card.

    uv run python scripts/train_tokenizer.py --gate      # compare variants, write nothing
    uv run python scripts/train_tokenizer.py --write     # train and ship
    uv run python scripts/train_tokenizer.py --recard    # re-measure what is shipped

Reads the JSONL produced by `export_corpus.py`. Fits on the **train** split and
measures on the **whole val** split — never a head slice. An earlier version took
`val[bucket][:5000]`, which in file order meant `dict/` supplied 47.5% of the Mon
sample while being 8.1% of the split, and `mon_shards/` — 47.2% of the split —
supplied nothing. Dictionary headwords and running prose compress very
differently, so that number described a sample nobody had chosen deliberately.

`--gate` exists because the syllable-atomicity filter is an open question, not a
decision. It trains both variants, prints both, and writes nothing.

`--recard` exists because a card can be wrong about an artifact that is right.
1.0.0 shipped a coverage figure measured over `val["mon"][:5000]` while the same
card declared "whole split, no cap", and the only way to correct it used to be a
retrain — which changes every token id to fix a number that describes the
existing ids perfectly well. This re-runs `evaluate()` against the **shipped**
`mon_tokenizer.json` and rewrites only the card. It never calls `train`, and it
refuses if the corpus on disk is not the one the card was built from.
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
    # No cap. This carried [:5000] while the module docstring above described
    # removing exactly that head-of-file bias, and card["eval"]["sampling"] was
    # hardcoded to "whole split, no cap" -- so the shipped 1.0.0 coverage figure
    # was measured over the first 5,000 Mon val lines and published as though it
    # covered the split. In file order that is not a random sample.
    mon_val = val.get("mon", [])
    out["coverage"] = coverage(tokenizer.get_vocab().keys(), "".join(mon_val))
    out["coverage"]["lines_measured"] = len(mon_val)
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


def _card_paths() -> tuple[Path, Path]:
    data_dir = ROOT / "src/mon_tokenizer/data"
    return data_dir / "mon_tokenizer.json", data_dir / "model_card.json"


def write_card(card_path: Path, card: dict, metrics: dict) -> None:
    """Serialize a card, with the eval block derived rather than asserted.

    A hardcoded sampling string here cannot be wrong in a way any test can catch,
    and it was wrong: it claimed the whole split while coverage read the first
    5,000 lines. `coverage_lines_measured` is the number the driver counted from
    what it actually read, so the claim and the measurement cannot drift apart.
    """
    card["eval"] = {
        "split": "val",
        "sampling": "whole split, no cap",
        "coverage_lines_measured": metrics.get("coverage", {}).get("lines_measured"),
    }
    card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def recard(val: dict[str, list[str]], meta: dict) -> int:
    """Re-measure the shipped artifact and rewrite its card. Trains nothing.

    The artifact is opened read-only and never re-saved: a retrain to correct a
    card would change every token id, invalidate every downstream embedding table
    and force a Hub republish, all to fix a number that describes the ids already
    published. The measurement code path is `evaluate()` — the same one `--write`
    uses — so a card produced here and one produced by a retrain of the same
    corpus differ only where the artifact does.
    """
    from tokenizers import Tokenizer

    artifact, card_path = _card_paths()
    for path in (artifact, card_path):
        if not path.exists():
            print(f"{path} not found — nothing to re-measure", file=sys.stderr)
            return 1

    previous = json.loads(card_path.read_text(encoding="utf-8"))
    # The guard that makes this safe. Re-measuring against a corpus the artifact
    # was not trained on would silently republish another dataset's numbers under
    # this artifact's name, which is a worse defect than the one being fixed.
    if previous["corpus"]["source_digest"] != meta["source_digest"]:
        print(
            f"corpus digest {meta['source_digest']} does not match the card's "
            f"{previous['corpus']['source_digest']}. This corpus did not train this "
            f"artifact; re-measuring against it would publish numbers from another "
            f"dataset. Retrain with --write instead.",
            file=sys.stderr,
        )
        return 1

    tokenizer = Tokenizer.from_file(str(artifact))
    if tokenizer.get_vocab_size() != previous["vocab_size"]:
        print(
            f"artifact has {tokenizer.get_vocab_size():,} pieces but the card records "
            f"{previous['vocab_size']:,} — the two are already out of sync",
            file=sys.stderr,
        )
        return 1

    metrics = evaluate(tokenizer, val)
    show(f"shipped artifact {previous['artifact_version']} re-measured", metrics)
    card = build_model_card(
        TrainConfig(**previous["config"]),
        CorpusStats(
            lines=meta["lines"],
            chars=meta["chars"],
            by_bucket={b: sum(s.values()) for b, s in meta["by_bucket_split"].items()},
            source_digest=meta["source_digest"],
        ),
        metrics,
        previous["artifact_version"],
        tokenizer.get_vocab_size(),
    )
    write_card(card_path, card, metrics)
    print(f"\nwrote {card_path} (artifact untouched)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "build/corpus.jsonl")
    parser.add_argument("--vocab-size", type=int, default=64_000)
    parser.add_argument("--gate", action="store_true", help="compare variants, write nothing")
    parser.add_argument("--atomicity", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--recard",
        action="store_true",
        help="re-measure the shipped artifact and rewrite its card; trains nothing",
    )
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

    # Before any training happens: --recard's whole point is that it does none.
    if args.recard:
        return recard(val, meta)

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
    artifact, card_path = _card_paths()
    artifact.parent.mkdir(parents=True, exist_ok=True)
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
    write_card(card_path, card, metrics)
    print(f"\nwrote {artifact} ({artifact.stat().st_size / 1e6:.1f}MB)")
    print(f"wrote {card_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
