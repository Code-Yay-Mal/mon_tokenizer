#!/usr/bin/env python3
"""Export mon_OCR's bucketed corpus to a JSONL the trainer can read.

Two scripts rather than one, because the dependency sets are disjoint. This one
needs `monocr` — which pulls torch, opencv and lightning — and `train_tokenizer.py`
needs `tokenizers`. Merging them would drag a 2GB training stack into a
repository whose entire runtime dependency list is one Rust wheel.

    # in mon_OCR's environment
    cd ../mon_OCR && uv run python ../mon_tokenizer/scripts/export_corpus.py \
        --output ../mon_tokenizer/build/corpus.jsonl

Three things come from `monocr` and are deliberately not reimplemented here:

* `corpus.bucket_of` — language from the directory, which is ground truth.
  `corpus.language_of` exists but its own docstring says not to use it for
  labelling: 38% of Mon lines carry no Mon-exclusive codepoint and would be
  labelled "unknown".
* `corpus.split_of` — the stable blake2b partition. Using the same function is
  what makes "measured on val" mean the same thing in both repositories.
* `charset.normalize_text` — mon_OCR's definition, which `mon_tokenizer`'s
  `normalization.py` now matches. The equivalence is asserted in
  `tests/test_normalization.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/raw/corpus"),
        help="mon_OCR's bucketed corpus root",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--min-length", type=int, default=8, help="shorter lines carry no segmentation signal"
    )
    args = parser.parse_args()

    try:
        from monocr.charset import normalize_text
        from monocr.corpus import bucket_of, has_percent_encoding, has_url, split_of
    except ImportError as exc:
        print(
            f"cannot import monocr ({exc}).\n"
            "Run this from mon_OCR's environment:\n"
            "  cd ../mon_OCR && uv run python ../mon_tokenizer/scripts/export_corpus.py ...",
            file=sys.stderr,
        )
        return 1

    corpus_root = args.corpus.resolve()
    if not corpus_root.exists():
        print(f"corpus root does not exist: {corpus_root}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    # Value is the bucket the line was first kept under, so a later duplicate can
    # be classified. A plain set cannot tell "this file repeats a line" from
    # "two languages claim the same line", and those are very different facts.
    seen: dict[str, str] = {}
    digests: list[str] = []
    written = chars = duplicate_same = duplicate_cross = dropped_content = 0

    # Bucket priority, not alphabetical read order. A line present in more than
    # one language directory used to be assigned to whichever file `sorted(rglob)`
    # reached first — and because "burmese" < "custom" < "mon", that silently
    # moved **15,853 lines** out of Mon. It made 30.5% of the Burmese eval bucket
    # Mon-corpus text, so every Burmese number was measured on a stratum a third
    # of which was not Burmese.
    #
    # The bias was order-dependent rather than content-dependent: renaming
    # `burmese/` to `myanmar/` would have moved those lines back. Priority makes
    # the assignment a decision instead of an accident, and the count of affected
    # lines is reported rather than hidden.
    priority = {"mon": 0, "burmese": 1, "english": 2}
    files: list[tuple[int, Path, str]] = []
    for path in sorted(corpus_root.rglob("*.txt")):
        try:
            bucket = bucket_of(path, corpus_root)
        except ValueError as exc:
            # bucket_of fails closed on an unmapped directory, by design —
            # defaulting one to Mon is how 24,939 Burmese lines once landed in
            # the Mon bucket. Report and skip rather than guess.
            print(f"skipping: {exc}", file=sys.stderr)
            continue
        # Digest every file that is part of the corpus, including ones whose
        # lines are all duplicates. This used to sit after the `continue`, so a
        # skipped directory fell outside the fingerprint entirely.
        digests.append(hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest())
        files.append((priority.get(bucket, 99), path, bucket))

    with args.output.open("w", encoding="utf-8") as handle:
        for _, path, bucket in sorted(files, key=lambda item: (item[0], str(item[1]))):
            for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = normalize_text(raw.strip())
                if len(line) < args.min_length:
                    continue
                # The same content filters mon_OCR's `load_corpus` applies before
                # rendering an image. This script reads the .txt files directly
                # rather than going through `load_corpus` — that path builds the
                # whole corpus in memory and takes ~30s — and the two had silently
                # diverged: the tokenizer was being fitted on URLs and
                # percent-encoded link remains that the OCR model never sees.
                #
                # The cost was measured. 1,483 pieces of a 64,000-piece
                # vocabulary (2.3%) were fragments like `%E1%80%86%E1%80%`, from
                # 9,009 lines carrying 2.27% of all characters. Those are slots
                # that should hold Mon.
                #
                # `load_corpus` drops two more classes — unbased combining marks
                # and charset-survival fragments — which are not replicated here.
                # They need the charset, and their measured incidence is 0.16%
                # and 0.07%. Recorded as a known divergence rather than left to
                # be discovered again.
                if has_url(line) or has_percent_encoding(line):
                    dropped_content += 1
                    continue
                if (first := seen.get(line)) is not None:
                    if first == bucket:
                        duplicate_same += 1
                    else:
                        duplicate_cross += 1
                    continue
                seen[line] = bucket
                split = split_of(line)
                handle.write(
                    json.dumps({"text": line, "bucket": bucket, "split": split}, ensure_ascii=False)
                    + "\n"
                )
                counts.setdefault(bucket, {}).setdefault(split, 0)
                counts[bucket][split] += 1
                written += 1
                chars += len(line)

    fingerprint = hashlib.blake2b("".join(sorted(digests)).encode(), digest_size=16).hexdigest()
    meta = {
        "source": str(corpus_root),
        "files": len(digests),
        "lines": written,
        "chars": chars,
        "by_bucket_split": counts,
        "source_digest": fingerprint,
        # Two different facts, kept apart. `within_bucket` is ordinary repetition
        # inside one language's files; `across_buckets` is two languages claiming
        # the same line, which is what the priority rule resolves and what
        # contaminates a per-language eval stratum if it is not resolved.
        "dropped_url_or_percent_encoded": dropped_content,
        "duplicates_within_bucket": duplicate_same,
        "duplicates_across_buckets": duplicate_cross,
    }
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"{written:,} unique lines / {chars:,} chars -> {args.output}")
    for bucket, splits in sorted(counts.items()):
        print(f"  {bucket:<9} " + "  ".join(f"{k} {v:,}" for k, v in sorted(splits.items())))
    print(f"{dropped_content:,} lines dropped as URL or percent-encoded")
    print(
        f"duplicates dropped: {duplicate_same:,} within a bucket, "
        f"{duplicate_cross:,} across buckets (kept once, by priority)"
    )
    print(f"corpus digest {fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
