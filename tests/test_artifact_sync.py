"""The artifact here and the artifact on the Hub must be the same bytes.

This repository builds `mon_tokenizer.json`; `hf_mon_tokenizer` publishes it as
`tokenizer.json`. They are two copies of one file, pushed to two places by two
different commands, and nothing but a digest comparison makes them agree.

That is not hypothetical. A release shipped a `tokenizer.json` on the Hub that was
the 4,000-piece predecessor while `tokenizer.model` beside it was the 32,000 one,
and `AutoTokenizer` prefers the former — so anyone following the documented path
got a tokenizer measuring 0.93 chars/token against an advertised 5.22. Nothing
detected it but a user, because the check was a thing someone remembered to do.
`hf_mon_tokenizer/NEXT_STEPS.md` carried "make the identity check automatic" as
its single open item; this file and the `artifact-sync` job in
`.github/workflows/ci.yml` are it.

`scripts/preflight.py` compares versions, corpus digests and the ids produced for
eight probe strings. Probes are a sample: two different artifacts can agree on
eight strings. A digest cannot be sampled.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
ARTIFACT = ROOT / "src/mon_tokenizer/data/mon_tokenizer.json"

# sha256 of every artifact ever published, keyed by the `artifact_version` that
# names it. Append a row when you retrain; never edit one. An edited row is a
# record of nothing — the whole value of this table is that it says what was
# published under a name, including when that turns out to be the wrong thing.
#
# The digest is over the file exactly as it ships. `uv build` copies it without
# rewriting, and the Hub serves it byte for byte, so all three copies hash equal.
ARTIFACT_DIGESTS = {
    "1.0.0": "34d181532eee7e6754bfbad693753ee4660bdda00eb36a2ccc7cd2f372c71282",
}

# The published copy, when a checkout is sitting beside this repository. CI does
# not have one — it fetches from the Hub instead — so its absence is a skip, not
# a failure. A test that fails for being run in the wrong directory teaches
# people to ignore it.
HF_CHECKOUT = ROOT.parent / "hf_mon_tokenizer"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # The artifact is ~4.9MB. Chunked so this does not depend on being able
        # to hold the whole file, and so it stays honest if the vocabulary grows.
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def artifact_version() -> str:
    card = json.loads((ARTIFACT.parent / "model_card.json").read_text(encoding="utf-8"))
    return str(card["artifact_version"])


def test_the_artifact_hashes_to_the_digest_published_under_its_version(artifact_version: str):
    """A retrain that forgets to bump `artifact_version` fails here.

    `tests/test_model_card.py` already pins each version to the corpus digest it
    was trained on, which catches a new corpus. This catches the other half: the
    same corpus, retrained, producing different bytes. Unigram training is not
    bit-reproducible across `tokenizers` releases, so "same inputs" does not imply
    "same artifact" and only the output digest can say so.
    """
    assert ARTIFACT.exists(), f"the artifact is missing from {ARTIFACT}"
    assert artifact_version in ARTIFACT_DIGESTS, (
        f"artifact_version {artifact_version!r} has no row in ARTIFACT_DIGESTS. "
        f"If this is a retrain, add it with the sha256 of the artifact shipped "
        f"under that name."
    )
    expected = ARTIFACT_DIGESTS[artifact_version]
    actual = _sha256(ARTIFACT)
    assert actual == expected, (
        f"the artifact hashes to {actual}, but {expected} was published as "
        f"{artifact_version}. Two different tokenizers under one name is the "
        f"defect this repository shipped once already — bump artifact_version "
        f"and add a row rather than editing the existing one."
    )


def test_the_published_copy_is_byte_identical_to_the_packaged_one(artifact_version: str):
    """Same bytes, not merely same vocabulary size or same ids on a few probes.

    Skipped when `../hf_mon_tokenizer` is absent. CI covers the case that actually
    matters — the copy live on the Hub — in the `artifact-sync` job, because a
    local checkout can be stale in ways the Hub is not.
    """
    published = HF_CHECKOUT / "tokenizer.json"
    if not published.exists():
        pytest.skip(f"no Hugging Face checkout at {HF_CHECKOUT}")

    assert _sha256(published) == ARTIFACT_DIGESTS[artifact_version], (
        f"{published} is not the artifact published as {artifact_version}. "
        f"Publish from this repository rather than editing files there; the two "
        f"copies move in one commit or they do not move."
    )


def test_the_two_model_cards_are_the_same_document():
    """The card and the artifact fall out of sync in different commits, always.

    `preflight.py` checks two of the card's fields across the pair. Everything
    else in it — every measured number a reader quotes — went unchecked, which is
    how a coverage figure could be corrected here and stay wrong on the Hub.
    """
    published = HF_CHECKOUT / "model_card.json"
    if not published.exists():
        pytest.skip(f"no Hugging Face checkout at {HF_CHECKOUT}")

    ours = (ARTIFACT.parent / "model_card.json").read_bytes()
    assert published.read_bytes() == ours, (
        "the two model_card.json copies differ. They are one document published "
        "to two places; copy, do not re-derive."
    )
