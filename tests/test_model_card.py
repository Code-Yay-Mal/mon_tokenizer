"""The model card must describe the artifact shipped beside it.

v1's numbers went stale in every place they were written. The README claimed 5.22
chars/token, 100% round-trip and 0.00% byte fallback; none of the three
reproduced on the corpus. The CHANGELOG gave a different compression figure
(5.17) and described the corpus three different ways (41.4M / 92.8M / 177M
characters).

The fix is not discipline, it is a test: these re-measure the *shipped* artifact
and fail when the recorded numbers drift from what it actually does.
"""

from __future__ import annotations

import pytest

from mon_tokenizer import MonTokenizer, model_card
from mon_tokenizer.metrics import measure

# Held-out-ish text: real Mon, Burmese, English and a mixed line. Small enough to
# run in milliseconds, varied enough that a broken artifact cannot pass.
SAMPLES = {
    "mon": [
        "ဂွံအခေါင်အရာမွဲသ္ဂောံဒုင်စသိုင်ကၠာကၠာရ။",
        "ပ္ဍဲအခိင်မာံနဲသဵု မဒှ်ဘဝကွးဘာတက္ကသိုလ်ဂှ် ပါလုပ်ချဳဓရာင်ကၠုင်",
    ],
    "burmese": ["မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသော နိုင်ငံဖြစ်သည်။"],
    "english": ["The Mon language is spoken by about one million people."],
    "mixed": ["ကျော် page 42 — “quoted” ၏ 🙏 ภาษา ə café ×2 …"],
}


@pytest.fixture(scope="module")
def tokenizer() -> MonTokenizer:
    return MonTokenizer()


@pytest.fixture(scope="module")
def card() -> dict:
    return model_card()


def test_the_card_describes_this_artifact(tokenizer: MonTokenizer, card: dict):
    """Vocabulary size is the cheapest thing to get wrong and the loudest to catch."""
    assert card["vocab_size"] == tokenizer.get_vocab_size()
    assert card["algorithm"] == "unigram"
    assert set(card["languages"]) == {"mnw", "mya", "eng"}


# ---------------------------------------------------------------------------
# The artifact version names the model, not the package
# ---------------------------------------------------------------------------

# Every artifact ever shipped, with the corpus digest it was trained on. Append
# a row when you retrain; never edit one, because the pairing is the record of
# what was published under that name.
ARTIFACT_LINEAGE = {
    "1.0.0": "11941a573a5e0c618edbc91d34dd787d",
}


def test_the_artifact_version_names_the_corpus_it_was_trained_on(card: dict):
    """Retraining without bumping this publishes two tokenizers under one name.

    That is precisely the defect this release fixed on the Hub, where
    `tokenizer.json` was the 4,000-piece predecessor while the card advertised
    32,000: two different models, one identity, and nothing that could tell them
    apart after the fact.

    This is a lineage check rather than an equality check. A test asserting
    `card["artifact_version"] == "1.0.0"` would have to be edited on every
    retrain, and a test you edit to make it pass is not a test.
    """
    recorded = card["artifact_version"]
    assert recorded in ARTIFACT_LINEAGE, (
        f"artifact_version {recorded!r} has no row in ARTIFACT_LINEAGE. If this "
        f"is a retrain, add it with the corpus digest it was trained on."
    )
    assert card["corpus"]["source_digest"] == ARTIFACT_LINEAGE[recorded], (
        f"artifact_version {recorded!r} was published against corpus "
        f"{ARTIFACT_LINEAGE[recorded]}, but this card records "
        f"{card['corpus']['source_digest']}. A new corpus is a new artifact — "
        f"bump artifact_version rather than reusing one."
    )


def test_the_artifact_version_is_not_coupled_to_the_package_version(card: dict):
    """These are independent numbers, and the card must not re-merge them.

    A bugfix release of the library ships a byte-identical tokenizer. If the
    card carried the package version, that release would require regenerating
    and republishing the Hugging Face artifact to change one string — churn that
    invites skipping the republish, which is how the two fall out of sync.

    The check is structural, not numeric: they are equal today (both 1.0.0) and
    that is fine. What must not come back is a `version` key that silently means
    one or the other.
    """
    assert "version" not in card, (
        "the card has a bare `version` key again. It sat next to `__version__` "
        "with nothing saying which it tracked; the field is `artifact_version`."
    )
    assert "artifact_version" in card


def test_the_card_records_the_corpus_it_was_trained_on(card: dict):
    """A trained artifact with no record of its inputs is not reproducible.

    v1 had no machine-readable record at all, which is why its corpus ended up
    described three different ways in prose.
    """
    corpus = card["corpus"]
    assert corpus["lines"] > 500_000
    assert corpus["chars"] > 50_000_000
    assert len(corpus["source_digest"]) == 32, "expected a blake2b-128 hex digest"
    assert set(corpus["by_bucket"]) == {"mon", "burmese", "english"}


def test_the_whole_val_split_was_measured(card: dict):
    """Guards against a sampling cap creeping back in.

    An earlier driver measured `val[bucket][:5000]` in file order, which gave
    `dict/` 47.5% of the Mon sample while it is 8.1% of the split, and left
    `mon_shards/` — 47.2% of the split — contributing nothing.

    This used to assert only that the string "no cap" appeared in the card. That
    string was hardcoded next to a coverage call that still carried `[:5000]`, so
    the test asserted a claim the code contradicted, and 1.0.0 shipped a coverage
    figure measured over 5,000 lines while declaring the whole split. Assert the
    count instead: a number the driver derives from what it actually read.
    """
    assert card["eval"]["split"] == "val"
    assert "no cap" in card["eval"]["sampling"]

    measured = card["eval"].get("coverage_lines_measured")
    assert measured is not None, (
        "the card must record how many lines coverage actually read; a prose "
        "claim with no number behind it is what allowed this to be wrong"
    )
    assert measured > 5_000, (
        f"coverage measured {measured:,} lines. At exactly 5,000 the old cap is "
        f"back; the Mon val split is far larger than that."
    )
    assert measured == card["metrics"]["mon"]["lines"], (
        f"coverage read {measured:,} lines but the Mon val split is "
        f"{card['metrics']['mon']['lines']:,}. Coverage must cover all of it, "
        f"not a prefix."
    )


def test_byte_fallback_is_real_and_not_merely_declared(tokenizer: MonTokenizer, card: dict):
    """The config saying `true` proved nothing — the first implementation never
    applied it, so the card would have asserted a property the artifact lacked.

    This checks the vocabulary actually contains the byte pieces the model would
    fall back to. `models.Unigram(byte_fallback=True)` still emits `<unk>` if they
    are absent, and `UnigramTrainer` never creates them.
    """
    assert card["config"]["byte_fallback"] is True
    vocab = tokenizer.get_vocab()
    missing = [f"<0x{value:02X}>" for value in range(256) if f"<0x{value:02X}>" not in vocab]
    assert not missing, f"{len(missing)} byte pieces absent; byte fallback cannot work"


def test_text_outside_the_vocabulary_round_trips_rather_than_vanishing(tokenizer: MonTokenizer):
    """The failure this project exists to prevent.

    Without byte fallback, unknown characters are not marked — they are deleted.
    `'ကျော် page 42 — “quoted” ၏ 🙏 ภาษา ə café ×2 …'` decoded to
    `'ကော် e   oed       '` on the first build: fluent-looking output with content
    silently removed.
    """
    for text in ["🙏🏽 emoji", "ภาษาไทย", "漢字とかな", "Ω≈ç√∫˜µ", "ကျော် 🙏 ภาษา"]:
        encoded = tokenizer.encode(text)
        assert tokenizer.decode_ids(encoded["ids"]) == encoded["text"], text


@pytest.mark.parametrize("bucket", ["mon", "burmese", "english", "mixed"])
def test_the_samples_round_trip(tokenizer: MonTokenizer, bucket: str):
    """Fidelity on real text of each kind. This one can genuinely fail."""
    fallback = frozenset(i for piece, i in tokenizer.get_vocab().items() if piece.startswith("<0x"))
    fresh = measure(
        lambda t: (tokenizer.encode(t)["pieces"], tokenizer.encode(t)["ids"]),
        lambda _pieces, ids: tokenizer.decode_ids(ids),
        SAMPLES[bucket],
        fallback,
    )
    assert fresh.roundtrip == 1.0, f"{bucket} does not round-trip"


def test_the_recorded_compression_is_within_plausible_bounds(card: dict):
    """A guard on the card itself, not a re-measurement.

    An earlier version of this test compared these figures against a two-line
    sample and allowed a tolerance. That was a bad test: the card's numbers come
    from the whole val split (Mon 30,060 lines) and a hand-picked sample cannot
    be expected to match it — measured deltas were 2.3 and 2.1 chars/token, both
    legitimate. A test whose tolerance has to be widened until it passes is
    measuring nothing.

    Reproducing the card properly needs the corpus, which is not in the wheel;
    that is `scripts/train_tokenizer.py`'s job. What *is* checkable here is that
    the recorded numbers are in a range a working tokenizer can produce, so a
    hand-edited or stale card fails. The artifact's identity is pinned separately
    and exactly by `test_the_segmentation_of_a_known_string_is_pinned`.
    """
    floors = {"mon": 4.0, "burmese": 3.5, "english": 3.5, "mixed": 3.0}
    for bucket, floor in floors.items():
        recorded = card["metrics"][bucket]["chars_per_token"]
        assert floor <= recorded < 12.0, f"{bucket} records {recorded} chars/token"


def test_every_recorded_stratum_round_trips_completely(card: dict):
    """100% on all four strata is the claim; it is also the thing most worth guarding.

    A retrain that regressed fidelity would still look fine on compression.
    """
    for bucket, metrics in card["metrics"].items():
        if bucket == "coverage":
            continue
        assert metrics["roundtrip"] == 1.0, f"{bucket} round-trip is {metrics['roundtrip']}"


def test_the_syllable_denominator_is_reported_with_the_rate(card: dict):
    """A violation rate over a tiny denominator is noise wearing a percentage sign.

    Full English val contains zero multi-character Myanmar syllables, so its rate
    is `0.00%` over nothing — which must be readable as "not measured" rather than
    "perfect".
    """
    for bucket, metrics in card["metrics"].items():
        if bucket == "coverage":
            continue
        assert "syllables" in metrics, f"{bucket} reports a rate without its denominator"
    assert card["metrics"]["english"]["syllables"] == 0
