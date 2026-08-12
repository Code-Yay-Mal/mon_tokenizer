"""Tests for the code that builds the artifact, not the artifact it built.

`trainer.py` had **zero** tests — open audit finding M6. It is the only file in
the package that can silently destroy the property the project exists for: the
byte-fallback guarantee is not a flag, it is 256 vocabulary entries injected by
hand at `_finalise`, because `UnigramTrainer` never emits them and
`models.Unigram(byte_fallback=True)` falls back to pieces that are not there.
Delete that injection and every test in `test_tokenizer.py` still passes, because
they all load the *shipped* `tokenizer.json` — which already has the pieces in it.
The next retrain would be the one that ships text-deleting behaviour.

So these fit real tokenizers on a tiny synthetic corpus and assert what the
training code guarantees. The whole file trains four models and runs in well under
a second; `build/corpus.jsonl` is 209 MB and is never touched.

Every claim here was measured against the fixtures below, not reasoned about:
the counterfactual in `test_without_the_injection_characters_are_deleted_not_marked`
is the real output of a model built exactly like the shipped one minus the 256
pieces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mon_tokenizer import model_card
from mon_tokenizer.syllable import starts_mid_syllable
from mon_tokenizer.trainer import (
    BOOK_PUNCTUATION,
    BYTE_PIECES,
    BYTE_SCORE,
    MYANMAR_RANGES,
    SPECIAL_TOKENS,
    WORD_MARKER,
    CorpusStats,
    TrainConfig,
    build_model_card,
    corpus_fingerprint,
    filter_partial_syllable_pieces,
    guaranteed_alphabet,
    train,
)

MON = "ဂွံအခေါင်အရာမွဲသ္ဂောံဒုင်စသိုင်ကၠာကၠာရ။"
MON_LONG = "ပ္ဍဲအခိင်မာံနဲသဵု မဒှ်ဘဝကွးဘာတက္ကသိုလ်ဂှ် ပါလုပ်ချဳဓရာင်ကၠုင်"
BURMESE = "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသော နိုင်ငံဖြစ်သည်။"
ENGLISH = "The Mon language is spoken by about one million people."

# Repeated because Unigram needs frequency counts to work with, not because the
# text matters. Deliberately narrow: no Thai, no emoji, no Greek, no Myanmar
# digits — the tests below rely on those being *absent* from training, which is
# what makes "it still round-trips" mean something.
CORPUS = [MON, MON_LONG, BURMESE, ENGLISH] * 30

# Above `len(guaranteed_alphabet())` — 338 — or `UnigramTrainer` raises "the
# vocabulary is not large enough to contain all chars". Small enough that each
# fit costs ~10 ms.
TINY_VOCAB = 500

BYTE_PIECE_SET = frozenset(BYTE_PIECES)


@pytest.fixture(scope="module")
def tiny():
    """The default configuration, at a size that fits in milliseconds."""
    return train(CORPUS, TrainConfig(vocab_size=TINY_VOCAB))


@pytest.fixture(scope="module")
def tiny_vocab(tiny) -> list[tuple[str, float]]:
    """`(piece, score)` pairs as the model stores them, scores included.

    `get_vocab()` throws the scores away, and the scores are half of what the
    byte-piece injection promises: present *and* losing to everything real.
    """
    return [(piece, score) for piece, score in json.loads(tiny.to_str())["model"]["vocab"]]


# ---------------------------------------------------------------------------
# Byte-piece injection — the guarantee that has no flag
# ---------------------------------------------------------------------------


def test_all_256_byte_pieces_are_injected(tiny_vocab: list[tuple[str, float]]):
    """`UnigramTrainer` emits none of these, and it has no byte-fallback option.

    A partial injection is the dangerous case: 255 of 256 still round-trips
    almost everything, so it would pass any spot check and lose exactly the text
    containing the missing byte.
    """
    present = {piece for piece, _ in tiny_vocab} & BYTE_PIECE_SET
    assert len(present) == 256, f"{256 - len(present)} byte pieces were not injected"


def test_byte_pieces_lose_to_every_trained_piece(tiny_vocab: list[tuple[str, float]]):
    """`BYTE_SCORE = -100.0` is only correct if nothing real scores below it.

    Viterbi maximises total score, so a byte piece scoring above any ordinary
    piece would be chosen in preference to it and compression would collapse.
    Measured on this corpus: trained scores run -9.313 to -2.070, so the -100.0
    floor has ~90 log-units of margin. Asserted rather than assumed, because the
    constant is a guess about a distribution.
    """
    scores = dict(tiny_vocab)
    injected = [scores[piece] for piece in BYTE_PIECES]
    assert set(injected) == {BYTE_SCORE}, "an injected byte piece does not carry BYTE_SCORE"

    trained = [
        score
        for piece, score in tiny_vocab
        if piece not in BYTE_PIECE_SET and piece not in SPECIAL_TOKENS
    ]
    assert min(trained) > BYTE_SCORE, (
        f"the lowest trained score is {min(trained)}, at or below the byte floor "
        f"{BYTE_SCORE} — byte fallback would start winning against real pieces"
    )


@pytest.mark.parametrize("text", ["🙏 emoji", "ภาษาไทย", "漢字とかな", "café", "Ωπ√", "ə ɪ ʊ"])
def test_text_absent_from_training_round_trips_through_the_injected_pieces(tiny, text: str):
    """None of these scripts is in `CORPUS`, and all of them survive.

    This is the property `mon-vlm` depends on: a scanned page meets the tokenizer
    with whatever is printed on it, and the tokenizer must not be able to
    silently drop part of it.
    """
    assert tiny.decode(tiny.encode(text).ids) == text


def test_without_the_injection_characters_are_deleted_not_marked(tiny, tiny_vocab):
    """The counterfactual, so the guarantee is attributed to the right code.

    This builds the shipped configuration minus the 256 pieces — same normalizer,
    same pre-tokenizer, same decoder, special tokens re-added, still
    `byte_fallback=True` — which is exactly what `_finalise` produces if the
    injection block is removed. Measured output of that model:

        '🙏 emoji'  ->  ' emoji'
        'ภาษาไทย'   ->  ''
        'café'      ->  'caf'

    Not `<unk>`, not an exception: the characters are **gone**, and the result
    still reads as fluent text. For a pipeline whose output feeds a corpus, that
    is the worst failure available, and it is the reason this file exists.

    The `add_special_tokens` line is load-bearing and was missing from the first
    draft of this test, which then decoded `'<unk> emoji'` and looked like a
    tolerable failure. `Tokenizer.decode` defaults to `skip_special_tokens=True`,
    so once `<unk>` is registered as special it never reaches the output — the
    silence is produced by the two facts *together*, and a model that merely
    lacks byte pieces looks fine in a REPL where nobody registered them.
    """
    from tokenizers import AddedToken, Tokenizer, models

    from mon_tokenizer import trainer

    without_bytes = [(p, s) for p, s in tiny_vocab if p not in BYTE_PIECE_SET]
    unk_id = next(i for i, (p, _) in enumerate(without_bytes) if p == "<unk>")
    naked = trainer._attach(
        Tokenizer(models.Unigram(without_bytes, unk_id=unk_id, byte_fallback=True))
    )
    naked.add_special_tokens([AddedToken(token, special=True) for token in SPECIAL_TOKENS])

    for text in ["🙏 emoji", "ภาษาไทย", "café"]:
        lost = naked.decode(naked.encode(text).ids)
        assert lost != text, (
            f"{text!r} survived a vocabulary with no byte pieces — this test no "
            f"longer isolates the injection, so the one above proves less than it claims"
        )
        assert "<unk>" not in lost, "the failure mode is silent deletion, not a marker"
        # And the same text through the real thing.
        assert tiny.decode(tiny.encode(text).ids) == text


def test_byte_fallback_off_injects_nothing(tiny):
    """The flag has to mean something in both directions.

    An injection that ran unconditionally would make `byte_fallback=False`
    indistinguishable from `True` in the vocabulary, and the `--gate` comparison
    in `train_tokenizer.py` would be comparing a variable against itself.
    """
    off = train(CORPUS, TrainConfig(vocab_size=TINY_VOCAB, byte_fallback=False))
    assert not [p for p in BYTE_PIECES if off.token_to_id(p) is not None]
    assert off.get_vocab_size() == tiny.get_vocab_size() - 256


def test_the_special_tokens_survive_the_rebuild(tiny):
    """`_finalise` builds a *fresh* `Tokenizer`, which has no added tokens.

    The first implementation dropped all four here, so `<pad>` resolved to
    `<unk>` — the exact v1 defect (`pad_id` was -1 and padding a batch padded
    with unknown tokens) reintroduced by the fix for a different one.
    """
    assert [tiny.token_to_id(token) for token in SPECIAL_TOKENS] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# guaranteed_alphabet — what stays cheap regardless of corpus frequency
# ---------------------------------------------------------------------------


def test_the_guaranteed_alphabet_is_exactly_its_three_parts():
    """96 + 223 + 19 = 338, and the arithmetic is the point.

    96 is `string.printable` (100) minus tab, CR, VT and FF. 223 is every
    codepoint in `MYANMAR_RANGES` — 160 + 31 + 32, unassigned ones included,
    because a range is easier to check than a curated list and an unassigned
    codepoint costs one vocabulary slot. 19 is `BOOK_PUNCTUATION`.

    A count rather than a spot check: dropping Myanmar Extended-A would still
    leave `ကခဂ` present, so any test that only samples common characters would
    pass while Mon-exclusive letters silently became three bytes each.
    """
    alphabet = guaranteed_alphabet()
    myanmar = sum(end - start + 1 for start, end in MYANMAR_RANGES)
    assert myanmar == 223
    assert len(alphabet) == 96 + myanmar + len(BOOK_PUNCTUATION) == 338

    # `initial_alphabet` takes characters. A multi-character entry is accepted
    # and then silently ignored by the trainer.
    assert all(len(entry) == 1 for entry in alphabet)
    assert alphabet == sorted(set(alphabet)), "expected a sorted, deduplicated list"


def test_tab_and_the_control_whitespace_are_excluded_but_space_and_newline_are_not():
    """Whitespace is content, and these are two different kinds of it.

    U+0020 and U+000A appear in the corpus and must be tokens. Tab, CR, VT and FF
    are stripped or folded upstream, so a slot for each is a slot not spent on Mon.
    """
    alphabet = set(guaranteed_alphabet())
    assert {" ", "\n"} <= alphabet
    assert not alphabet & {"\t", "\r", "\x0b", "\x0c"}


def test_a_myanmar_digit_absent_from_the_corpus_is_still_one_token(tiny):
    """The whole purpose of `initial_alphabet`, on a case that can be checked.

    `၀` U+1040 does not occur anywhere in `CORPUS`, so frequency alone would
    leave it to byte fallback at three tokens. It is a single token because the
    guaranteed alphabet forced it in — and `Ω`, which is *not* in that alphabet
    and equally absent, costs the three bytes.
    """
    assert not any("၀" in line for line in CORPUS), "the fixture no longer tests anything"

    assert tiny.token_to_id("၀") is not None
    assert tiny.encode("၀").tokens == [WORD_MARKER, "၀"]
    assert tiny.encode("Ω").tokens == [WORD_MARKER, "<0xCE>", "<0xA9>"]


# ---------------------------------------------------------------------------
# filter_partial_syllable_pieces
# ---------------------------------------------------------------------------


def test_only_pieces_that_can_never_start_a_syllable_are_dropped():
    """Sound, not complete — and the incomplete half is deliberate.

    `ာ` and `ျ` cannot begin a syllable: a vowel sign has nothing to attach to
    unless a base character precedes it in another token. `င` can *look* like a
    mid-syllable fragment (`ဒုင်` ends with one) but is a legitimate consonant on
    its own, so it stays. Dropping it would cost slots for a guarantee this
    filter does not make.
    """
    vocab: list[tuple[str, float]] = [
        ("▁ဂွံ", -3.0),  # word-marked, starts with a base
        ("ကၠာ", -4.0),
        ("င", -5.0),  # sometimes a killed final; never removed
        ("ာ", -6.0),  # vowel sign: mid-syllable only
        ("ျ", -7.0),  # medial: mid-syllable only
        ("▁ေ", -8.0),  # the marker does not exempt it
        ("The", -9.0),
    ]
    kept, dropped = filter_partial_syllable_pieces(vocab)

    assert dropped == 3
    assert [piece for piece, _ in kept] == ["▁ဂွံ", "ကၠာ", "င", "The"]
    assert not any(starts_mid_syllable(piece, WORD_MARKER) for piece, _ in kept)
    assert len(kept) + dropped == len(vocab), "pieces were invented or lost, not filtered"


def test_the_filter_never_drops_a_special_or_byte_piece():
    """`_finalise` raises if `<unk>` is filtered away, and this is why it never is.

    `<0xE1>` begins with `<`, not a combining mark, but the guard in
    `starts_mid_syllable` is explicit rather than incidental — a piece named
    `<0x...>` must never be a candidate no matter how the check is rewritten.
    """
    vocab: list[tuple[str, float]] = [(token, -1.0) for token in SPECIAL_TOKENS]
    vocab += [(piece, BYTE_SCORE) for piece in BYTE_PIECES]
    kept, dropped = filter_partial_syllable_pieces(vocab)

    assert dropped == 0
    assert len(kept) == len(vocab)


def test_atomicity_drops_pieces_and_keeps_byte_fallback_intact(tiny):
    """One rebuild path, both variants — the reason `_finalise` is unconditional.

    The first version applied byte fallback only when atomicity was on, so
    `--gate` compared `atomicity + byte_fallback` against neither and could not
    isolate what it was gating. Measured on this corpus: 612 pieces off, 546 on,
    66 dropped, 256 byte pieces in both.
    """
    atomic = train(CORPUS, TrainConfig(vocab_size=TINY_VOCAB, enforce_syllable_atomicity=True))

    assert atomic.get_vocab_size() < tiny.get_vocab_size(), "the filter dropped nothing"
    assert not [p for p in BYTE_PIECES if atomic.token_to_id(p) is None]
    survivors = [piece for piece in atomic.get_vocab() if starts_mid_syllable(piece, WORD_MARKER)]
    assert not survivors, f"{len(survivors)} mid-syllable pieces survived the filter"
    assert atomic.decode(atomic.encode("ภาษาไทย").ids) == "ภาษาไทย"


# ---------------------------------------------------------------------------
# corpus_fingerprint
# ---------------------------------------------------------------------------


@pytest.fixture
def corpus_files(tmp_path: Path) -> list[Path]:
    first = tmp_path / "mon_shards.txt"
    second = tmp_path / "dict.txt"
    first.write_text("ဂွံအခေါင်အရာမွဲ\n", encoding="utf-8")
    second.write_text("The Mon language\n", encoding="utf-8")
    return [first, second]


def test_the_fingerprint_is_deterministic_and_order_independent(corpus_files: list[Path]):
    """It goes in the model card, so an unstable digest makes the card unusable.

    Order-independence is required, not incidental: the caller globs a directory
    and glob order is filesystem order. A digest that changed with it would make
    `--recard` refuse on the same corpus it was built from.
    """
    digest = corpus_fingerprint(corpus_files)
    assert digest == corpus_fingerprint(corpus_files)
    assert digest == corpus_fingerprint(list(reversed(corpus_files)))
    assert len(digest) == 32, "expected a blake2b-128 hex digest, as the card records"


def test_a_file_edited_in_place_changes_the_fingerprint(corpus_files: list[Path]):
    """The case names and mtimes miss, and the reason this hashes content.

    `train_tokenizer.py --recard` refuses when the corpus on disk does not match
    the card's digest, because re-measuring against a different dataset would
    publish another corpus's numbers under this artifact's name. That refusal is
    only as good as this sensitivity.
    """
    before = corpus_fingerprint(corpus_files)
    corpus_files[0].write_text("ဂွံအခေါင်အရာမွဲ\nand one more line\n", encoding="utf-8")
    assert corpus_fingerprint(corpus_files) != before


def test_a_dropped_or_added_file_changes_the_fingerprint(corpus_files: list[Path], tmp_path: Path):
    both = corpus_fingerprint(corpus_files)
    assert corpus_fingerprint(corpus_files[:1]) != both

    extra = tmp_path / "burmese.txt"
    extra.write_text("မြန်မာနိုင်ငံ\n", encoding="utf-8")
    assert corpus_fingerprint([*corpus_files, extra]) != both


def test_renaming_a_file_does_not_change_the_fingerprint(corpus_files: list[Path]):
    """Documented behaviour, asserted so it stays a decision.

    The digest describes the *text* that was trained on. Renaming `dict.txt` to
    `dictionary.txt` changes no training input, and a fingerprint that moved
    would force a spurious retrain — or, worse, teach whoever hit it to bypass
    the `--recard` guard.
    """
    before = corpus_fingerprint(corpus_files)
    renamed = corpus_files[1].with_name("dictionary.txt")
    corpus_files[1].rename(renamed)
    assert corpus_fingerprint([corpus_files[0], renamed]) == before


# ---------------------------------------------------------------------------
# build_model_card
# ---------------------------------------------------------------------------


@pytest.fixture
def card() -> dict:
    return build_model_card(
        TrainConfig(vocab_size=TINY_VOCAB),
        CorpusStats(lines=4, chars=180, by_bucket={"mon": 3}, source_digest="0" * 32),
        {"mon": {"chars_per_token": 4.686}},
        artifact_version="9.9.9",
        vocab_size=TINY_VOCAB + 256,
    )


def test_the_card_has_no_bare_version_key(card: dict):
    """`artifact_version` names the trained model; `__version__` names the package.

    A `version` key sitting next to `__version__` with nothing saying which it
    tracked is an invitation to couple them, and a bugfix release of the library
    would then require republishing the Hub artifact to change one string.
    `preflight.py` checks this on the shipped card; this checks the function that
    produces it, so the two cannot be fixed in one place and left broken in the
    other.
    """
    assert "version" not in card
    assert card["artifact_version"] == "9.9.9"


def test_the_recorded_vocab_size_is_the_real_one_not_the_requested_one(card: dict):
    """These differ by the injected bytes, and conflating them is how a card lies.

    The shipped card records 64,256 with `config.vocab_size` 64,000, and
    `preflight.py` asserts the card's figure equals what the installed artifact
    reports. Taking it from the config instead would make that check compare the
    request against itself.
    """
    assert card["vocab_size"] == TINY_VOCAB + 256
    assert card["config"]["vocab_size"] == TINY_VOCAB
    assert card["vocab_size"] != card["config"]["vocab_size"]


def test_the_card_round_trips_back_into_the_dataclasses_it_came_from(card: dict):
    """`--recard` does exactly this: `TrainConfig(**previous["config"])`.

    So a field added to `TrainConfig` and not handled, or a card written with a
    key the dataclass does not take, breaks re-carding — a path that is only ever
    exercised against the real 209 MB corpus, months after the change.
    """
    assert TrainConfig(**card["config"]) == TrainConfig(vocab_size=TINY_VOCAB)
    assert CorpusStats(**card["corpus"]).source_digest == "0" * 32


def test_the_card_carries_the_measurements_and_the_caveats_that_qualify_them(card: dict):
    """Compression is a property of a corpus. A number published without that
    sentence beside it is the one v1 shipped: 5.22 chars/token, unreproducible,
    quoted everywhere."""
    assert card["metrics"]["mon"]["chars_per_token"] == 4.686
    assert set(card["notes"]) == {
        "compression_is_corpus_dependent",
        "violation_rate",
        "roundtrip",
        "byte_fallback",
    }
    assert card["corpus"]["source_digest"] == "0" * 32


# ---------------------------------------------------------------------------
# TrainConfig defaults
# ---------------------------------------------------------------------------


def test_the_default_vocab_size_is_the_one_the_shipped_artifact_was_trained_at():
    """This read 48,000 while the artifact is 64,000 + 256 — audit finding A5.

    `train(lines)` with no config therefore built a tokenizer 16,000 pieces
    smaller than anything this repository ships or measures, and nothing said so:
    `scripts/train_tokenizer.py` passes `--vocab-size` explicitly, so the driver
    never touched the default and the drift was invisible to every existing test.

    Tied to the card rather than pinned to a literal, so a retrain at a new size
    fails here exactly once — when the two genuinely disagree — instead of
    needing this test edited every time.
    """
    assert TrainConfig().vocab_size == model_card()["config"]["vocab_size"]


def test_training_on_nothing_is_refused_rather_than_producing_an_empty_model():
    """A tokenizer fitted on zero lines is a 4-piece vocabulary that ships."""
    with pytest.raises(ValueError, match="no lines to train on"):
        train([])
