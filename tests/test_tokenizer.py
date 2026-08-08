"""Behaviour tests for MonTokenizer.

The suite this replaces had six tests and roughly half could not fail:
`test_tokenizer_initialization` asserted `tokenizer is not None` after a
constructor that either raises or returns an object; `test_vocab_size` asserted
`0 < size < 100000`, which passed for the old 4,000-piece model and the 32,000
one alike, so the 8x vocabulary upgrade was invisible; `test_vocab` compared
`len(get_vocab())` against `get_vocab_size()` where the former is built by
iterating the latter; and `test_decode_ids` asserted a substring while claiming a
tolerance for "minor character differences" that did not exist.

Every test below can fail, and each names what breaking it would mean.
"""

from __future__ import annotations

import pytest

from mon_tokenizer import MonTokenizer

MON = "ဂွံအခေါင်အရာမွဲသ္ဂောံဒုင်စသိုင်ကၠာကၠာရ။"
MON_LONG = "ပ္ဍဲအခိင်မာံနဲသဵု မဒှ်ဘဝကွးဘာတက္ကသိုလ်ဂှ် ပါလုပ်ချဳဓရာင်ကၠုင်"
BURMESE = "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသော နိုင်ငံဖြစ်သည်။"
ENGLISH = "The Mon language is spoken by about one million people."
MIXED = "ကျော် page 42 — “quoted” ၏ 🙏 ภาษา ə café ×2 …"

ALL_TEXTS = [MON, MON_LONG, BURMESE, ENGLISH, MIXED]


@pytest.fixture(scope="module")
def tokenizer() -> MonTokenizer:
    return MonTokenizer()


# ---------------------------------------------------------------------------
# Round-trip — exactly, and without stripping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ALL_TEXTS)
def test_ids_round_trip_exactly(tokenizer: MonTokenizer, text: str):
    """`decode(encode(x))` returns x, compared against the normalized form.

    No `.strip()` on either side. An earlier metric stripped both, which would
    have hidden leading- and trailing-whitespace loss — the exact class of defect
    v2 exists to fix.
    """
    result = tokenizer.encode(text)
    assert tokenizer.decode_ids(result["ids"]) == result["text"]


@pytest.mark.parametrize("text", ALL_TEXTS)
def test_pieces_round_trip_exactly(tokenizer: MonTokenizer, text: str):
    result = tokenizer.encode(text)
    assert tokenizer.decode(result["pieces"]) == result["text"]


def test_whitespace_runs_survive(tokenizer: MonTokenizer):
    """v1 lost these: 18,603 collapsed U+0020 across the corpus, and it was the
    entire cause of its 4.42% round-trip failure. Whitespace is content."""
    for text in ["ဂွံ  အခေါင်", "a   b", "one  two   three"]:
        result = tokenizer.encode(text)
        assert tokenizer.decode_ids(result["ids"]) == text


@pytest.mark.xfail(
    strict=True,
    reason="Metaspace(prepend_scheme='always') gives ' abc' and 'abc' identical ids, "
    "so the leading space is lost at encode and no decoder can recover it. Fixing it "
    "needs prepend_scheme='never', which changes every token id and therefore requires "
    "a retrain and a new Hub artifact. Recorded rather than denied.",
)
def test_a_single_leading_space_survives(tokenizer: MonTokenizer):
    """The corpus cannot surface this: export_corpus.py strips every line.

    That is why three independent guards -- the model card, preflight, and
    test_whitespace_runs_survive -- all report 100% round-trip while sharing one
    blind spot. This test is that blind spot, written down. strict=True means it
    fails the suite if a future retrain fixes the behaviour, so the claim and the
    code cannot drift apart again.
    """
    assert tokenizer.encode(" abc")["ids"] != tokenizer.encode("abc")["ids"]


@pytest.mark.xfail(
    strict=True,
    reason="U+2581 is the Metaspace word marker, so a literal one in the input is "
    "indistinguishable from a space after pre-tokenization. Same fix, same cost.",
)
def test_a_literal_word_marker_is_not_read_as_a_space(tokenizer: MonTokenizer):
    """U+2581 LOWER ONE EIGHTH BLOCK is a box-drawing character, so it is plausible
    in exactly the scanned-book and table content this tokenizer targets."""
    assert tokenizer.decode_ids(tokenizer.encode("a▁b")["ids"]) == "a▁b"


def test_pieces_and_ids_describe_the_same_segmentation(tokenizer: MonTokenizer):
    for text in ALL_TEXTS:
        result = tokenizer.encode(text)
        assert len(result["pieces"]) == len(result["ids"])
        assert [tokenizer.id_to_piece(i) for i in result["ids"]] == result["pieces"]


# ---------------------------------------------------------------------------
# Normalization, now enforced by the artifact rather than by a caller
# ---------------------------------------------------------------------------


def test_invisible_characters_do_not_change_the_tokenization(tokenizer: MonTokenizer):
    """v1's headline bug, now structurally impossible.

    Its `encode()` applied no normalization while the model was trained on
    normalized text with `normalization_rule_name="identity"`, so a single ZERO
    WIDTH SPACE turned `['▁ဂွံအခေါင်']` into
    `['▁ဂွံ', '<0xE2>', '<0x80>', '<0x8B>', 'အခေါင်']` — five tokens, three of
    them byte fallback. In v2 the normalizer lives inside `tokenizer.json`.
    """
    clean = tokenizer.encode("ဂွံအခေါင်")["ids"]
    for invisible in ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u00ad"]:
        assert tokenizer.encode(f"ဂွံ{invisible}အခေါင်")["ids"] == clean, (
            f"U+{ord(invisible):04X} changed the tokenization"
        )


def test_encode_reports_the_text_it_actually_encoded(tokenizer: MonTokenizer):
    """`result["text"]` is the normalized string, not the caller's input.

    v1 echoed the input verbatim, which made `assert result["text"] == text`
    tautological and made any decode comparison against it compare with something
    that was never encoded.
    """
    dirty = "ဂွံ\u200bအခေါင်"
    result = tokenizer.encode(dirty)
    assert result["text"] == "ဂွံအခေါင်" != dirty


# ---------------------------------------------------------------------------
# Coverage — the property that makes this usable on scanned pages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["🙏 emoji", "ภาษาไทย", "漢字", "Ωπ√∫", "Ελληνικά", "Кириллица", "עברית", "ə ɪ ʊ"],
)
def test_text_far_outside_the_training_distribution_round_trips(tokenizer: MonTokenizer, text: str):
    """Byte fallback, exercised on scripts the corpus barely contains.

    Without it these are not marked unknown — they are deleted. For a pipeline
    whose output feeds a corpus, silent deletion is the worst failure available.
    """
    result = tokenizer.encode(text)
    assert tokenizer.decode_ids(result["ids"]) == result["text"]


def test_every_byte_piece_is_present(tokenizer: MonTokenizer):
    """`byte_fallback=True` does nothing without them.

    `UnigramTrainer` never emits `<0xNN>` pieces and has no byte-fallback option,
    so they are injected explicitly at the end of training. If that step were
    dropped the model would silently return to emitting `<unk>`.
    """
    vocab = tokenizer.get_vocab()
    missing = [f"<0x{v:02X}>" for v in range(256) if f"<0x{v:02X}>" not in vocab]
    assert not missing, f"{len(missing)} byte pieces missing"


def test_common_characters_are_single_tokens(tokenizer: MonTokenizer):
    """Byte fallback makes everything *representable*; this makes it *cheap*.

    A page number or an English caption should not cost three tokens per
    character because the corpus happened to be thin there.
    """
    vocab = tokenizer.get_vocab()
    for char in "abcXYZ0123456789.,!?-()":
        assert char in vocab or f"▁{char}" in vocab, f"ASCII {char!r} is not a single token"
    for char in "ကခဂဃငာိုေ်ျြွှ၀၁၂၃၄၅၆၇၈၉၊။":
        assert char in vocab or f"▁{char}" in vocab, f"Myanmar {char!r} is not a single token"


# ---------------------------------------------------------------------------
# The model, pinned
# ---------------------------------------------------------------------------


def test_vocabulary_size_is_pinned(tokenizer: MonTokenizer):
    """64,000 trained pieces plus the 256 injected byte pieces.

    v1's bound was `0 < size < 100000`, which its 4,000-piece predecessor also
    satisfied. Pinning is what makes an accidental model swap visible.
    """
    assert tokenizer.get_vocab_size() == 64_256


def test_special_tokens_all_have_real_ids(tokenizer: MonTokenizer):
    """`pad_id` was **-1** in v1.

    The trainer passed `pad_piece="<pad>"` without assigning an id, so
    `PieceToId("<pad>")` returned 0 — `<unk>`. Anyone padding a batch padded with
    unknown tokens and nothing raised.
    """
    assert (tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id) == (0, 1, 2, 3)
    assert len({tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id}) == 4


def test_the_segmentation_of_a_known_string_is_pinned(tokenizer: MonTokenizer):
    """A retrain changes these, which is the point.

    It must be a deliberate version bump with a changelog entry, not something
    that slips through.
    """
    assert tokenizer.encode(MON)["pieces"] == [
        "▁ဂွံ",
        "အခေါင်အရာ",
        "မွဲ",
        "သ္ဂောံ",
        "ဒုင်စသိုင်",
        "ကၠာ",
        "ကၠာရ။",
    ]


@pytest.mark.parametrize(
    ("text", "floor"), [(MON, 4.0), (MON_LONG, 4.0), (BURMESE, 3.5), (ENGLISH, 3.5)]
)
def test_compression_does_not_regress(tokenizer: MonTokenizer, text: str, floor: float):
    """Characters per token, per language, with margin under the measured value.

    Measured on the full val split: Mon 4.798, Burmese 4.106, English 4.112.
    These floors sit below those so single-sentence variation cannot flake, while
    a genuine regression drops far below them.
    """
    result = tokenizer.encode(text)
    ratio = len(result["text"]) / len(result["ids"])
    assert ratio >= floor, f"{ratio:.2f} chars/token is below the {floor} floor"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_a_missing_artifact_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        MonTokenizer(str(tmp_path / "nope.json"))


def test_an_unknown_piece_is_refused_rather_than_dropped(tokenizer: MonTokenizer):
    """Silently skipping it would return a plausible string missing a character."""
    with pytest.raises(ValueError, match="not in the vocabulary"):
        tokenizer.decode(["▁ဂွံ", "not-a-piece"])


def test_the_artifact_is_cached_across_instances():
    """The artifact is 4.8MB and v1 re-read its model on every construction."""
    assert MonTokenizer()._tokenizer is MonTokenizer()._tokenizer


def test_batch_encoding_matches_single_encoding(tokenizer: MonTokenizer):
    assert tokenizer.encode_batch(ALL_TEXTS) == [tokenizer.encode_ids(t) for t in ALL_TEXTS]
