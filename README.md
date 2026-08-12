# Mon Tokenizer

<p align="center">
  <a href="https://pypi.org/project/mon-tokenizer/"><img src="https://img.shields.io/pypi/v/mon-tokenizer" alt="PyPI"></a>
  <a href="https://pypi.org/project/mon-tokenizer/"><img src="https://img.shields.io/pypi/pyversions/mon-tokenizer" alt="Python versions"></a>
  <a href="https://huggingface.co/janakhpon/mon_tokenizer"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow" alt="Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Ruff"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/badge/package%20manager-uv-purple" alt="uv"></a>
  <img src="https://img.shields.io/badge/language-Mon%20(mnw)-orange" alt="Mon language">
</p>

<p align="center">
  <strong>Unigram tokenizer for Mon (mnw), Burmese and English, with full byte fallback</strong>
</p>

Mon mixes with Burmese constantly and English routinely, so all three are trained
on and measured separately. Anything else on the page (Thai, emoji, IPA, CJK)
round-trips through byte fallback rather than being lost.

```bash
pip install mon-tokenizer
```

```python
from mon_tokenizer import MonTokenizer

tokenizer = MonTokenizer()
result = tokenizer.encode("ဂွံအခေါင်အရာမွဲသ္ဂောံဒုင်စသိုင်ကၠာကၠာရ။")

result["pieces"]  # ['▁ဂွံ', 'အခေါင်အရာ', 'မွဲ', 'သ္ဂောံ', 'ဒုင်စသိုင်', 'ကၠာ', 'ကၠာရ။']
result["ids"]  # token ids
result["text"]  # the normalized string that was encoded

assert tokenizer.decode_ids(result["ids"]) == result["text"]
```

Also on the Hub, producing identical ids:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("janakhpon/mon_tokenizer")
```

## Measured

Vocabulary 64,256 · trained on the train split of 893,936 lines / 85.8M
characters · scored on the **whole** validation split.

| stratum | chars/token | syllable violations | round-trip | byte fallback |
| :--- | ---: | ---: | ---: | ---: |
| Mon | 4.686 | 1.07% *(n=492,469)* | **100%** | 0.00% |
| Burmese | 4.117 | 0.93% *(n=25,546)* | **100%** | 0.00% |
| English | 4.112 | — *(n=0)* | **100%** | 0.02% |
| mixed script | 3.804 | 0.81% *(n=28,133)* | **100%** | 0.19% |

98.74% of the distinct characters in the Mon validation split are single tokens —
392 of 397, over all 29,600 lines. The remaining five (a combining diaeresis
below, a Greek capital pi, two Sinhala letters and one emoji; seven occurrences
in 2.28M characters) round-trip through byte fallback at one to four tokens each.

Compression describes a corpus, not a tokenizer — quote it with the corpus.
Violations count token boundaries inside a Myanmar syllable; the denominator is
given because English legitimately has none. Round-trip compares after
normalization, which is a deliberate transform.

`model_card()` returns these machine-readably, and a test fails if they drift
from what the artifact does.

## Round-trip, and its two known limits

```python
for text in ["🙏 emoji", "ภาษาไทย", "漢字", "Ωπ√∫", "ကျော် page 42 — “quoted” ၏"]:
    r = tokenizer.encode(text)
    assert tokenizer.decode_ids(r["ids"]) == r["text"]
```

Without byte fallback, characters outside the vocabulary aren't flagged — they're
**deleted**, leaving fluent-looking output with content missing. That matters if
you feed OCR output back into a corpus.

Two inputs are known not to round-trip. The loss happens at `encode`, so no
decoder can recover it:

```python
tokenizer.encode(" abc")["ids"] == tokenizer.encode("abc")["ids"]  # True
tokenizer.encode("a b")["ids"] == tokenizer.encode("a▁b")["ids"]  # True
```

A single leading space is dropped, and a literal U+2581 is read as a space. The
table above still holds: the corpus it was measured on is stripped per line and
contains no U+2581, so it excludes both cases by construction rather than
disproving them. Both are recorded as xfail tests.

Fixing this needs `prepend_scheme="never"`, which changes every token id and so
requires a retrain and a new Hub artifact. Deferred on purpose, because id
stability is worth more than these two cases. Interior and trailing whitespace,
tabs, newlines, NUL and control characters all round-trip.

## Normalization

Applied automatically, and stored **inside** the artifact so it cannot drift from
the model: invisible characters stripped, Unicode space separators folded to
`U+0020`, then NFC. Runs of spaces are preserved.

## API

| | |
| :--- | :--- |
| `encode(text)` | `{"pieces": [...], "ids": [...], "text": normalized}` |
| `encode_ids(text)` / `encode_batch(texts)` | `list[int]` / `list[list[int]]` |
| `decode(pieces)` / `decode_ids(ids)` | `str` |
| `normalize(text)` | the artifact's own normalizer |
| `get_vocab_size()` / `get_vocab()` | `int` / `dict[str, int]` |
| `id_to_piece(id)` / `piece_to_id(piece)` | `str` / `int` |
| `unk_id` `bos_id` `eos_id` `pad_id` | `0` `1` `2` `3` |
| `model_card()` | `artifact_version`, corpus digest, config, measured metrics |

The artifact is cached by path, so constructing many instances is cheap.

## CLI

```bash
pip install 'mon-tokenizer[cli]'

mon-tokenizer "ဂွံအခေါင်အရာမွဲ"          # tokenize
mon-tokenizer -v "ဂွံအခေါင်အရာမွဲ"       # per-token table
mon-tokenizer -d --ids "316,12644,294"   # decode ids
```

Exit codes: `0` ok · `1` usage · `2` artifact failed to load · `130` interrupted.

## Upgrading from 0.2.x

**Every token id has changed.** Rebuild any embedding matrix built against 0.2.x,
or pin `mon-tokenizer<1.0` — it stays on PyPI and works.

## Requirements

Python 3.11+. Two direct dependencies: `tokenizers` and `regex`. The install is
17 packages / ~30MB, because `tokenizers` requires `huggingface-hub`.

## Development

```bash
uv sync --all-extras
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest
```

Retraining, releasing and the house rules for tests:
[docs/how_to_contribute.md](docs/how_to_contribute.md).

## Design

[docs/architecture.md](docs/architecture.md) — every decision with the measurement
behind it and the condition that would reverse it: algorithm, vocabulary size,
why grapheme clusters are the wrong unit for Myanmar, and what was rejected.

## Links

- [Hugging Face](https://huggingface.co/janakhpon/mon_tokenizer)
- [MonCorpusCollection](https://github.com/MonDevHub/MonCorpusCollection) — the corpus
- [Awesome Mon NLP](https://github.com/janakhpon/awesome-mon-nlp) — the ecosystem

MIT.
