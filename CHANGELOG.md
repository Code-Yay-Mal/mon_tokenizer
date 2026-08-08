# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-08

**Breaking: every token id has changed.** Rebuild any embedding matrix built
against 0.2.x, or pin `mon-tokenizer<1.0` — it stays on PyPI and works.

The artifact is now `tokenizer.json`. The SentencePiece `.model` is gone, and so
is the `sentencepiece` dependency.

### Measured

Vocabulary 64,256. Trained on the train split of 893,936 lines / 85.8M
characters, scored on the whole validation split. Corpus digest
`11941a573a5e0c618edbc91d34dd787d`.

| stratum | chars/token | syllable violations | round-trip | byte fallback |
|---|---:|---:|---:|---:|
| Mon | 4.686 | 1.07% (n=492,469) | 100% | 0.00% |
| Burmese | 4.117 | 0.93% (n=25,546) | 100% | 0.00% |
| English | 4.112 | — (n=0) | 100% | 0.02% |
| mixed script | 3.804 | 0.81% (n=28,133) | 100% | 0.19% |

100% of characters in the Mon validation split are single tokens.

**Not comparable to any number published before 1.0.0.** Earlier figures were
measured on a different eval sample; see "Corrections".

### Fixed

- **Normalization is applied, and lives inside the artifact** so it cannot drift
  from the model. 0.2.x applied none at encode time while having been *trained*
  on normalized text, so one ZERO WIDTH SPACE cost five tokens instead of one.
- **Full byte fallback.** Thai, emoji, IPA, CJK and typographic punctuation now
  round-trip. Previously they were not flagged as unknown — they were **silently
  deleted**. Enabling it needed the 256 `<0xNN>` pieces injected explicitly;
  `UnigramTrainer` has no byte-fallback option and never emits them.
- **`<pad>` has a real id (3).** It was `-1`, so padding a batch padded with
  `<unk>`.
- **Runs of spaces are preserved.** SentencePiece collapsed them, which was the
  entire cause of 0.2.x's round-trip loss.
- **URL and percent-encoded text removed from training.** 1,483 of 64,256 pieces
  (2.3%) were fragments like `%E1%80%86%E1%80%`. The corpus export and mon_OCR's
  loader had also diverged — the tokenizer was fitted on text the OCR model never
  sees.

### Changed

- **Unigram kept, now on evidence rather than assertion.** BPE was measured and
  compresses better in every bucket (+6.8% Mon, +25.6% English, 9x faster) but
  splits Myanmar syllables 2.5x more often. Byte-level BPE was rejected on
  measurement: Myanmar is three UTF-8 bytes per character, giving 1.516
  chars/token against 4.620.
- **Vocabulary 32,000 → 64,000** — the measured knee, where violations flatten.
- **`min_mon_ratio=0.3` removed from training.** It excluded Burmese- and
  English-dominant lines from the fit.
- Tests 36 → 98. New: `syllable.py`, `trainer.py`, `metrics.py`,
  `model_card.json`, `docs/architecture.md`, `scripts/preflight.py`.

### Corrections

Recorded because the same mistakes are easy to repeat:

- **Eval sets differed.** An earlier sweep absorbed val *and* test. Measured
  like-for-like, 0.2.x's Mon compression is 3.019 not 2.765, and its round-trip
  87.1% not 93.8% — so a claimed +23.6% gain was really +13.2%.
- **Head-of-file sampling was unrepresentative**: `dict/` was 47.5% of the Mon
  sample but 8.1% of the split, and `mon_shards/` contributed nothing. All 1.0.0
  numbers use the whole split.
- **The violation metric measured the wrong unit** — Unicode grapheme clusters
  break before `ာ`, so a cut through `ကျော်` scored clean.
- **Byte-fallback rate was structurally zero**: it matched piece strings, but
  HuggingFace Unigram returns the original surface substring for an unknown span.
- **A `Split(\X)` pre-tokenizer for syllable atomicity gives 1.144 chars/token**
  — 4.7x worse. A pre-tokenizer bounds how far a token may span; it cannot forbid
  a split inside one. Do not retry it.

## [0.2.4] — never released, folded into 1.0.0

Prepared and tagged nothing. The work below was completed against the 0.2.x
SentencePiece artifact and then superseded by the 1.0.0 retrain before it
shipped, so **there is no 0.2.4 on PyPI** — `pip install mon-tokenizer==0.2.4`
will fail. It is kept here because the fixes are real and 1.0.0 carries all of
them; the packaging and CI work in particular is independent of the retrain.

The model is unchanged. Everything here is the library and the packaging around
it, where an audit found three defects that had already shipped to PyPI.

### Fixed

- **`encode()` now applies the training-time normalization.** The model was
  trained on text with invisible characters stripped and NFC applied, and with
  `normalization_rule_name="identity"` so SentencePiece never normalizes on its
  own. `encode()` did neither, which made every call a train/inference skew.
  Measured: `"ဂွံ‌အခေါင်"` with one ZERO WIDTH SPACE tokenized as
  `['▁ဂွံ', '<0xE2>', '<0x80>', '<0x8B>', 'အခေါင်']` — **five tokens, three of
  them byte fallback, where the normalized form is one**. Zero-width joiners
  arrive routinely from web pages and PDF extraction. Pass `normalize=False` for
  the old behaviour; it is a debugging aid, not a supported production mode.
- **`twine` is no longer a runtime dependency.** It is a package *publishing*
  tool and was declared unconditionally, so every install pulled keyring,
  cryptography, requests, docutils, readme-renderer and the `jaraco.*` tree.
  `click` and `rich` moved to a new `[cli]` extra. **The install went from 31
  packages / 25MB to 2 packages.**
- **`sentencepiece` is now bounded `>=0.2.0,<0.2.2`.** The bundled model raises
  `RuntimeError: INTERNAL: piece is too long` on 0.2.2, which validates
  `user_defined_symbols` length on load. The previous floor of `>=0.1.99` with no
  ceiling meant a fresh install could resolve to a version unable to load the
  model shipped beside it.
- **`__version__` is read from installed metadata.** It was hardcoded `"0.1.5"`
  and stayed there through 0.2.0, 0.2.1, 0.2.2 and 0.2.3, so
  `mon-tokenizer --version` reported 0.1.5 on all of them.
- **Build configuration now uses a table the backend reads.** `[tool.uv-build]`
  is not one; the correct name is `[tool.uv.build-backend]`. The `setuptools`
  tables and `MANIFEST.in` were also inert under `uv_build`. The 1MB model was
  shipping only because the backend includes the module directory by default.
- **One dev-dependency declaration.** There were two, and `uv sync --dev` — the
  documented command — resolved the one *without* ruff, mypy, black or isort, so
  none of the declared tooling was ever installed.
- `--version` no longer reports the wrong number; `--ids` decodes ids
  unambiguously (`--tokens` cannot express a piece containing a comma); Ctrl-C in
  interactive mode exits cleanly instead of printing a traceback; the model is no
  longer loaded before argument validation.

### Added

- **CI.** The repository had one workflow, `release.yml`, which published to PyPI
  on any `v*` tag with no test run, no lint and no Python matrix — the mechanism
  by which the defects above shipped. There is now a `ci` workflow (ruff, mypy,
  pytest on 3.11/3.12/3.13, plus a job that builds a wheel, asserts the model is
  inside it, and installs it into a clean environment), and `release.yml` waits
  on it. The release job also refuses to publish when the tag disagrees with the
  version in `pyproject.toml`.
- **Special-token accessors** — `unk_id`, `bos_id`, `eos_id`, `pad_id`. Worth
  knowing: **`pad_id` is `-1`**. The trainer passed `pad_piece="<pad>"` but never
  assigned it an id, so `<pad>` resolves to `<unk>` and padding a batch with it
  pads with unknown tokens. Fixing that needs a retrain; for now it is at least
  visible.
- `encode_ids()` for callers that never look at the piece strings.
- `normalize_text()` and `default_model_path()` are exported.

### Changed

- **The model is cached by path.** It is ~1MB and was re-read on every
  `MonTokenizer()` construction.
- **`encode()` segments once, not twice.** It called `EncodeAsPieces` *and*
  `EncodeAsIds`, running the full Viterbi decode twice per call to produce
  strings `IdToPiece` already maps. Equivalence is asserted in the tests.
- **`encode()["text"]` is the normalized string that was actually encoded**, not
  the raw input. A caller comparing a decode against it was previously comparing
  against something that was never encoded.
- `model_path` is always a `Path`. It was annotated `Optional[str]` while holding
  a `Path` on the default branch, and `py.typed` ships — so every consumer's type
  checker was told something untrue.
- Model loading is located with `importlib.resources` rather than `__file__`.
- The test suite went from 6 tests to 36. The previous suite could not detect the
  4,000 → 32,000 vocabulary change: `test_vocab_size` asserted
  `0 < size < 100000`, and roughly half the others were tautological. Vocabulary
  size, exact pieces and exact ids for a fixed input, and per-language
  compression floors are now pinned.
- The `docs` extra is removed. It installed sphinx and sphinx-rtd-theme against
  no sphinx project.

### Measured

Against mon_OCR's corpus (913,044 unique lines / 83.7M characters), compared to a
277-character one-token-per-grapheme Mon OCR charset:

| bucket | chars/token | byte fallback | round-trip | vs charset |
|---|---|---|---|---|
| Mon | 3.73 | 3.14% | 99.1% | 2.25× |
| Burmese | 4.60 | 0.04% | 100.0% | 2.87× |
| English | 2.88 | 0.14% | 99.9% | 2.88× |

Two notes on numbers this file previously carried:

- **The 5.17× / 5.22× compression figure does not reproduce on this corpus** —
  3.63–3.73 chars/token measured. The difference is corpus composition: the Mon
  bucket here includes dictionary headwords and proper nouns, which compress
  poorly. The figure is a property of a corpus, not of the tokenizer.
- **"100% round-trip" is correct for the script and misleading as stated.** 4.42%
  of corpus lines do not survive `decode(encode(x))` — and every one of those
  failures is SentencePiece collapsing runs of `U+0020`. Zero Mon characters are
  lost or gained. The whitespace behaviour is fixed in the v2 retrain.

## [0.2.0] - 2026-04-26

### Added
- Major model upgrade: Vocabulary size expanded from 4,000 to 32,000.
- 5.17x compression ratio (2.6x improvement over 0.1.x).
- Grapheme cluster atomicity via Unicode extended grapheme clusters.
- Trained on full 177M character Mon corpus.
- 100% round-trip accuracy guaranteed for Mon script.

## [0.1.0] - 2025-08-23

### Added
- Initial release of Mon tokenizer
- Core tokenization functionality with SentencePiece
- CLI interface with encode/decode capabilities
- Python API with MonTokenizer class
- Support for custom model paths
- Rich CLI output with verbose mode
