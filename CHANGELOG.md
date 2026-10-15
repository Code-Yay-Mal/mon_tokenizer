# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases not listed here (0.1.1–0.1.5, 0.2.1): see the PyPI release history,
<https://pypi.org/project/mon-tokenizer/#history>.

## [1.0.2] - 2026-10-15

**The artifact is unchanged.** `mon_tokenizer.json` still hashes to
`34d181532eee7e6754bfbad693753ee4660bdda00eb36a2ccc7cd2f372c71282`. What changed
is the licence the package declared over it.

### Corrected

- **The distribution no longer declares `license = "MIT"`.** That expression
  claimed one permissive grant over the whole wheel, including
  `mon_tokenizer.json` — the same file the Hugging Face repository's `LICENSE`
  says it grants no rights over, because its training data includes CC BY-SA 4.0
  text and sources with no established licence. `LICENSE` now scopes the MIT
  grant to everything except that file, `LICENSE-ARTIFACT.md` states the
  artifact's terms, and `license-files` ships both in the wheel and the sdist.
  This follows mon-language-detector 0.2.3.
- **README links to `docs/` are absolute**, so they resolve on PyPI as well as
  GitHub.
- **The dependency comment in `pyproject.toml` says two**, matching the two it
  sits above.

## [1.0.1] - 2026-09-03

**The artifact is unchanged.** `mon_tokenizer.json` still hashes to
`34d181532eee7e6754bfbad693753ee4660bdda00eb36a2ccc7cd2f372c71282`, every token
id is the same, and nothing was retrained. What changed is a number that
described it wrongly.

### Changed

- **The README carries one badge instead of seven.** PyPI already prints the
  version, the Python requirement and the licence in its own header and sidebar,
  so those badges repeated the page they sat on; the ruff and uv badges described
  this repository's tooling rather than anything a caller needs. The licence badge
  was also broken on the card, linking to a relative `LICENSE` path that resolves
  on GitHub and nowhere on pypi.org. The Hugging Face URL is unchanged and still
  listed under Links.

### Corrected

- **Single-token coverage on the Mon val split is 98.74%, not 100%.** 1.0.0's
  driver computed coverage over `val["mon"][:5000]` while the card it wrote
  hardcoded `"sampling": "whole split, no cap"` — so the figure was measured on
  17% of the split and published as though it covered all of it. Over all 29,600
  lines the split holds 397 distinct characters, of which 392 are single tokens.
  The five that are not — U+0324, U+03A0, U+0DBA, U+0DD2 and U+1F931, seven
  occurrences in 2,280,192 characters — round-trip through byte fallback.

  The 5,000-line slice really does score 100% over 199 distinct characters, which
  is why nothing looked wrong: the head of the file is Mon dictionary text, and
  the rarer scripts arrive later in the split.

  Every other figure on the card re-measured **identically** against the shipped
  artifact, so this correction is scoped to coverage alone.

### Added

- **`scripts/train_tokenizer.py --recard`** re-measures the shipped artifact and
  rewrites only its card. Correcting a card used to require a retrain, which
  changes every token id to fix a number that describes the existing ids
  correctly. It refuses if the corpus on disk is not the one the card names.
- **`tests/test_artifact_sync.py`** pins the artifact's sha256 and asserts the
  Hugging Face copy is byte-identical when that checkout is present; CI now
  fetches the Hub copy and compares. The identity of the two copies was checked
  by hand, and the one time it was not, users got a 4,000-piece tokenizer against
  a card advertising 32,000.

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

98.74% of the distinct characters in the Mon validation split are single tokens —
392 of 397, over all 29,600 lines. The other five round-trip through byte
fallback.

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
  compresses better in every bucket (+6.2% Mon, +25.8% English, 10x faster) but
  splits Myanmar syllables 2.5x more often. Byte-level BPE was rejected on
  measurement: Myanmar is three UTF-8 bytes per character, giving 1.524
  chars/token against 4.507, both at 48,000 pieces over the 29,600-line Mon
  validation split.
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
  sample but 8.1% of the split, and `mon_shards/` contributed nothing. Every
  1.0.0 compression, round-trip, violation and fallback figure uses the whole
  split. **Coverage did not** — that one shipped measured over the first 5,000
  Mon lines; see Unreleased.
- **The violation metric measured the wrong unit** — Unicode grapheme clusters
  break before `ာ`, so a cut through `ကျော်` scored clean.
- **Byte-fallback rate was structurally zero**: it matched piece strings, but
  HuggingFace Unigram returns the original surface substring for an unknown span.
- **A `Split(\X)` pre-tokenizer for syllable atomicity gives 1.144 chars/token**
  — 4.7x worse. A pre-tokenizer bounds how far a token may span; it cannot forbid
  a split inside one. Do not retry it.

## [0.2.4] — never released, folded into 1.0.0

Library and packaging fixes made against the 0.2.x SentencePiece artifact, then
superseded by the 1.0.0 retrain before they shipped. There is no 0.2.4 on PyPI;
1.0.0 carries all of it. The full entry is in this file's git history.

## [0.2.3] - 2026-04-27

- Sync the 32k model with the refined 41.4M-character corpus (`v0.2.3` tag).

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
