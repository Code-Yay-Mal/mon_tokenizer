# Architecture

`mon-lm/src/train-tokenizer.py:15` has pointed at this file since April. It did
not exist. Here it is.

Each section is a decision, what it was measured against, and what would reverse
it.

---

## 1. Unigram, not BPE

**Measured**, at 48k on the full val split, same harness:

| | Mon | Burmese | English | mixed | syllable violations | train |
|---|---|---|---|---|---|---|
| **unigram** | 4.620 | 3.842 | 4.088 | 3.693 | **1.24%** | 138s |
| bpe | **4.932** | **4.047** | **5.136** | **4.156** | 3.08% | 15s |
| byte-level bpe | 1.516 | 1.545 | 4.964 | 2.150 | — | 7s |
| wordpiece | 4.937 | 3.981 | 5.131 | 4.128 | — | 15s |

**BPE compresses better in every bucket** — +6.8% Mon, +25.6% English — and trains
nine times faster. Unigram wins on one thing: it splits Myanmar syllables 2.5×
less often, over a denominator of 489,745 syllables.

Unigram is chosen because the downstream consumer is OCR and a VLM reading scanned
pages, where the syllable is the unit a reader sees. That is a judgement about the
use case, not a claim that Unigram is better in general — the compression cost is
real and documented.

**Byte-level BPE is out on evidence**: Myanmar is three UTF-8 bytes per character,
so it starts three times behind and never recovers. 1.516 chars/token against
4.620, and only 44.5% of characters single-token.

**WordPiece and byte-level BPE show "—" for violations, not 0%.** Their decoders
were misconfigured in that run, so every line was unreconstructable and the
denominator was zero. A zero denominator is missing data, not a perfect score.

*Reverse this if:* compression becomes the binding constraint, or a measurement
shows syllable violations do not affect downstream accuracy. Nobody has measured
that second thing — the claim that atomicity matters is inherited, not tested.

---

## 2. Vocabulary 64,000

**Measured**, unigram, full val split:

| vocab | Mon | Burmese | violations | per-1000-slot gain |
|---|---|---|---|---|
| 32k | 4.345 | 3.512 | 1.73% | — |
| 48k | 4.620 | 3.842 | 1.24% | 0.0172 |
| **64k** | **4.805** | **4.063** | **1.09%** | 0.0116 |
| 96k | 5.053 | 4.336 | 1.07% | 0.0078 |

Compression never plateaus, but its efficiency per slot halves across the range.
Violations *do* plateau — flat after 64k. 64k is that knee.

96k was rejected because the gain is small and the tail thins: at ~19M training
tokens, the rarest of 96,000 entries are seen a handful of times each, and those
become poorly-estimated rows in any downstream embedding matrix.

---

## 3. HuggingFace `tokenizers`, not SentencePiece

Three concrete reasons, in order of weight:

1. **The normalizer is serialized inside `tokenizer.json`.** v1's `encode()`
   applied none, while its model was trained on normalized text with
   `normalization_rule_name="identity"` — so SentencePiece never did it either.
   One ZERO WIDTH SPACE cost five tokens instead of one. In v2 that failure is not
   available: the normalizer travels with the model.
2. **No C++ load validation to trip on.** v1's model raises
   `RuntimeError: INTERNAL: piece is too long` on `sentencepiece` 0.2.2, which
   validates piece length on load. The declared range was `>=0.1.99`, so a fresh
   install could resolve to a version unable to load the model beside it.
3. **`transformers` reads this format natively**, which downstream vocabulary
   expansion needs anyway.

The runtime dependency is a single Rust wheel with no Python dependencies.

---

## 4. Byte fallback, and how it actually works

The corpus spans **1,458 distinct characters across 24 Unicode blocks** — 23,650
Thai characters from dictionary translations, 64,866 typographic quotes and
dashes, 621 emoji, plus Greek, IPA, Devanagari, Cyrillic, Arabic, Hebrew and CJK.
A scanned book will contain more.

Without byte fallback those are **not marked unknown — they are deleted**:

```
'ကျော် page 42 — “quoted” ၏ 🙏 ภาษา ə café ×2 …'
    decodes to
'ကော် e   oed       '
```

**Enabling it is not one flag.** `UnigramTrainer` has no byte-fallback parameter,
and `models.Unigram(vocab, byte_fallback=True)` still emits `<unk>` because the
256 `<0xNN>` pieces are not in the vocabulary — SentencePiece's trainer adds them,
HuggingFace's does not. So `trainer.py` injects them explicitly after training and
attaches a decoder that reassembles them:

```python
decoders.Sequence([ByteFallback(), Fuse(), Replace("▁", " "), Strip(" ", 1, 0)])
```

Order was determined by testing. `ByteFallback` must run before `Fuse` so byte
runs become characters first.

`initial_alphabet` separately guarantees that printable ASCII, all three Myanmar
ranges and common book punctuation are each a *single* token — byte fallback makes
everything representable, this makes the common cases cheap.

---

## 5. Myanmar syllables, not grapheme clusters

`syllable.py` implements:

```
syllable := base (U+1039 base)* mark* (base U+103A mark*)*
```

The last clause matters: **a consonant carrying U+103A ASAT is a syllable final,
not the start of the next one.** Without it `ဒုင်စသိုင်` segments as five instead
of three.

`\X` (Unicode extended grapheme cluster) is wrong here. UAX #29 gives `GCB=Other`
to `ါ`, `ာ` and `း`, so `\X` returns `['ကျေ', 'ာ်']` for `ကျော်` — and a metric
built on it scores a cut through the middle of that syllable as clean.

Character classes are derived from `unicodedata.category`, not hand-typed. Hand
ranges for this script are how `ၞ ၟ ၠ` (U+105E–1060) came to be described as
letters when they are medial *signs* — `ၝၞၟၠ` is one syllable, not four.

The fixture in `SYLLABLE_FIXTURE` is validated by `tests/test_syllable.py`. Three
of its expectations were wrong when written and the segmenter was right each time,
which is why it is asserted rather than trusted.

---

## 6. Rejected: the syllable-atomicity filter

Dropping vocabulary pieces that begin with a combining mark should force the
segmentation to respect syllable edges. **Measured at 64k:**

**Measured on the pre-cleanup corpus** (before URL and percent-encoded lines were
filtered), so these figures are internally comparable but do not match §7's:

| | violations | chars/token | character coverage | byte fallback | unreconstructable |
|---|---|---|---|---|---|
| off | 1.10% | 4.798 | **100.0%** | **0.00%** | 4 |
| on | 0.51% | 4.734 | 88.4% | 1.34% | 1,337 |

It halves violations for only −1.3% compression, and would have passed the
written threshold. Rejected anyway, for three reasons the threshold missed:

1. **Coverage falls to 88.4%** — 11.6% of Mon characters stop being single-token.
2. **The improvement is not like-for-like.** Unreconstructable lines went 4 →
   1,337, and those are excluded from the denominator (492,469 → 427,690). The
   excluded lines are exactly the ones now falling back to bytes.
3. 1.10% is already low, at full coverage and zero fallback.

*Reverse this if:* a downstream measurement shows syllable violations cost real
accuracy, and the coverage loss can be bought back another way.

---

## 7. Training data

**A note on which corpus a number came from.** The tables in §1, §2 and §6 were
measured before URL and percent-encoded lines were filtered out; §8 and the
shipped model card are after. They are internally consistent within each section
and must not be read across sections — that is exactly the mistake that made the
0.2.4 comparisons invalid.

mon_OCR's bucketed `data/raw/corpus/`, exported by `scripts/export_corpus.py`:
**893,936 unique lines / 85.8M characters**, digest
`11941a573a5e0c618edbc91d34dd787d`.

MonCorpusCollection is the upstream source of truth for Mon text, and 95.7% of its
lines are already here; mon_OCR adds the language bucketing, the Burmese and
English that MCC does not carry, and the stable-hash split.

Two rules:

- **Train split only.** v1 was fitted on the whole corpus including text it was
  later scored on. The measured effect was small (0.6%), but a number measured on
  held-out text is worth more than a slightly larger one measured on the fit.
- **Bucket priority `mon > burmese > english`.** A line in two language
  directories used to be assigned by alphabetical read order, which silently moved
  15,544 lines out of Mon and made 30.5% of the Burmese eval stratum Mon-corpus
  text. Priority makes it a decision instead of an accident, and the count is
  reported.

v1's `min_mon_ratio=0.3` filter is gone. It dropped lines less than 30% Myanmar,
so Burmese- and English-dominant text never entered the fit.

---

## What is not decided here

**Whether syllable violations matter downstream.** Everything above treats them as
worth optimising, on the inherited assumption that a boundary inside a visual unit
is bad. That has never been measured against OCR or LM accuracy. If it turns out
not to matter, BPE at 64k is the better tokenizer and §1 and §6 both change.


---

## 8. Rejected from the corpus: URL and percent-encoded remains

`load_corpus` already dropped lines containing URLs. It did not drop the *encoded
remains* of one — `%E1%80%9E` is not a URL, and every character in it is a
legitimate charset member, so it passed straight through the charset filter into
rendered training images.

`export_corpus.py` was worse: it read the `.txt` files directly and applied
**neither** filter, so the tokenizer was fitted on text the OCR model never saw.
Two paths, two distributions, silently.

**Measured cost, before the fix:**

| | |
|---|---|
| corpus lines carrying a run of percent-escapes | 9,009 (0.997%) |
| share of all characters | 2.27% |
| vocabulary pieces that were percent-encoded fragments | **1,483 of 64,256 (2.3%)** |

After filtering: **0 such pieces**, and 9,496 lines dropped.

**Mon compression reads 4.798 → 4.686, and that is not a regression.** The val
split changed too (30,060 → 29,600 lines), so it is not a like-for-like number.
The old figure was *inflated by the junk it is now free of*: a piece like
`%E1%80%86%E1%80%` covers sixteen characters in one token, so lines full of them
scored extremely well. 4.686 is what the tokenizer does on clean Mon.

Syllable violations improved on the same change: Mon 1.10% → 1.07%, Burmese
0.98% → 0.93%, mixed 0.87% → 0.81%.

The rule of two-or-more escapes is deliberate: one is prose (`50% off`, `50% 2x`),
a run is machine output. Both directions are asserted in
`mon_OCR/tests/test_corpus.py`.

---

## 9. Two version numbers, deliberately independent

`model_card.json` carries `artifact_version`. It is **not** the package version,
and the two must not be coupled.

| | names | changes when | lives in |
| :--- | :--- | :--- | :--- |
| `__version__` | the Python package | any release, including a docs or bugfix release | `pyproject.toml`, read via `importlib.metadata` |
| `artifact_version` | the trained tokenizer | the model changes — new corpus, new vocabulary, new algorithm | `model_card.json`, both here and on the Hub |

They read `1.0.0` and `1.0.0` today because the package and the artifact shipped
together. That coincidence is the hazard: a bugfix release ships a byte-identical
tokenizer, so coupling them would force regenerating and republishing the Hugging
Face artifact to change one string — churn that invites skipping the republish,
which is exactly how the two fall out of sync.

The field was previously called `version`, sat next to `__version__`, and was
hardcoded in `scripts/train_tokenizer.py` with nothing comparing it to anything.
That is the same shape as the defect this release fixed, where `__version__`
stayed `"0.1.5"` across four releases.

Two things hold it now:

- `tests/test_model_card.py` pins every artifact version to the corpus digest it
  was trained on (`ARTIFACT_LINEAGE`), so a retrain that reuses a version goes
  red, and a bare `version` key coming back goes red.
- `scripts/preflight.py` refuses to publish if the Hub's card and the package's
  card disagree on either field.

**What would reverse this:** if the package ever ships *only* to carry an
artifact — no library code of its own — the two numbers would describe the same
thing and keeping them apart would be ceremony.

---

## 10. What survives grafting onto a foreign BPE model

§1 compares the algorithms **standalone**. It says nothing about the other way
this tokenizer gets used: mon-vlm injects its Mon pieces into a base VLM whose
own tokenizer is byte-level BPE. "Most modern models use BPE — should we be on
Unigram?" is a fair question, and standalone numbers do not answer it.

**Measured**, 2,000 held-out Mon lines, pieces injected into Qwen2.5-VL-3B:

| Mon pieces sourced from | Mon tok/char on Qwen | round-trip |
|---|---:|---:|
| *(nothing injected)* | 1.084 | 100% |
| **unigram** — the shipped artifact | **0.392** | 100% |
| bpe — same corpus, same 64k vocabulary | **0.366** | 100% |

Inventory overlap between the two: 2,575 of 4,000 pieces.

**BPE-derived pieces graft 6.6% better**, which tracks §1's standalone +6.8% Mon
almost exactly. The compression edge survives the graft; it is not an artefact of
running Unigram inference.

### The part that matters more than the 6.6%

**Only the piece inventory transfers. The algorithm does not.** HuggingFace
registers added tokens in a trie that pre-splits text *before* BPE runs, so
Unigram's Viterbi inference is discarded entirely on graft. At inference inside a
VLM you are running the base model's BPE plus a longest-match trie, whichever
source produced the pieces.

That has a sharp consequence, learned the expensive way: because the trie
pre-splits, **injecting pieces that are not Myanmar actively damages the base
model's own tokenization.** A first pass injected the top-4000 pieces by id — 54.7%
of them ASCII, because this tokenizer is trilingual — and English went from 0.21
to 0.66 tok/char, a 3.1× regression that looked like a real cost of expansion and
was entirely this bug. Filtering to Myanmar script fixed it and improved Mon from
0.52 to 0.38. See `mon-vlm/docs/adr/0002-base-model.md`.

### So why keep Unigram

Because the graft is not the only consumer. mon_OCR and any standalone use run
*this* tokenizer's own inference, and that is where §1's real difference lives:
**1.24% syllable violations against BPE's 3.08%**, 2.5× fewer over 489,745
syllables.

Switching to BPE would trade a 2.5× regression in syllable integrity — the
property this tokenizer was chosen for — for a 6.6% compression gain, on an
assumption nobody has tested.

*Reverse this if:* the mon-vlm evaluation runs both inventories and finds CER
indistinguishable. Then syllable integrity has been shown not to matter for the
downstream task, BPE wins on compression, and §1's reversal condition is met with
evidence rather than ecosystem convention. **That measurement does not exist yet**,
and until it does, "most modern models use BPE" is an observation about tooling
inertia, not about Mon.
