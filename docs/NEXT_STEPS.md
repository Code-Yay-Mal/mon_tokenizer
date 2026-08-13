# Next steps

Forward work for this repository. Findings live in
[AUDIT-2026-08-08.md](AUDIT-2026-08-08.md) and
[AUDIT-2026-08-08-hf-artifact.md](AUDIT-2026-08-08-hf-artifact.md).

**Re-verified 2026-08-13** with `uv run pytest`: `141 passed, 2 xfailed in 4.88s`
· version, tag and changelog all agree at `1.0.0` · CI tracked, matrix on
3.11/3.12/3.13, **seven action references pinned to 40-character commit SHAs**
across `ci.yml` and `release.yml` — the eighth `uses:` is `release.yml`'s call
into `./.github/workflows/ci.yml`, a local path with no SHA to pin · H1's
round-trip claim is qualified in the README immediately below the table and its
two limits are xfail-tested.

The line above read `107 passed, 3 xfailed` under a 2026-08-09 date until today.
The suite has grown by 34 tests since and carries two `xfail` markers, both in
`tests/test_tokenizer.py`, matching the two documented round-trip limits. Nothing
regressed. Dates on this line are the day the command was run, not the day the
file was edited.

Every finding in both audits is closed. This file is therefore not a repair list —
it is what a shipping package does next, and the honest answer is *less than you
would think*.

---

## The standing instruction

**This repository is the reference implementation for the ecosystem.** Four other
repos consume it or should. When something here is good, the cheapest win
available anywhere is copying it sideways rather than adding to it.

Two things are worth copying today, and neither costs this repo anything:

| Asset | Who needs it | Note |
|---|---|---|
| `.github/workflows/ci.yml` | mon_OCR, mon-vlm, mon-lm, MonCorpusCollection, mon-language-detector | 1 of 8 repos in the ecosystem has CI. This is that one |
| `syllable.py` and `normalization.py` | mon-lm, MonCorpusCollection, mon-language-detector | Three separate normalisation paths already exist. `tests/test_normalization.py:9` records that they had drifted before v2 |

Neither is work for this repo. Both are recorded here because this is where the
originals live and the drift is a repeated finding elsewhere.

---

## 1. The chars/token disagreement, and which side carries the denominator

`README.md:53` reports **4.686** chars/token on Mon.
`mon_OCR/docs/adr/0014-vlm-line-recognition-pipeline.md` quoted **4.798** for the
same tokenizer, also naming version 1.0.0 and also calling it the val split, and
built an accepted architectural decision on it. **Closed 2026-08-13:** that ADR
now carries 4.686 with the denominator below, and its table gained `lines` and
`chars / tokens` columns.

**This repository's figure is the one that can be checked.** `model_card.json`
carries the full basis, verified 2026-08-09:

```
mon: lines 29,600 · chars 2,280,192 · tokens 486,631
     2,280,192 / 486,631 = 4.686
```

ADR-0014 stated no denominator, so its 4.798 could not be reproduced from what
was written down. That is what the fix changed: the denominator is now in the
table, not the prose. The most likely explanation is that the two "val splits" are
different corpora — `mon_OCR` partitions by a blake2b hash of each line and this
repository has its own split — in which case both numbers are correct and neither
is labelled well enough to say so.

Also verified 2026-08-09: `tokenizer.json` on the Hugging Face repo is
**byte-identical** to the copy inside the wheel (sha256 `34d181532eee7e6754bf…`),
and every metric in the HF card matches `model_card.json`. So the disagreement is
not two artifacts; it is one artifact measured against two corpora.

**The fix has landed on both sides**, so nothing here is outstanding. Verified
2026-08-13: `mon_OCR/docs/adr/0014-vlm-line-recognition-pipeline.md:44` carries
4.686 with `2,280,192 / 486,631` in the row beside it.

What it taught: a figure quoted without its denominator cannot be reconciled,
only re-measured. Correcting 4.798 to 4.686 was the smaller half — the half that
holds is that both tables now carry the denominator, so the next disagreement
resolves itself. The canonical home for a tokenizer measurement is the tokenizer,
and this one ships its denominator machine-readably.

---

## 2. Extract the two reusable modules

`syllable.py` (163 lines) and `normalization.py` (115 lines) are general Myanmar
text utilities that happen to live inside a tokenizer package. `syllable.py`
derives its character classes rather than hardcoding them and documents why
Unicode `\X` grapheme clusters are the wrong boundary for this script. That is
genuinely useful to anyone working with Myanmar text, and today it is reachable
only by depending on a 4.8 MB tokenizer artifact.

**Two options, and the second is better:**

| Option | Cost |
|---|---|
| Split into a `mon-text` package | A second release process, a second changelog, a version-compatibility matrix. For one maintainer that is a real ongoing tax |
| **Keep one package; document these as public API** and cover them in the README | Free. The import path is already `from mon_tokenizer.syllable import ...` |

**Recommendation: the second.** `small-team-audit.md` §5 asks what happens if we
do nothing — the answer is that the modules stay usable and slightly obscure,
which does not justify a second release pipeline. Revisit if a consumer appears
that genuinely cannot take the tokenizer dependency, and note that the Rust port
in `mon-programme.md` Tier 3a is that consumer's likely shape.

---

## 3. Raise the bar where it is highest

These are the items that would matter to a team evaluating this package, in the
order they would notice.

| | Work | Why |
|---|---|---|
| 3.1 | **Coverage measurement in CI**, reported not gated | `.coverage` exists locally and no number is published. A gate invites gaming; a number invites attention |
| 3.2 | **A benchmark against alternatives.** SentencePiece BPE and stock Qwen on the same split | The 4.686 is a fact about this tokenizer. "Better than X for Mon" is the claim a reader actually wants, and it is currently unstated |
| 3.3 | **Dependency scanning** — `pip-audit` or Dependabot | Two runtime dependencies, so the surface is small. That makes it cheap rather than unnecessary |
| 3.4 | **Publish the training script's determinism guarantee.** `corpus_fingerprint` exists; state whether a rebuild from the same corpus is byte-identical, and test it | The strongest claim an artifact can make is that it can be rebuilt. Nothing currently says whether it can |
| 3.5 | **A `SECURITY.md` and a supported-versions row** | The package is on PyPI. This is the one piece of published-package etiquette missing |

**3.4 is the one worth doing first.** Everything else on this list is
conventional; a reproducible artifact is the thing most published tokenizers
cannot claim.

---

## What we are deliberately not doing

| Not doing | Why |
|---|---|
| **Raising the vocabulary above 64,256** | Chosen with measurement. Growing it costs every downstream embedding matrix and buys compression the corpus cannot justify |
| **Switching Unigram to BPE** | `docs/architecture.md` records the reasoning. Unigram suits a syllable-structured script, and the measured round-trip is 100% with 0.00% byte fallback on Mon |
| **Fixing the two known round-trip limits** | A leading space is dropped and literal U+2581 reads as a space. Both are documented, both are xfail-tested, and both are inherent to the SentencePiece-style space marker. Documented beats silently patched |
| **A second package for the text utilities** | §2 |
| **Gating CI on a coverage threshold** | Thresholds get met by testing what is easy. `trainer.py`'s pure functions were worth testing because the byte-fallback guarantee rests on them, not because a number needed moving |

---

## How this file stays true

Every measurement carries the date it was taken and the command that produced it.
When §1 closes, the number moves into the model card and this section goes.

If a future audit finds nothing, say so and leave this file short. A next-steps
document that grows to justify its own existence is how a healthy repository
acquires work it does not need.
