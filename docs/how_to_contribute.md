# Contributing

## Setup

```bash
git clone git@github.com:Code-Yay-Mal/mon_tokenizer.git
cd mon_tokenizer
uv sync --all-extras
```

`--all-extras`, not `--dev`: the latter installs the dev group but skips the
`[cli]` extra, so `mon-tokenizer` the command is missing and its tests are
skipped.

## The gate

Everything must pass before a commit. There is no CI shortcut — this is the same
set CI runs.

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

`check`, not `format`. `ruff format .` rewrites files, so a gate built on it
cannot fail — it fixes the problem and reports success.

## Before any release

```bash
uv run python scripts/preflight.py --hf ../hf_mon_tokenizer
```

**30 checks** against the **built wheel and a clean install**, not the source
tree — 19 of them without `--hf`, and eleven more comparing the Hugging Face
repo's ids, artifact version and corpus digest against the package's. A check
that passes because the repository happens to be on `sys.path` proves nothing
about what a user receives.

This said 25, which is the number of `check(...)` call sites in the file rather
than the number of checks it reports: two of those sites run in a loop, once per
gate command and once per required Hugging Face file. Count the output, not the
source.

It exits non-zero on any failure and prints `DO NOT PUBLISH`.

## Releasing

**Do not run `uv publish` or `twine upload` by hand.** Both appeared in an earlier
version of this document, and publishing outside the gate is how three defects
reached PyPI: `twine` as a runtime dependency, a version string four releases
stale, and a Hugging Face artifact that was the wrong model.

Releases are tag-driven. Pushing a `v*` tag runs the full CI suite on
3.11/3.12/3.13, builds and inspects the wheel, installs it into a clean
environment, and only then publishes. The job refuses if the tag disagrees with
`pyproject.toml`.

```bash
uv version --bump patch          # or minor / major
git commit -am "bump version"
git tag "v$(uv version --short)"
git push origin main --tags
```

Update `CHANGELOG.md` in the same commit, with measured numbers rather than
estimates.

## Retraining the tokenizer

Needs mon_OCR's bucketed corpus. Two steps because the dependency sets are
disjoint — the export needs `monocr` (torch, opencv), the training needs
`tokenizers`.

```bash
cd ../mon_OCR && uv run python ../mon_tokenizer/scripts/export_corpus.py \
    --output ../mon_tokenizer/build/corpus.jsonl

cd ../mon_tokenizer
uv run python scripts/train_tokenizer.py --gate     # compare variants, writes nothing
uv run python scripts/train_tokenizer.py --write --artifact-version 1.1.0
uv run python scripts/compare_algorithms.py         # unigram vs bpe vs alternatives
```

`--artifact-version` has no default and `--write` refuses without it. It names the
**artifact**, not the package — see [architecture.md](architecture.md) §9 — and
reusing the previous one would publish two different tokenizers under one name,
which is the shape of the defect this release fixed on the Hub. Add the new value
to `ARTIFACT_LINEAGE` in `tests/test_model_card.py` with the corpus digest it was
trained on, or the suite goes red.

`--gate` needs no version, because it writes nothing.

A retrain changes every token id, so it is a **major** version bump for the
package too, and needs the Hugging Face artifacts republished in step — including
`model_card.json`, which `preflight.py` now compares against the package's copy.

## Writing tests

Two rules, both learned here:

- **A test must be able to fail.** The suite this replaced asserted
  `0 < vocab_size < 100000`, which passed for the 4,000-piece model and the
  32,000-piece one alike — so an 8× vocabulary change was invisible to it.
- **Prove a new guard red.** Reintroduce the defect, watch the test fail, restore.
  A guard only ever seen green is one nobody has verified.

## House style

`ruff` settings match mon_OCR's, deliberately — two repositories with one
maintainer diverging on lint config is friction with no upside. `RUF001-003` are
off because this repository is Mon text and prose about Mon text, and they fire
on the subject matter itself.
