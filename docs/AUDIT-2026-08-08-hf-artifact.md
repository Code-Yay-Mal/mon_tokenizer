# Audit — 2026-08-08 — the Hub artifact repo

Cross-repo engineering audit of the Mon toolchain, recorded per repo. This one covers
`hf_mon_tokenizer`, the HuggingFace Hub repository that publishes the artifact this package ships.
See [AUDIT-2026-08-08.md](AUDIT-2026-08-08.md) for the package itself.

It is recorded here rather than in the Hub repo. That repo is a published artifact surface: what
lands there is the model card, the tokenizer, and its config. An internal engineering document
carrying severities and ratings is not an artifact, and a consumer loading the model should not have
to read past one.

Audited state: **working tree**, not `HEAD`. 7 modified, 3 deleted, 1 untracked — an uncommitted
SentencePiece to `tokenizers`-JSON migration.

This is a HuggingFace Hub artifact repository, not a code repository. It is judged as one: the
product is the model card and the artifact, and most engineering dimensions do not apply.

---

## Retracted

### M1 — "The 4.6 MB artifact is not covered by Git LFS" — **withdrawn 2026-08-08**

This was reported as Medium and is wrong. Acting on it would have fought the platform convention for
no benefit. Retained here rather than deleted, because a retracted finding is more useful than a
silently disappeared one.

Why it does not stand:

- **4.6 MB is under the Hub's 10 MB threshold.** LFS is required above that, not at this size.
- **HF's own default `.gitattributes` template does not LFS-track `*.json`.** This repo's patterns
  (`*.model`, `*.bin`, `*.safetensors`, `*.vocab`) are exactly that template. Large published models
  ship `tokenizer.json` as a plain blob.
- **LFS would make things worse for consumers.** A plain `git clone` without git-lfs installed yields
  a pointer file instead of a tokenizer. `from_pretrained()` resolves either way, so the change would
  add a failure mode and remove none.
- **The cost is smaller than stated.** `mon_tokenizer/docs/how_to_contribute.md` makes a retrain a
  major version bump, so this is a few MB on an occasional release, not per commit.

The original reasoning confused "large file" with "file that must be LFS-tracked". The threshold, and
the platform's own convention, are what decide it.

<details>
<summary>Original finding as written</summary>

### The 4.6 MB artifact is not covered by Git LFS

**Location:** `.gitattributes`

```
*.model filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.vocab filter=lfs diff=lfs merge=lfs -text
```

Verified:

```
tokenizer.json   4744KB   git check-attr filter -> NO lfs
```

**Reasoning.** The LFS patterns were written for the SentencePiece era, when the artifact was
`.model` and `.vocab`. The uncommitted migration replaces those with `tokenizer.json`, which no
pattern matches, so the new artifact is stored as a plain git blob.

**Impact.** 4.6 MB is under the Hub's hard limit and works today. The cost is permanent and
compounding: every future retrain adds another full copy to history, and history cannot be pruned
without a force-push that breaks every clone.

**Fix.** Add `*.json filter=lfs` scoped to the artifact, or `tokenizer.json` explicitly, before the
migration is committed. Doing it after means the plain blob is already in history.

</details>

---

## Low

### L1 — No `LICENSE` file

There is no `LICENSE` at the repository root. The licence *is* declared — `README.md:2` carries
`license: mit` in the model card frontmatter, and there is a `## License` section at `:148`.

The Hub reads the frontmatter, so this is cosmetic rather than a licensing gap. Recorded because an
initial read of this repo flagged it as a High finding on the assumption that no licence was declared
anywhere; that was wrong, and the correction is worth keeping.

---

## What is good, specifically

- **The `.gitignore` inverts the usual rule deliberately and correctly.** It ignores
  `pyproject.toml`, `uv.lock`, `.python-version`, `convert_to_hf.py`, `upload_to_hub.py` and
  `test_tokenizer.py` so that only artifacts reach the Hub. That is the right call for a model repo
  and it is not the obvious one.
- **The model card frontmatter is complete and valid**: `license`, `language` (`mnw`, `my`, `en`),
  `library_name`, and meaningful tags. A card that the Hub can index properly.
- **`model_card.json` carries `artifact_version` and `vocab_size`** as machine-readable facts rather
  than only prose, so the parity check in `mon_tokenizer`'s preflight can assert against them.

---

## Ratings

Most dimensions do not apply to an artifact repository. Marked `n/a` rather than given an invented
number.

| Dimension | Score | Why |
| :--- | ---: | :--- |
| Correctness | 7 | Card frontmatter valid; artifact and declared vocab size agree. |
| Reliability | 7 | Works as intended; no defect found once M1 was withdrawn. |
| Maintainability | 6 | Nine files, flat layout, clear purpose. |
| Security | n/a | No code, no execution surface. |
| Architecture | n/a | Not a meaningful axis for an artifact repo. |
| Performance | n/a | No runtime. |
| Testing | n/a | Tested from `mon_tokenizer`'s preflight, which is the right place. |
| Observability | n/a | Nothing to observe. |
| Readability | 7 | The card explains what the tokenizer is and how to load it. |
| Developer experience | 6 | Loading instructions are present and correct. |
| Production readiness | 6 | Published and usable; the migration is uncommitted. |
| Technical debt | 7 | Little to carry. The `.gitattributes` patterns match HF's own template. |

---

## Roadmap

**Immediate.** Commit the migration alongside `mon_tokenizer`'s 1.0.0 release so the package and the Hub
artifact stay in step. They are a matched pair: landing one without the other breaks the parity
check that `mon_tokenizer`'s preflight runs against this repo.

> Closed 2026-08-08. The Hub repo is at `954bd9d`, PyPI at 1.0.0, and the artifact is byte-identical
> across the package, the Hub, and the wheel on PyPI — `sha256 34d18153…`, verified against a clean
> install rather than against the source tree.

**Next.** Nothing outstanding.

## Deliberately not recommended

- **No CI, no tests, no packaging.** The artifact is validated by `mon_tokenizer`'s preflight against
  a clean wheel install, which is where that check belongs. Duplicating it here would add a second
  thing to keep in sync.
- **No `docs/` directory.** Nine files do not need one, and this document is the reason there is no
  pressure to create one: it lives with the code it audits.
