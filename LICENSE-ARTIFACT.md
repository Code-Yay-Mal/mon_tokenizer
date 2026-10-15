# Artifact licence and attribution

This file governs the trained artifact shipped inside the package:

    src/mon_tokenizer/data/mon_tokenizer.json

The same bytes are published on Hugging Face as
[`janakhpon/mon_tokenizer`](https://huggingface.co/janakhpon/mon_tokenizer)
`tokenizer.json`, and that repository's `LICENSE` says the same as this file.

The code is separate. `LICENSE` is an MIT grant over everything in the
repository except this file.

`mon_tokenizer.json` is a Unigram vocabulary fitted to a corpus this project neither
collected nor owns. A fitted vocabulary is a derived work of the text it was
fitted to, so its terms follow the training data — and that data has no single
set of terms.

## What the training data is

mon_OCR's bucketed `data/raw/corpus/`, 893,936 unique lines / 85.8M characters,
source digest `11941a573a5e0c618edbc91d34dd787d`. That digest is recorded in
`src/mon_tokenizer/data/model_card.json`, so the claim is checkable rather than
asserted.

MonCorpusCollection is the upstream source of truth for the Mon portion — 95.7%
of its lines are in the export. Its terms are stated per source, keyed to the
shard filename prefix:
[LICENSE-CORPUS.md](https://github.com/MonDevHub/MonCorpusCollection/blob/main/LICENSE-CORPUS.md).

Two rows there decide the practical answer for most consumers:

- Mon Wikipedia is the largest source and is CC BY-SA 4.0, which is copyleft.
- The Independent Mon News Agency archive is the second largest and has no
  established licence at all. That row is marked Unresolved upstream.

Several smaller sources are marked Unresolved too, and the Burmese and English
buckets are mon_OCR's own additions whose terms are recorded nowhere. Read
LICENSE-CORPUS.md rather than this summary before relying on any of it.

## What follows

- **Nothing in this repository grants rights over `mon_tokenizer.json`**, because
  no such rights were obtained.
- Whether a fitted Unigram vocabulary is Adapted Material under CC BY-SA 4.0
  §2(a), and therefore whether ShareAlike propagates to `mon_tokenizer.json`, is
  an open question the maintainers have not taken advice on. It is flagged here
  rather than answered.
- The unresolved sources are the harder problem and are not this repository's
  to close. They are tracked upstream in MonCorpusCollection.

The package is on PyPI. From 1.0.2 its wheel and sdist carry this file and
`LICENSE`, and its metadata declares no single licence. Publishing settles none
of the questions above.

## Attribution, if you use it

Attribute **Mon Tokenizer** and the underlying sources of its training data.

## Reporting a problem

If you hold rights in material used to train it and it should not have been,
open an issue at
[Code-Yay-Mal/mon_tokenizer](https://github.com/Code-Yay-Mal/mon_tokenizer)
and the artifact will be withdrawn pending a retrain.
