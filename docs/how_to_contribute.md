# Contributing

## Setup

```bash
git clone git@github.com:Code-Yay-Mal/mon_tokenizer.git
cd mon_tokenizer
uv sync --dev
uv run pytest
```

## Build & Test

```bash
# build
uv build
uv build --no-sources

# test
uv run pytest
```

## Publish

```bash
# bump version
uv version --bump patch
uv version --short

# publish to pypi
uv publish --trusted-publishing

# or use twine
uv run twine upload dist/*
```

## Release

```bash
# create tag
git tag v0.1.2 
git push origin v0.1.2

# delete tag if needed
git tag -d v0.1.2 && git push origin :refs/tags/v0.1.2
```

## Notes

- use `--trusted-publishing` flag for uv publish
- make sure you have `.pypirc` configured
- gh-actions can auto-publish on tag push
