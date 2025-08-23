# Mon Tokenizer

Tokenize Mon text like a pro. No fancy stuff, just gets the job done.

## Quick Start

```bash
# Using pip
pip install mon-tokenizer

# Using uv (faster)
uv add mon-tokenizer
```

```python
from mon_tokenizer import MonTokenizer

tokenizer = MonTokenizer()
text = "ဂွံအခေါင်အရာမွဲသ္ဂောံဒုင်စသိုင်ကၠာကၠာရ။"

# Tokenize
result = tokenizer.encode(text)
print(result["pieces"])  # ['▁ဂွံ', 'အခေါင်', 'အရာ', 'မွဲ', 'သ္ဂောံ', 'ဒုင်စသိုင်', 'ကၠာ', 'ကၠာ', 'ရ', '။']

# Decode
decoded = tokenizer.decode(result["pieces"])
print(decoded)  # ဂွံအခေါင်အရာမွဲသ္ဂောံဒုင်စသိုင်ကၠာကၠာရ။
```

## CLI

```bash
# Tokenize
mon-tokenizer "ဂွံအခေါင်အရာမွဲသ္ဂောံဒုင်စသိုင်ကၠာကၠာရ။"

# Verbose mode (shows all the details)
mon-tokenizer -v "ဂွံအခေါင်အရာမွဲသ္ဂောံဒုင်စသိုင်ကၠာကၠာရ။"

# Decode tokens back to text
mon-tokenizer -d -t "▁ဂွံ,အခေါင်,အရာ,မွဲ,သ္ဂောံ,ဒုင်စသိုင်,ကၠာ,ကၠာ,ရ,။"
```

## API

- `encode(text)` - Chop text into tokens
- `decode(pieces)` - Glue tokens back together
- `decode_ids(ids)` - Convert IDs back to text
- `get_vocab_size()` - How many tokens we know
- `get_vocab()` - The whole vocabulary

## Dev Setup

```bash
git clone git@github.com:janakhpon/mon_tokenizer.git
cd mon_tokenizer
uv sync --dev
uv run pytest

# Release workflow
uv version --bump patch
git add pyproject.toml
git commit -m "v0.1.1"
git tag v0.1.1
git push origin main --tags
```

## License

MIT - do whatever you want with it.
