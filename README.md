# Mon Tokenizer

Tokenize Mon text like a pro. No fancy stuff, just gets the job done.

## quick start

```bash
# using pip
pip install mon-tokenizer

# using uv
uv add mon-tokenizer
```

```python
from mon_tokenizer import MonTokenizer

tokenizer = MonTokenizer()
text = "ဂွံအခေါင်အရာမွဲသ္ဂောံဒုင်စသိုင်ကၠာကၠာရ။"

# tokenize
result = tokenizer.encode(text)
print(result["pieces"])  # ['▁ဂွံ', 'အခေါင်', 'အရာ', 'မွဲ', 'သ္ဂောံ', 'ဒုင်စသိုင်', 'ကၠာ', 'ကၠာ', 'ရ', '။']
print(result["ids"])     # [1234, 5678, ...]

# decode
decoded = tokenizer.decode(result["pieces"])
print(decoded)  # ဂွံအခေါင်အရာမွဲသ္ဂောံဒုင်စသိုင်ကၠာကၠာရ။
```

### Tokenizer in Hugging Face Format

```python
from transformers import AutoTokenizer

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("janakhpon/mon_tokenizer")

# tokenize
text = "ပ္ဍဲအခိင်မာံနဲသဵု မဒှ်ဘဝကွးဘာတက္ကသိုလ်ဂှ် ပါလုပ်ချဳဓရာင်ကၠုင်"
tokens = tokenizer(text, return_tensors="pt")
input_ids = tokens["input_ids"][0]

print("token ids:", input_ids.tolist())
print("tokens:", tokenizer.convert_ids_to_tokens(input_ids))
print("decoded:", tokenizer.decode(input_ids, skip_special_tokens=True))
```

## cli

```bash
# tokenize
mon-tokenizer "ဂွံအခေါင်အရာမွဲသ္ဂောံဒုင်စသိုင်ကၠာကၠာရ။"

# verbose output
mon-tokenizer -v "ဂွံအခေါင်အရာမွဲသ္ဂောံဒုင်စသိုင်ကၠာကၠာရ။"

# decode tokens
mon-tokenizer -d -t "▁ဂွံ,အခေါင်,အရာ,မွဲ,သ္ဂောံ,ဒုင်စသိုင်,ကၠာ,ကၠာ,ရ,။"

# interactive mode
mon-tokenizer
```

## API

- `encode(text: str)` → `{"pieces": list, "ids": list, "text": str}`
- `decode(pieces: list)` → `str`
- `decode_ids(ids: list)` → `str`
- `get_vocab_size()` → `int`
- `get_vocab()` → `dict`

## Dev Setup

```bash
git clone git@github.com:Code-Yay-Mal/mon_tokenizer.git
cd mon_tokenizer
uv sync --dev
uv run pytest

# Release workflow
uv version --bump patch
git add pyproject.toml
git commit -m "bump version"
git tag v0.1.5
git push origin main --tags
```

## Resources

- [hugging face model](https://huggingface.co/janakhpon/mon_tokenizer)

## License

MIT - do whatever you want with it.
