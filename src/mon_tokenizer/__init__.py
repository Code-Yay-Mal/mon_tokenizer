"""A Unigram tokenizer for Mon (mnw), Burmese and English, with full byte fallback.

Mon is not written in isolation — it mixes with Burmese constantly and English
routinely — so all three are trained on and measured separately. Anything else
(Thai, emoji, IPA, CJK) round-trips through byte fallback rather than being lost.

    from mon_tokenizer import MonTokenizer

    tokenizer = MonTokenizer()
    result = tokenizer.encode("ဂွံအခေါင်အရာမွဲ")
    assert tokenizer.decode_ids(result["ids"]) == result["text"]

`model_card()` returns the measured record for the shipped artifact: the corpus
digest it was trained on, the training config, and per-language metrics.
"""

from importlib.metadata import PackageNotFoundError, version

from .normalization import normalize_text
from .tokenizer import Encoding, MonTokenizer, default_model_path, model_card

# Read from installed metadata rather than hardcoded. This was pinned at "0.1.5"
# through the 0.2.0, 0.2.1, 0.2.2 and 0.2.3 releases, so `mon-tokenizer --version`
# reported 0.1.5 on every one of them. A version maintained by hand in a second
# place is a version that will be wrong.
try:
    __version__ = version("mon-tokenizer")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__author__ = "Code-Yay-Mal"
__email__ = "jnovaxer@gmail.com"

__all__ = [
    "Encoding",
    "MonTokenizer",
    "__version__",
    "default_model_path",
    "load_tokenizer",
    "model_card",
    "normalize_text",
]


def load_tokenizer(model_path: str | None = None) -> MonTokenizer:
    """Construct a `MonTokenizer`. The artifact is cached by path."""
    return MonTokenizer(model_path)
