from .tokenizer import MonTokenizer

__version__ = "0.1.0"
__author__ = "janakhpon"
__email__ = "jnovaxer@gmail.com"

__all__ = ["MonTokenizer"]


def load_tokenizer(model_path=None):
    """load tokenizer with default settings"""
    return MonTokenizer(model_path)
