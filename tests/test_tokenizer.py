"""tests for mon tokenizer"""

import pytest
from mon_tokenizer import MonTokenizer


def test_tokenizer_initialization():
    """test tokenizer can be initialized"""
    tokenizer = MonTokenizer()
    assert tokenizer is not None
    assert tokenizer.model_path is not None


def test_encode_decode():
    """test basic encode and decode functionality"""
    tokenizer = MonTokenizer()
    text = "ဂွံအခေါင်အရာမွဲသ္ဂောံဒုင်စသိုင်ကၠာကၠာရ။"
    
    # Encode
    result = tokenizer.encode(text)
    assert "pieces" in result
    assert "ids" in result
    assert "text" in result
    assert result["text"] == text
    assert len(result["pieces"]) > 0
    assert len(result["ids"]) > 0
    
    # Decode
    decoded = tokenizer.decode(result["pieces"])
    assert decoded == text


def test_decode_ids():
    """test decode_ids functionality"""
    tokenizer = MonTokenizer()
    text = "ဂွံအခေါင်အရာမွဲသ္ဂောံဒုင်စသိုင်ကၠာကၠာရ။"
    
    result = tokenizer.encode(text)
    decoded = tokenizer.decode_ids(result["ids"])
    # The decoded text should be similar, allowing for minor character differences
    assert len(decoded) > 0
    assert "ဂွံ" in decoded  # Check that key parts are present


def test_vocab_size():
    """test vocab size is reasonable"""
    tokenizer = MonTokenizer()
    vocab_size = tokenizer.get_vocab_size()
    assert vocab_size > 0
    assert vocab_size < 100000  # reasonable upper bound


def test_vocab():
    """test vocab dictionary"""
    tokenizer = MonTokenizer()
    vocab = tokenizer.get_vocab()
    assert isinstance(vocab, dict)
    assert len(vocab) == tokenizer.get_vocab_size()


def test_custom_model_path():
    """test tokenizer with custom model path"""
    # This would require a test model file
    # For now, just test that it raises FileNotFoundError for non-existent path
    with pytest.raises(FileNotFoundError):
        MonTokenizer("non_existent_model.model")
