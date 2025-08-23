#!/usr/bin/env python3
"""Example usage of mon_tokenizer"""

from mon_tokenizer import MonTokenizer

def main():
    # Initialize the tokenizer
    tokenizer = MonTokenizer()
    
    # Example Mon text
    text = "ဂွံအခေါင်အရာမွဲသ္ဂောံဒုင်စသိုင်ကၠာကၠာရ။"
    
    print(f"Original text: {text}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    
    # Encode the text
    result = tokenizer.encode(text)
    print(f"\nEncoded tokens: {result['pieces']}")
    print(f"Token IDs: {result['ids']}")
    
    # Decode back to text
    decoded = tokenizer.decode(result['pieces'])
    print(f"\nDecoded text: {decoded}")
    
    # Decode from IDs
    decoded_from_ids = tokenizer.decode_ids(result['ids'])
    print(f"Decoded from IDs: {decoded_from_ids}")
    
    # Show some vocabulary statistics
    vocab = tokenizer.get_vocab()
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"First 10 vocabulary items: {list(vocab.items())[:10]}")

if __name__ == "__main__":
    main()
