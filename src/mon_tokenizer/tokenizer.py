"""core tokenizer for mon text"""

from pathlib import Path
from typing import Dict, List, Union, Optional

import sentencepiece as spm


class MonTokenizer:
    
    def __init__(self, model_path: Optional[str] = None):
        """init tokenizer with model path"""
        if model_path is None:
            # use bundled model
            package_dir = Path(__file__).parent
            model_path = package_dir / "data" / "mon_tokenizer.model"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"model file not found: {model_path}")
        
        self._processor = spm.SentencePieceProcessor()
        self._processor.Load(str(model_path))
        self.model_path = model_path
    
    def encode(self, text: str) -> Dict[str, Union[List[str], List[int]]]:
        """encode text to tokens"""
        pieces = self._processor.EncodeAsPieces(text)
        ids = self._processor.EncodeAsIds(text)
        
        return {
            "pieces": pieces,
            "ids": ids,
            "text": text,
        }
    
    def decode(self, pieces: List[str]) -> str:
        """decode tokens back to text"""
        return self._processor.DecodePieces(pieces)
    
    def decode_ids(self, ids: List[int]) -> str:
        """decode token ids back to text"""
        return self._processor.DecodeIds(ids)
    
    def get_vocab_size(self) -> int:
        """get vocab size"""
        return self._processor.GetPieceSize()
    
    def get_vocab(self) -> Dict[str, int]:
        """get vocab mapping"""
        vocab = {}
        for i in range(self.get_vocab_size()):
            piece = self._processor.IdToPiece(i)
            vocab[piece] = i
        return vocab
