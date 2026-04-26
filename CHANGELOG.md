# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-26

### Added
- Major model upgrade: Vocabulary size expanded from 4,000 to 32,000.
- 5.17x compression ratio (2.6x improvement over 0.1.x).
- Grapheme cluster atomicity via Unicode extended grapheme clusters.
- Trained on full 177M character Mon corpus.
- 100% round-trip accuracy guaranteed for Mon script.

## [0.1.0] - 2025-08-23

### Added
- Initial release of Mon tokenizer
- Core tokenization functionality with SentencePiece
- CLI interface with encode/decode capabilities
- Python API with MonTokenizer class
- Support for custom model paths
- Rich CLI output with verbose mode
