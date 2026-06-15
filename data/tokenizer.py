"""Tokenizer for Ralph launch track — GPT-2 BPE via tiktoken."""

from __future__ import annotations

import tiktoken

DEFAULT_ENCODING = "gpt2"
VOCAB_SIZE = 50257
EOT_TOKEN = 50256  # GPT-2's <|endoftext|>


def get_tokenizer(encoding: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding)


def tokenize_text(text: str, tokenizer: tiktoken.Encoding | None = None) -> list[int]:
    tok = tokenizer if tokenizer is not None else get_tokenizer()
    return tok.encode_ordinary(text)
