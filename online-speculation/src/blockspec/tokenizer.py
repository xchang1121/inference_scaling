"""Local tokenizer JSON only; no model code, remote code, or implicit chat template."""

from pathlib import Path


class LocalTokenizer:
    def __init__(self, directory):
        from tokenizers import Tokenizer
        self.backend = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))

    def encode(self, text, add_special_tokens=False):
        return self.backend.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=False):
        return self.backend.decode(ids, skip_special_tokens=skip_special_tokens)
