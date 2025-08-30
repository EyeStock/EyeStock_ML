from sentence_transformers import SentenceTransformer
import torch

import os
EMBEDDING_PATH = os.getenv("EMBEDDING_PATH")

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_PATH, device="cpu")
        if hasattr(_embedder, "tokenizer") and _embedder.tokenizer.pad_token is None:
            if _embedder.tokenizer.eos_token is not None:
                _embedder.tokenizer.pad_token = _embedder.tokenizer.eos_token
            else:
                _embedder.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return _embedder
