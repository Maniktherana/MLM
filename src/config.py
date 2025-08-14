from __future__ import annotations
import os
from pathlib import Path

TOKENIZER_SAMPLE_FRACTION = 0.20
MIN_CHARS = 200
VOCAB_SIZE = 52_000
MAX_LENGTH = 1024
BATCH_SIZE = 4
SEED = 42

GPT_CONFIG_124M = {
    "vocab_size": VOCAB_SIZE,  # will be overwritten by tokenizer.vocab_size
    "context_length": MAX_LENGTH,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


APP_NAME = "ManikLM"
VOLUME_NAME = "mlm-artifacts"


def get_paths():
    root = Path(os.environ.get("MLM_ROOT", "/vol")).resolve()
    ckpt = root / "checkpoints"
    final = root / "final"
    tok = root / "tokenizer"
    return root, ckpt, final, tok


ROOT, CKPT_DIR, FINAL_DIR, TOK_DIR = get_paths()
