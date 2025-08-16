from __future__ import annotations
import os
from pathlib import Path

TOKENIZER_SAMPLE_FRACTION = 0.50
MIN_CHARS = 200
VOCAB_SIZE = 50_257
MAX_LENGTH = 1024
SEED = 42

BATCH_SIZE = 6
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP_NORM = 1.0
WARMUP_STEPS = 1000

DATASET_NAME = "Skylion007/openwebtext"
REVISION = "refs/convert/parquet"
TRAIN_SPLIT_FRACTION = 0.4

GPT_CONFIG_124M = {
    "vocab_size": VOCAB_SIZE,
    "context_length": MAX_LENGTH,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

TRAINING_CONFIG = {
    "num_epochs": 1,
    "eval_freq": 1000,
    "eval_iter": 15,
    "generate_freq": 10000,
    "generation_prompts": [
        "The future of artificial intelligence",
        "In a world where technology",
        "Scientists have discovered",
        "The most important thing to understand",
    ],
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
